"""v5 依頼B：JRA-VAN の生データを、既存パイプラインが読めるカラムに変換する。

jvlink.py が保存した RA.csv / SE.csv / HR.csv（構造体のフィールド名のまま・全部文字列）を読み、

    race_result 相当  … features.py 以降にそのまま渡せる DataFrame
    lap_df            … pace.attach_race_pace() に渡せるレース単位のペース表
    payout_df         … trifecta.build_race_table() に渡せる払戻表

の3つを作る。features.py / trifecta.py / validate.py は**無改修**で動く
（jvmap が列名を Kaggle 側に合わせるので、下流は違いを知らなくてよい）。

■ ここでやる5つの処理（順番に意味がある）

  1. データ区分による重複解消    同じレース・同じ馬が段階的に何度も届くので、最も確定度の高いものを残す
  2. 中央競馬への絞り込み        競馬場コード 01〜10、データ区分 A/B（地方・海外）以外
  3. 学習に使える確定度で絞る    レースもその馬も 5/6/7（全馬着順確定以降）のものだけ
  4. 異常区分での除外            出走取消・除外・中止・失格の馬を落とす
  5. 単位変換と列名の対応付け    "1345" → 94.5秒、"0123" → 12.3倍 など

■ 列名の方針
  馬・騎手・調教師は**名前ではなくコード**で識別する（同名馬・改名・文字化け対策）。
  config の候補リストで 'horse' → 血統登録番号、'jockey' → 騎手コード、
  'trainer' → 調教師コード に解決されるよう、名前の列は候補に無い名前
  （馬名表示・騎手名略称・調教師名略称）にしてある。

■ そのレースの結果そのものの列は `_` で始める
  4コーナー順位や公式脚質判定はレースの**結果**なので、そのまま特徴量にするとリークになる。
  features.add_all_features() は `_` で始まる列を最後に必ず削除するので、
  過去集計に使ったあと特徴量に紛れ込むことがない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# コード表（JV-Data仕様書 4.9.0.1「コード表」シートより）
# ---------------------------------------------------------------------------
# 2001.競馬場コード（中央 01〜10 のみ）
JYO_NAMES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
CENTRAL_JYO = set(JYO_NAMES)

# 2003.グレードコード -> Kaggle の「リステッド・重賞競走」列の表記
#   A=G1 B=G2 C=G3 D=グレードのない重賞（Kaggle の "G" に相当）
#   F/G/H = 障害の J.G1/J.G2/J.G3（Kaggle と同じ表記）、L = リステッド、E = 重賞以外の特別
# 障害重賞とリステッドは config.GRADED_VALUES に含まれないので「重賞」扱いにならない。
# 詳しくは README v5 の「グレードの対応」を参照（Kaggle 側の定義を確認する手順あり）。
GRADE_LABELS = {
    "A": "G1", "B": "G2", "C": "G3", "D": "G",
    "F": "J.G1", "G": "J.G2", "H": "J.G3",
    "L": "L",
}
# 障害重賞を平地と同じ G1/G2/G3 として数えたい場合に使う対応（既定では使わない）
GRADE_LABELS_JUMP_AS_FLAT = {**GRADE_LABELS, "F": "G1", "G": "G2", "H": "G3"}

# 2010.馬場状態コード / 2011.天候コード / 2202.性別コード
BABA_LABELS = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}
TENKO_LABELS = {"1": "晴", "2": "曇", "3": "雨", "4": "小雨", "5": "雪", "6": "小雪"}
SEX_LABELS = {"1": "牡", "2": "牝", "3": "セ"}

# 2101.異常区分コード
#   1 出走取消 / 2 発走除外 / 3 競走除外 / 4 競走中止 / 5 失格 / 6 落馬再騎乗 / 7 降着
# 1〜5 は着順が意味を持たない（走っていない・完走していない・順位が無効）ので落とす。
# Kaggle 版でも着順・タイムの欠損行を落としていたのと同じ扱い。
# 6（再騎乗して完走）と 7（降着。確定着順がある）は残す。
EXCLUDE_IJYO = {"1", "2", "3", "4", "5"}

# データ区分の優先度（大きいほど確定度が高い）。"0" は削除、"A"/"B" は地方・海外。
# 9（レース中止）は最上位に置き、届いた時点でそのレースは中止として扱う。
RACE_KUBUN_PRIORITY = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "9": 9,
                       "A": 0, "B": 0}
HR_KUBUN_PRIORITY = {"1": 1, "2": 2, "9": 9}

# 学習・検証に使ってよいデータ区分（全馬の着順が揃っているもの）
#   3/4 は「3着まで」「5着まで」しか含まないので、使うと出走頭数や着順が欠ける
USABLE_RACE_KUBUN = {"5", "6", "7"}
USABLE_HR_KUBUN = {"1", "2"}

RACE_KEY = ["id.Year", "id.MonthDay", "id.JyoCD", "id.Kaiji", "id.Nichiji", "id.RaceNum"]

# HR のフラグ配列は [単勝, 複勝, 枠連, 馬連, ワイド, 予備, 馬単, 3連複, 3連単] の9要素。
# jvlink は列名を1始まりで付けているので、3連複=[8]、3連単=[9]。
FLAG_INDEX = {"3連複": 8, "3連単": 9}


# ---------------------------------------------------------------------------
# トラックコード（2009）-> 芝ダ・回り・内外
# ---------------------------------------------------------------------------
def track_surface(code: str) -> str | None:
    """10〜22 芝 / 23〜29 ダート（27,28 はサンド）/ 51〜59 障害。"""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if 10 <= c <= 22:
        return "芝"
    if c in (23, 24, 25, 26, 29):
        return "ダート"
    if c in (27, 28):
        return "サンド"   # 中央では使われない（海外・地方用）
    if 51 <= c <= 59:
        return "障害"
    return None


_LEFT = {11, 12, 13, 14, 15, 16, 23, 25, 27, 53}
_RIGHT = {17, 18, 19, 20, 21, 22, 24, 26, 28}
_STRAIGHT = {10, 29}


def track_turn(code: str) -> str | None:
    """右 / 左 / 直線。障害は仕様上ほとんど回りが入っていない（53 のみ左）。"""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c in _STRAIGHT:
        return "直線"
    if c in _LEFT:
        return "左"
    if c in _RIGHT:
        return "右"
    return None


_OUTER = {12, 16, 18, 22, 26, 55, 59}
_IN_TO_OUT = {13, 19, 57}
_OUT_TO_IN = {14, 20, 56}


def track_inout(code: str) -> str | None:
    """内 / 外 / 内→外 / 外→内。直線と障害の一部は None。"""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c in _OUTER:
        return "外"
    if c in _IN_TO_OUT:
        return "内→外"
    if c in _OUT_TO_IN:
        return "外→内"
    if c in _STRAIGHT:
        return None
    return "内"


# ---------------------------------------------------------------------------
# 数値の変換（固定長の数字文字列 -> 数値）
# ---------------------------------------------------------------------------
def to_int(s: pd.Series, invalid: tuple[str, ...] = ()) -> pd.Series:
    """数字文字列を整数に。空白・無効値（"000" など）は NaN。"""
    s = s.astype(str).str.strip()
    s = s.where(~s.isin(set(invalid) | {""}))
    return pd.to_numeric(s, errors="coerce")


def to_tenths(s: pd.Series, invalid: tuple[str, ...] = ()) -> pd.Series:
    """"345" -> 34.5 のように、末尾1桁が小数第1位の数値（99.9秒・999.9倍・0.1kg）。"""
    return to_int(s, invalid) / 10.0


def to_race_time(s: pd.Series) -> pd.Series:
    """走破タイム "1345"（9分99秒9 形式）-> 94.5 秒。"0000" は NaN。"""
    v = to_int(s, invalid=("0000",))
    minutes = v // 1000
    seconds_tenths = v % 1000
    return minutes * 60 + seconds_tenths / 10.0


def to_signed_diff(sign: pd.Series, diff: pd.Series) -> pd.Series:
    """馬体重の増減。符号が別フィールドなので結合する。

    増減差: "999" = 計量不能 → NaN、"000" = 前差なし → 0、空白 = 初出走 → NaN。
    符号: "+" / "-" / 空白。空白で差が 0 以外のときは符号不明なので NaN。
    """
    d = to_int(diff, invalid=("999",))
    sgn = sign.astype(str).str.strip().map({"+": 1.0, "-": -1.0})
    out = d * sgn
    return out.where(~((d == 0) & sgn.isna()), 0.0)


# ---------------------------------------------------------------------------
# 読み込みと重複解消
# ---------------------------------------------------------------------------
def load_raw(data_dir: str | os.PathLike | None = None,
             record_types: tuple[str, ...] = ("RA", "SE", "HR")) -> dict[str, pd.DataFrame]:
    """jvlink.py が保存した CSV を文字列のまま読む（先頭ゼロを壊さないため）。"""
    base = Path(data_dir or config.JV_DATA_DIR)
    out = {}
    for rt in record_types:
        path = base / f"{rt}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} がありません。先に `py -m src.jvlink fetch` を実行してください")
        out[rt] = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return out


def _make_date_key(df: pd.DataFrame) -> pd.Series:
    """レコードヘッダーの「データ作成年月日」を YYYYMMDD の整数にする（無ければ 0）。"""
    cols = ["head.MakeDate.Year", "head.MakeDate.Month", "head.MakeDate.Day"]
    if not all(c in df.columns for c in cols):
        return pd.Series(0, index=df.index)
    text = df[cols[0]].astype(str).str.zfill(4) + df[cols[1]].astype(str).str.zfill(2) \
        + df[cols[2]].astype(str).str.zfill(2)
    return pd.to_numeric(text, errors="coerce").fillna(0).astype("int64")


def apply_data_kubun(df: pd.DataFrame, key_cols: list[str],
                     priority: dict[str, int]) -> pd.DataFrame:
    """データ区分のルールで重複を解消し、1キー1行にする。

    **データ作成年月日の古い順**（同じ日なら取得した順 _seq）に1行ずつ見て:
      - 区分 "0"（該当レコード削除）→ そのキーの既存行を消す
      - それ以外 → 既存より優先度が**同じか高ければ**置き換える
                  （同じ優先度なら作成日の新しい方＝訂正版を採る）

    なぜ取得した順（_seq）だけで並べないのか:
      取得の順番は提供の順番と一致しない。例えば直近1ヶ月を先に取ってから
      全期間のセットアップをすると、「1987年のレースの訂正版（今月作成）」が先に入り、
      「1987年のレースの元の版（1987年作成）」が後から届く。取得順で並べると、
      同じ区分7どうしなので古い元の版が訂正版を上書きしてしまう。
      ヘッダーのデータ作成年月日で並べれば、取得の順番に関係なく新しい版が勝つ。

    なぜ「作成日が最新のものを残す」だけにしないのか:
      確定(7)の後に、作成日の新しい速報(3)が別ファイルで届くことがありうる。
      確定版を速報で上書きしないよう、優先度でも比べる。
    """
    if df.empty:
        return df
    order = pd.DataFrame({
        "date": _make_date_key(df).to_numpy(),
        "seq": pd.to_numeric(df["_seq"], errors="coerce").fillna(-1).to_numpy(),
    })
    df = df.iloc[np.lexsort((order["seq"].to_numpy(), order["date"].to_numpy()))]
    keys = list(zip(*(df[c].to_numpy() for c in key_cols)))
    kubun = df["head.DataKubun"].to_numpy()

    kept: dict[tuple, int] = {}       # キー -> 行位置
    kept_pri: dict[tuple, int] = {}   # キー -> 優先度
    for pos, (key, k) in enumerate(zip(keys, kubun)):
        if k == "0":
            kept.pop(key, None)
            kept_pri.pop(key, None)
            continue
        pri = priority.get(k, -1)
        if pri < 0:
            continue  # 仕様に無い区分は無視
        if key not in kept or pri >= kept_pri[key]:
            kept[key] = pos
            kept_pri[key] = pri

    return df.iloc[sorted(kept.values())].reset_index(drop=True)


def make_race_id(df: pd.DataFrame) -> pd.Series:
    """Kaggle 形式の12桁レースID（年4 + 場2 + 回2 + 日2 + R2）。月日は含めない。"""
    return (df["id.Year"].str.zfill(4) + df["id.JyoCD"].str.zfill(2)
            + df["id.Kaiji"].str.zfill(2) + df["id.Nichiji"].str.zfill(2)
            + df["id.RaceNum"].str.zfill(2))


def make_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["id.Year"] + df["id.MonthDay"], format="%Y%m%d", errors="coerce")


def central_only(df: pd.DataFrame) -> pd.DataFrame:
    """中央競馬だけを残す（競馬場コード 01〜10 かつ データ区分が A/B でない）。"""
    mask = df["id.JyoCD"].isin(CENTRAL_JYO) & ~df["head.DataKubun"].isin({"A", "B"})
    return df.loc[mask]


# ---------------------------------------------------------------------------
# 変換の本体
# ---------------------------------------------------------------------------
@dataclass
class BuildReport:
    """各段階で何行残ったか。数字が想定外なら、どこで落ちたかを追える。"""

    steps: list = field(default_factory=list)

    def add(self, what: str, n: int) -> None:
        self.steps.append((what, n))

    def note(self, text: str) -> None:
        """件数ではない補足（内訳など）を1行足す。"""
        self.steps.append((text, None))

    def __str__(self) -> str:
        return "\n".join(f"  {what:<40} {n:>12,}" if n is not None else f"      └ {what}"
                         for what, n in self.steps)


def prepare_races(ra: pd.DataFrame, report: BuildReport | None = None,
                  jump_as_flat: bool = False) -> pd.DataFrame:
    """RA を1レース1行にし、学習に使えるレースだけ残してラベルを付ける。"""
    report = report or BuildReport()
    report.add("RA 生レコード", len(ra))
    ra = apply_data_kubun(ra, RACE_KEY, RACE_KUBUN_PRIORITY)
    report.add("RA 重複解消後（1レース1行）", len(ra))
    ra = central_only(ra)
    report.add("RA 中央競馬のみ", len(ra))
    cancelled = int((ra["head.DataKubun"] == "9").sum())
    ra = ra.loc[ra["head.DataKubun"].isin(USABLE_RACE_KUBUN)]
    report.add(f"RA 全馬着順確定(5/6/7)のみ（中止{cancelled}件を除外）", len(ra))

    return map_races(ra, jump_as_flat=jump_as_flat)


def map_races(ra: pd.DataFrame, jump_as_flat: bool = False) -> pd.DataFrame:
    """RA（1レース1行にしたもの）を、既存パイプラインの列名に対応付ける。

    確定したレース（prepare_races）と出馬表段階のレース（prepare_entries）で共通。
    """
    labels = GRADE_LABELS_JUMP_AS_FLAT if jump_as_flat else GRADE_LABELS
    out = pd.DataFrame({
        "レースID": make_race_id(ra).to_numpy(),
        "レース日付": make_date(ra).to_numpy(),
        "競馬場コード": ra["id.JyoCD"].to_numpy(),
        "競馬場名": ra["id.JyoCD"].map(JYO_NAMES).to_numpy(),
        "競走名": ra["RaceInfo.Hondai"].to_numpy(),
        "競走名略称": ra["RaceInfo.Ryakusyo10"].to_numpy(),
        "グレードコード": ra["GradeCD"].to_numpy(),
        "リステッド・重賞競走": ra["GradeCD"].map(labels).to_numpy(),
        "距離(m)": to_int(ra["Kyori"]).to_numpy(),
        "トラックコード": ra["TrackCD"].to_numpy(),
        "芝・ダート区分": ra["TrackCD"].map(track_surface).to_numpy(),
        "右左回り・直線区分": ra["TrackCD"].map(track_turn).to_numpy(),
        "内外区分": ra["TrackCD"].map(track_inout).to_numpy(),
        "天候": ra["TenkoBaba.TenkoCD"].map(TENKO_LABELS).to_numpy(),
        "公式出走頭数": to_int(ra["SyussoTosu"], invalid=("00",)).to_numpy(),
        "発走時刻": ra["HassoTime"].to_numpy(),
        "_RAデータ区分": ra["head.DataKubun"].to_numpy(),
    })
    _attach_three_furlongs(out, ra)
    # 馬場状態は芝なら芝の、ダート・障害なら該当する方を使う
    siba = ra["TenkoBaba.SibaBabaCD"].map(BABA_LABELS).to_numpy()
    dirt = ra["TenkoBaba.DirtBabaCD"].map(BABA_LABELS).to_numpy()
    out["馬場状態1"] = np.where(out["芝・ダート区分"].eq("ダート"), dirt, siba)
    # 障害で芝の馬場状態が空ならダート側で補う
    out["馬場状態1"] = pd.Series(out["馬場状態1"]).where(
        pd.Series(out["馬場状態1"]).notna(), pd.Series(dirt)).to_numpy()
    return out


def _attach_three_furlongs(out: pd.DataFrame, ra: pd.DataFrame) -> None:
    """レース単位の前3F・後3Fを付ける。公式値が空ならラップタイムから計算して補う。

    公式値（HaronTimeS3/L3）は古い年に入っていないことがある。仕様書の定義では
      前3ハロン = ラップの前半3本の合計（200mで割り切れない距離は、最初の1本が端数）
      後3ハロン = ラップの後半3本の合計
    なので、ラップがあれば同じ量を計算できる（Kaggle 版でやっていた計算と同じ）。
    どちらから取ったかは `_3F出所`（公式 / ラップ / なし）に残す。
    平地のみ（障害はラップも3Fも入らない）。
    """
    from . import pace as pace_mod

    official_s3 = to_tenths(ra["HaronTimeS3"], invalid=("000", "999")).to_numpy()
    official_l3 = to_tenths(ra["HaronTimeL3"], invalid=("000", "999")).to_numpy()

    lap_cols = [f"LapTime[{i}]" for i in range(1, 26)]
    if all(c in ra.columns for c in lap_cols):
        laps = pd.DataFrame({"レースID": out["レースID"].to_numpy()})
        for i, c in enumerate(lap_cols, start=1):
            laps[f"ラップタイム{i}"] = to_tenths(ra[c], invalid=("000", "999")).to_numpy()
        from_laps = pace_mod.compute_race_pace(laps)
        lap_s3 = from_laps["前半3F"].to_numpy(dtype="float64")
        lap_l3 = from_laps["上がり3F"].to_numpy(dtype="float64")
    else:
        lap_s3 = lap_l3 = np.full(len(out), np.nan)

    has_official = ~np.isnan(official_s3) & ~np.isnan(official_l3)
    has_laps = ~np.isnan(lap_s3) & ~np.isnan(lap_l3)
    out["_前3F"] = np.where(has_official, official_s3, lap_s3)
    out["_後3F"] = np.where(has_official, official_l3, lap_l3)
    out["_3F出所"] = np.select([has_official, has_laps], ["公式", "ラップ"], default="なし")
    # 公式値とラップからの計算値を別々にも残す（両方あるレースで一致を確かめるため。check-3f）
    out["_前3F公式"] = official_s3
    out["_後3F公式"] = official_l3
    out["_前3F計算"] = lap_s3
    out["_後3F計算"] = lap_l3
    out["_ラップ1本目"] = laps["ラップタイム1"].to_numpy(dtype="float64") if all(
        c in ra.columns for c in lap_cols) else np.full(len(out), np.nan)


def prepare_horses(se: pd.DataFrame, races: pd.DataFrame,
                   report: BuildReport | None = None) -> pd.DataFrame:
    """SE を1頭1行にし、レース情報と結合して race_result 相当にする。"""
    report = report or BuildReport()
    report.add("SE 生レコード", len(se))
    se = apply_data_kubun(se, RACE_KEY + ["KettoNum"], RACE_KUBUN_PRIORITY)
    report.add("SE 重複解消後（1レース1頭1行）", len(se))
    se = central_only(se)
    report.add("SE 中央競馬のみ", len(se))
    se = se.loc[se["head.DataKubun"].isin(USABLE_RACE_KUBUN)]
    report.add("SE 全馬着順確定(5/6/7)のみ", len(se))

    horses = map_horses(se)

    # レース情報と結合（学習に使えるレースの馬だけが残る）
    merged = horses.merge(races, on="レースID", how="inner")
    report.add("SE レース情報と結合後", len(merged))

    excluded = merged["異常区分"].isin(EXCLUDE_IJYO)
    report.add(f"SE 取消・除外・中止・失格を除外（{int(excluded.sum()):,}頭）", int((~excluded).sum()))
    merged = merged.loc[~excluded]

    merged = merged.loc[merged["着順"].notna()]
    report.add("SE 着順あり", len(merged))
    return merged.reset_index(drop=True)


def map_horses(se: pd.DataFrame) -> pd.DataFrame:
    """SE（1レース1頭1行にしたもの）を、既存パイプラインの列名に対応付ける。

    確定したレース（prepare_horses）と出馬表段階のレース（prepare_entries）で共通。
    """
    return pd.DataFrame({
        "レースID": make_race_id(se).to_numpy(),
        "枠番": to_int(se["Wakuban"], invalid=("0",)).to_numpy(),
        "馬番": to_int(se["Umaban"], invalid=("00",)).to_numpy(),
        "血統登録番号": se["KettoNum"].to_numpy(),
        "馬名表示": se["Bamei"].to_numpy(),
        "性別": se["SexCD"].map(SEX_LABELS).to_numpy(),
        "馬齢": to_int(se["Barei"], invalid=("00",)).to_numpy(),
        "調教師コード": se["ChokyosiCode"].to_numpy(),
        "調教師名略称": se["ChokyosiRyakusyo"].to_numpy(),
        "騎手コード": se["KisyuCode"].to_numpy(),
        "騎手名略称": se["KisyuRyakusyo"].to_numpy(),
        "斤量": to_tenths(se["Futan"], invalid=("000",)).to_numpy(),
        # 馬体重: 999=計量不能 / 000=出走取消
        "馬体重": to_int(se["BaTaijyu"], invalid=("000", "999")).to_numpy(),
        "場体重増減": to_signed_diff(se["ZogenFugo"], se["ZogenSa"]).to_numpy(),
        "異常区分": se["IJyoCD"].to_numpy(),
        "着順": to_int(se["KakuteiJyuni"], invalid=("00",)).to_numpy(),
        "タイム": to_race_time(se["Time"]).to_numpy(),
        "単勝オッズ": to_tenths(se["Odds"], invalid=("0000", "----", "****")).to_numpy(),
        "人気": to_int(se["Ninki"], invalid=("00", "--", "**")).to_numpy(),
        # 馬ごとの上がり3F。古いデータは後4Fだけ入って後3Fが初期値のことがある
        # （仕様書）。後4Fで代用はしない（別の量なので）。欠損のまま扱う。
        "後３Ｆタイム": to_tenths(se["HaronTimeL3"], invalid=("000", "999")).to_numpy(),
        "タイム差": se["TimeDiff"].to_numpy(),
        # ↓ そのレース自身の結果。特徴量にはせず、過去集計の材料にだけ使う
        "_1コーナー順位": to_int(se["Jyuni1c"], invalid=("00",)).to_numpy(),
        "_2コーナー順位": to_int(se["Jyuni2c"], invalid=("00",)).to_numpy(),
        "_3コーナー順位": to_int(se["Jyuni3c"], invalid=("00",)).to_numpy(),
        "_4コーナー順位": to_int(se["Jyuni4c"], invalid=("00",)).to_numpy(),
        "_脚質判定": to_int(se["KyakusituKubun"], invalid=("0",)).to_numpy(),
        # JRA-VAN のデータマイニング予想（他人のモデルの予測）。初回は特徴量に入れない
        "DM_予想タイム": se["DMTime"].to_numpy(),
        "DM_予想順位": to_int(se["DMJyuni"], invalid=("00",)).to_numpy(),
        "_SEデータ区分": se["head.DataKubun"].to_numpy(),
        "_SE作成日": _make_date_key(se).to_numpy(),
    })


# 出馬表段階のデータ区分（仕様書: 1=出走馬名表(木曜) 2=出馬表(金・土曜)）
ENTRY_KUBUN = {"1", "2"}
ENTRY_KUBUN_LABELS = {"1": "出走馬名表", "2": "出馬表"}
# 出馬表の段階で除外すべき異常区分（出走取消・発走除外・競走除外）
EXCLUDE_IJYO_ENTRY = {"1", "2", "3"}


def prepare_entries(raw: dict[str, pd.DataFrame], from_date: str | None = None,
                    to_date: str | None = None, graded_only: bool = True,
                    jump_as_flat: bool = False) -> pd.DataFrame:
    """出馬表段階（データ区分 1・2）のレースを、予測の入力の形にする。

    確定したレース（5/6/7）とは別に、まだ結果の無いレースを取り出す。
    データ区分の重複解消は確定レースと同じ（作成日順・確定度順）なので、
    木曜の出走馬名表(1)の後に金土の出馬表(2)が届けば、出馬表の方が残る。
    同じレースに速報（3以上）が届いていれば、そのレースは結果が出始めているので対象外。

    出走取消・除外（異常区分 1〜3）が出馬表のデータに入っていれば、その馬は外す。
    データを取り直して予測をやり直せば、新しい取消・乗り替わりが反映される。

    返す列は prepare_horses と同じ（結果の列は空）に加えて:
      データ区分（出走馬名表 / 出馬表）、データ作成日、枠順確定（馬番が全頭そろっているか）
    """
    ra = apply_data_kubun(raw["RA"], RACE_KEY, RACE_KUBUN_PRIORITY)
    ra = central_only(ra)
    ra = ra.loc[ra["head.DataKubun"].isin(ENTRY_KUBUN)]
    races = map_races(ra, jump_as_flat=jump_as_flat)
    if from_date is not None:
        races = races.loc[races["レース日付"] >= pd.Timestamp(from_date)]
    if to_date is not None:
        races = races.loc[races["レース日付"] <= pd.Timestamp(to_date)]
    if graded_only:
        races = races.loc[races["リステッド・重賞競走"].isin(config.GRADED_VALUES)]
    if races.empty:
        return pd.DataFrame()

    se = apply_data_kubun(raw["SE"], RACE_KEY + ["KettoNum"], RACE_KUBUN_PRIORITY)
    se = central_only(se)
    se = se.loc[se["head.DataKubun"].isin(ENTRY_KUBUN)]
    horses = map_horses(se).merge(races, on="レースID", how="inner")
    horses = horses.loc[~horses["異常区分"].isin(EXCLUDE_IJYO_ENTRY)]

    g = horses.groupby("レースID", sort=False)
    horses["データ区分"] = horses["_RAデータ区分"].map(ENTRY_KUBUN_LABELS)
    horses["データ作成日"] = pd.to_datetime(
        g["_SE作成日"].transform("max").astype(str), format="%Y%m%d", errors="coerce")
    horses["枠順確定"] = g["馬番"].transform(lambda s: bool(s.notna().all()))
    return horses.sort_values(["レース日付", "レースID", "馬番"]).reset_index(drop=True)


def prepare_pace(races: pd.DataFrame) -> pd.DataFrame:
    """pace.attach_race_pace() が期待する形のレース単位ペース表を作る。

    JRA-VAN の公式値（RA の前3ハロン・後3ハロン）を使い、空のレースだけ
    ラップから計算した値で補う（_attach_three_furlongs）。
    """
    out = pd.DataFrame({
        "レースID": races["レースID"].to_numpy(),
        "前半3F": races["_前3F"].astype("float32").to_numpy(),
        "上がり3F": races["_後3F"].astype("float32").to_numpy(),
    })
    out["ペース指標"] = (out["前半3F"] - out["上がり3F"]).astype("float32")
    return out


def prepare_payout(hr: pd.DataFrame, report: BuildReport | None = None,
                   exclude_irregular: bool = False) -> pd.DataFrame:
    """trifecta.build_race_table() が期待する払戻表（100円あたり）を作る。

    exclude_irregular=True なら、その券種で不成立・特払・返還があったレースの払戻を NaN にする
    （検証から外れる）。既定は False（除外しない）。v4（Kaggle）では除外していなかったので、
    比較の条件を揃えるため（開発者の判断）。
    なお不成立・特払で組番が "000000" のときは、除外の設定に関係なく NaN になる（当たり目が無い）。
    同着で的中組が複数ある場合は、1番目の組の払戻を使う（Kaggle 版も1列だった）。
    """
    report = report or BuildReport()
    report.add("HR 生レコード", len(hr))
    hr = apply_data_kubun(hr, RACE_KEY, HR_KUBUN_PRIORITY)
    hr = hr.loc[hr["id.JyoCD"].isin(CENTRAL_JYO) & hr["head.DataKubun"].isin(USABLE_HR_KUBUN)]
    report.add("HR 重複解消・中央・確定のみ", len(hr))

    out = pd.DataFrame({"レースID": make_race_id(hr).to_numpy(),
                        "_HR日付": make_date(hr).to_numpy()})
    for kind, prefix in [("3連単", "PaySanrentan"), ("3連複", "PaySanrenpuku")]:
        kumi = hr[f"{prefix}[1].Kumi"].astype(str).str.strip()
        no_kumi = kumi.isin({"", "000000"})       # 当たり目が無い（発売なし・不成立・特払）
        pay = to_int(hr[f"{prefix}[1].Pay"]).where(~no_kumi)
        n_hits = sum((hr[f"{prefix}[{i}].Kumi"].astype(str).str.strip()
                      .pipe(lambda s: ~s.isin({"", "000000"}))).astype(int)
                     for i in range(1, 4))
        i = FLAG_INDEX[kind]
        fuseiritu_tokubarai = ((hr[f"FuseirituFlag[{i}]"] == "1")
                               | (hr[f"TokubaraiFlag[{i}]"] == "1"))
        irregular = fuseiritu_tokubarai | (hr[f"HenkanFlag[{i}]"] == "1")
        if exclude_irregular:
            report.add(f"HR {kind} 不成立・特払・返還で除外", int(irregular.sum()))
            pay = pay.where(~irregular)
        else:
            report.add(f"HR {kind} 不成立・特払・返還あり（除外しない）", int(irregular.sum()))
        out[f"{kind}払戻"] = pay.astype("float64").to_numpy()
        out[f"{kind}的中組数"] = np.asarray(n_hits)
        # 払戻が無い理由。発売開始日などは決め打ちせず、HR の中身で判定する:
        #   組番が空で不成立・特払フラグも無い → その券種が発売されていない（3連単は発売開始前など）
        #   組番が空で不成立・特払フラグがある → 不成立・特払
        #   組番はあるが除外の設定で外した     → 返還など（--exclude-irregular のとき）
        out[f"_{kind}払戻なし理由"] = np.select(
            [no_kumi & ~fuseiritu_tokubarai, no_kumi & fuseiritu_tokubarai,
             (~no_kumi) & irregular & exclude_irregular],
            ["発売なし", "不成立・特払", "返還など（除外の設定）"], default="")
    return out


@dataclass
class JVDataset:
    race_result: pd.DataFrame
    lap_df: pd.DataFrame
    payout_df: pd.DataFrame
    report: BuildReport

    def summary(self) -> str:
        rr = self.race_result
        lines = [
            "=== JRA-VAN データセット ===",
            str(self.report),
            f"  期間: {rr['レース日付'].min():%Y-%m-%d} 〜 {rr['レース日付'].max():%Y-%m-%d}",
            f"  レース数: {rr['レースID'].nunique():,} / 頭数: {len(rr):,}",
            "  格付け（レース数）: "
            + str(rr.drop_duplicates('レースID')['リステッド・重賞競走']
                  .value_counts(dropna=False).to_dict()),
            f"  3連単払戻あり: {self.payout_df['3連単払戻'].notna().sum():,} レース（最終レースのうち）",
        ]
        return "\n".join(lines)


# build の既定の開始日。v4（Kaggle、1986年1月〜）と条件を揃える。
# JRA-VAN には1986年より前の重賞の記録などが少し入っているが、払戻（HR）は1986年からしかなく、
# 混ぜると騎手・調教師の通算成績に昔の大レースだけが入ってしまう。生データには残す。
DEFAULT_START = "1986-01-01"


def build_dataset(data_dir: str | os.PathLike | None = None,
                  raw: dict[str, pd.DataFrame] | None = None,
                  start: str | None = DEFAULT_START, end: str | None = None,
                  jump_as_flat: bool = False,
                  exclude_irregular_payout: bool = False) -> JVDataset:
    """生 CSV から、既存パイプラインに渡せる3つの表を作る。

    start / end（"YYYY-MM-DD"）でレース日を絞る。既定は 1986-01-01 以降（DEFAULT_START）。
    None にすると絞らない。絞るのはレース（RA）の段階なので、範囲外のレースの馬も
    一緒に外れ、過去成績の特徴量にも入らない。**生データ（RA.csv / SE.csv）は変更しない。**
    ※ 学習期間を絞りたいだけなら、ここでは絞らず validate の fold で切ること
      （ここで絞ると、その前の成績が通算成績に入らなくなる）。
    """
    raw = raw or load_raw(data_dir)
    report = BuildReport()

    races = prepare_races(raw["RA"], report, jump_as_flat=jump_as_flat)

    # レース日での絞り込み（ここで切れば、範囲外のレースの馬も prepare_horses で外れる）
    for bound, label, keep in [
        (start, "開始日", lambda d, b: d >= b),
        (end, "終了日", lambda d, b: d <= b),
    ]:
        if bound is None:
            continue
        mask = keep(races["レース日付"], pd.Timestamp(bound))
        out_of_range = races.loc[~mask]
        report.add(f"RA {label} {bound} の範囲外を除外（生データには残す）", len(out_of_range))
        if len(out_of_range):
            report.note("開催年の内訳: " + _year_counts(out_of_range["レース日付"]))
        races = races.loc[mask]
    report.add("RA 期間内のレース", len(races))

    horses = prepare_horses(raw["SE"], races, report)
    payout = prepare_payout(raw["HR"], report, exclude_irregular=exclude_irregular_payout)

    # RA はあるのに、使える馬が1頭も残らなかったレース（＝ここで落ちる）
    with_horses = set(horses["レースID"])
    dropped = races.loc[~races["レースID"].isin(with_horses)]
    report.add("RA のうち馬のデータが無く除外したレース", len(dropped))
    if len(dropped):
        report.note("開催年の内訳: " + _year_counts(dropped["レース日付"]))
        report.note("SE が届いていない（過去レースの訂正で RA だけ届いた等）か、"
                    "全馬が取消・除外・中止だったレース")
    races = races.loc[races["レースID"].isin(with_horses)]
    report.add("最終レース数", len(races))

    # 3F の出所（公式 / ラップから計算 / なし）
    src = races["_3F出所"].value_counts().to_dict()
    report.note(f"前3F・後3Fの出所: 公式 {src.get('公式', 0):,} / ラップから計算 {src.get('ラップ', 0):,}"
                f" / なし {src.get('なし', 0):,}（障害はもともと無い）")

    # 払戻の有無の内訳（検証に使えるのは 3連単払戻があるレースだけ）
    pay = payout.set_index("レースID")
    ids = races["レースID"]
    in_hr = ids.isin(pay.index)
    sub = pay.reindex(ids[in_hr])
    has_pay = int(sub["3連単払戻"].notna().sum())
    report.add("最終レースのうち 3連単払戻あり", has_pay)
    reasons = sub.loc[sub["3連単払戻"].isna(), "_3連単払戻なし理由"]
    no_hr = int((~in_hr).sum())
    not_sold = reasons == "発売なし"
    report.note(f"払戻なしの内訳: HR が未着 {no_hr:,} / 3連単の発売なし {int(not_sold.sum()):,}"
                f" / 不成立・特払 {int((reasons == '不成立・特払').sum()):,}"
                f" / 返還など（除外の設定） {int((reasons == '返還など（除外の設定）').sum()):,}")
    if not_sold.any():
        dates = pd.to_datetime(sub.loc[reasons.index[not_sold], "_HR日付"])
        report.note("3連単の発売なしの開催年: " + _year_counts(dates))
    sold = sub.loc[sub["3連単払戻"].notna(), "_HR日付"]
    if len(sold):
        report.note(f"3連単の払戻がある最初のレース: {pd.to_datetime(sold).min():%Y-%m-%d}")

    horses = horses.loc[horses["レースID"].isin(set(races["レースID"]))]
    return JVDataset(
        race_result=horses.reset_index(drop=True),
        lap_df=prepare_pace(races),
        payout_df=payout.loc[payout["レースID"].isin(set(races["レースID"]))].reset_index(drop=True),
        report=report,
    )


def _year_counts(dates: pd.Series, max_items: int = 50) -> str:
    """{年: 件数} の文字列。年が多すぎるときは範囲だけにする。"""
    counts = pd.to_datetime(dates).dt.year.value_counts().sort_index()
    if len(counts) > max_items:
        return f"{counts.index.min()}〜{counts.index.max()}（{len(counts)}年、計{int(counts.sum()):,}）"
    return str({int(k): int(v) for k, v in counts.items()})


# 年ごとの欠損率を見る列（race_result の列名）
COVERAGE_HORSE_COLUMNS = ["_1コーナー順位", "_2コーナー順位", "_3コーナー順位", "_4コーナー順位",
                          "_脚質判定", "馬体重", "後３Ｆタイム", "単勝オッズ"]


def coverage_by_year(ds: "JVDataset") -> pd.DataFrame:
    """主要な列の年ごとの欠損率（%）。何年から使えるデータかを確かめるため。

    前3F・後3F はレース単位で、障害はもともと無いので**平地のレースだけ**で数える。
    `3F_ラップ補完` は、公式値が空でラップから計算したレースの割合。
    コーナー順位・脚質判定・馬体重などは馬単位。
    1・2コーナーは短い距離だと通らないので、欠損が多くても異常ではない。
    """
    rr = ds.race_result
    year = rr["レース日付"].dt.year
    horse = rr[COVERAGE_HORSE_COLUMNS].isna().groupby(year).mean() * 100

    races = rr.drop_duplicates("レースID")
    flat = races.loc[races["芝・ダート区分"].isin(["芝", "ダート"])]
    fy = flat["レース日付"].dt.year
    race_part = pd.DataFrame({
        "_前3F": flat["_前3F"].isna().groupby(fy).mean() * 100,
        "_後3F": flat["_後3F"].isna().groupby(fy).mean() * 100,
        "3F_ラップ補完": (flat["_3F出所"] == "ラップ").groupby(fy).mean() * 100,
    })
    out = race_part.join(horse, how="outer")
    out.insert(0, "頭数", year.value_counts().sort_index())
    out.index.name = "年"
    return out.round(1)


def grade_count_by_year(race_result: pd.DataFrame) -> pd.DataFrame:
    """年 × 格付けのレース数（Kaggle 側の数え方と突き合わせるため）。"""
    races = race_result.drop_duplicates("レースID")
    return pd.crosstab(races["レース日付"].dt.year,
                       races["リステッド・重賞競走"].fillna("平場"))
