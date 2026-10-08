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
        # レース単位の公式3ハロン（平地のみ。障害は初期値 000 → NaN）
        "_前3F": to_tenths(ra["HaronTimeS3"], invalid=("000", "999")).to_numpy(),
        "_後3F": to_tenths(ra["HaronTimeL3"], invalid=("000", "999")).to_numpy(),
    })
    # 馬場状態は芝なら芝の、ダート・障害なら該当する方を使う
    siba = ra["TenkoBaba.SibaBabaCD"].map(BABA_LABELS).to_numpy()
    dirt = ra["TenkoBaba.DirtBabaCD"].map(BABA_LABELS).to_numpy()
    out["馬場状態1"] = np.where(out["芝・ダート区分"].eq("ダート"), dirt, siba)
    # 障害で芝の馬場状態が空ならダート側で補う
    out["馬場状態1"] = pd.Series(out["馬場状態1"]).where(
        pd.Series(out["馬場状態1"]).notna(), pd.Series(dirt)).to_numpy()
    return out


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

    horses = pd.DataFrame({
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
    })

    # レース情報と結合（学習に使えるレースの馬だけが残る）
    merged = horses.merge(races, on="レースID", how="inner")
    report.add("SE レース情報と結合後", len(merged))

    excluded = merged["異常区分"].isin(EXCLUDE_IJYO)
    report.add(f"SE 取消・除外・中止・失格を除外（{int(excluded.sum()):,}頭）", int((~excluded).sum()))
    merged = merged.loc[~excluded]

    merged = merged.loc[merged["着順"].notna()]
    report.add("SE 着順あり", len(merged))
    return merged.reset_index(drop=True)


def prepare_pace(races: pd.DataFrame) -> pd.DataFrame:
    """pace.attach_race_pace() が期待する形のレース単位ペース表を作る。

    Kaggle 版はラップから前半3F・上がり3Fを計算し直していたが、
    JRA-VAN は公式値（RA の前3ハロン・後3ハロン）がそのまま入っているので使う。
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

    out = pd.DataFrame({"レースID": make_race_id(hr).to_numpy()})
    for kind, prefix in [("3連単", "PaySanrentan"), ("3連複", "PaySanrenpuku")]:
        kumi = hr[f"{prefix}[1].Kumi"].astype(str).str.strip()
        pay = to_int(hr[f"{prefix}[1].Pay"])
        pay = pay.where(~kumi.isin({"", "000000"}))  # 発売なし・特払・不成立
        n_hits = sum((hr[f"{prefix}[{i}].Kumi"].astype(str).str.strip()
                      .pipe(lambda s: ~s.isin({"", "000000"}))).astype(int)
                     for i in range(1, 4))
        i = FLAG_INDEX[kind]
        irregular = ((hr[f"FuseirituFlag[{i}]"] == "1") | (hr[f"TokubaraiFlag[{i}]"] == "1")
                     | (hr[f"HenkanFlag[{i}]"] == "1"))
        if exclude_irregular:
            report.add(f"HR {kind} 不成立・特払・返還で除外", int(irregular.sum()))
            pay = pay.where(~irregular)
        else:
            report.add(f"HR {kind} 不成立・特払・返還あり（除外しない）", int(irregular.sum()))
        out[f"{kind}払戻"] = pay.astype("float64").to_numpy()
        out[f"{kind}的中組数"] = np.asarray(n_hits)
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
            f"  3連単払戻あり: {self.payout_df['3連単払戻'].notna().sum():,} レース",
        ]
        return "\n".join(lines)


def build_dataset(data_dir: str | os.PathLike | None = None,
                  raw: dict[str, pd.DataFrame] | None = None,
                  start: str | None = None, end: str | None = None,
                  jump_as_flat: bool = False,
                  exclude_irregular_payout: bool = False) -> JVDataset:
    """生 CSV から、既存パイプラインに渡せる3つの表を作る。

    start / end（"YYYY-MM-DD"）でレース日を絞れる。JVOpen は「提供時刻」でしか
    範囲を指定できないので、レース日で切りたいときはここで行う。
    ※ 過去成績の特徴量は絞った範囲の中だけで計算されるので、学習期間を
      絞りたいだけなら、ここでは絞らず validate の fold で切ること。
    """
    raw = raw or load_raw(data_dir)
    report = BuildReport()

    races = prepare_races(raw["RA"], report, jump_as_flat=jump_as_flat)
    horses = prepare_horses(raw["SE"], races, report)
    payout = prepare_payout(raw["HR"], report, exclude_irregular=exclude_irregular_payout)

    # RA はあるのに、使える馬が1頭も残らなかったレース（＝ここで落ちる）
    with_horses = set(horses["レースID"])
    dropped = races.loc[~races["レースID"].isin(with_horses)]
    report.add("RA のうち馬のデータが無く除外したレース", len(dropped))
    if len(dropped):
        by_year = dropped["レース日付"].dt.year.value_counts().sort_index().to_dict()
        report.note(f"開催年の内訳: {by_year}")
        report.note("SE が届いていない（過去レースの訂正で RA だけ届いた等）か、"
                    "全馬が取消・除外・中止だったレース")

    if start is not None:
        horses = horses.loc[horses["レース日付"] >= pd.Timestamp(start)]
    if end is not None:
        horses = horses.loc[horses["レース日付"] <= pd.Timestamp(end)]
    races = races.loc[races["レースID"].isin(set(horses["レースID"]))]
    report.add("最終レース数", len(races))

    # 払戻の有無の内訳（検証に使えるのは 3連単払戻があるレースだけ）
    pay = payout.set_index("レースID")["3連単払戻"]
    ids = races["レースID"]
    no_hr = int((~ids.isin(pay.index)).sum())
    no_pay = int(ids.isin(pay.index).sum() - pay.reindex(ids).notna().sum())
    report.add("最終レースのうち 3連単払戻あり", len(ids) - no_hr - no_pay)
    if no_hr or no_pay:
        report.note(f"払戻なしの内訳: HR が未着 {no_hr} / 不成立・特払・返還など {no_pay}")

    return JVDataset(
        race_result=horses.reset_index(drop=True),
        lap_df=prepare_pace(races),
        payout_df=payout,
        report=report,
    )


def grade_count_by_year(race_result: pd.DataFrame) -> pd.DataFrame:
    """年 × 格付けのレース数（Kaggle 側の数え方と突き合わせるため）。"""
    races = race_result.drop_duplicates("レースID")
    return pd.crosstab(races["レース日付"].dt.year,
                       races["リステッド・重賞競走"].fillna("平場"))
