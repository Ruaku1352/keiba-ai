"""v6：実戦用の予測（モデルの学習・保存、出馬表からの予測、練習モード）。

買い方は v4/v5 で確定したものから変えない：**重賞のみ・予測上位6頭の3連単BOX（120点・12,000円）**。

■ 特徴量は「前日までに確定したレース」だけで作る（as_of="day"）
  実戦では前日の夜に予測するので、当日の他のレースの結果はまだ無い。
  validate（as_of="race"）は騎手・調教師の成績に同じ日のより前のレースを含めていたので、
  そのままだと学習と予測で特徴量の意味がずれる。v6 は学習も予測も as_of="day" にそろえる。
  （as_of="race" の結果は変えていない。validate は今までどおり再現できる）

■ 予測の経路（build_prediction_features）
  1. 確定したレース（履歴）と予測対象のレースを1つの表につなげる
  2. 予測対象の**結果の列はすべて消す**（着順・タイム・コーナー順位・脚質判定・上がり・オッズ・人気など）
  3. 馬体重は「その馬の直近の確定レースの馬体重」、増減は 0 で代用する（発表前のため）
  4. 予測対象の日ごとに、その日より前の確定レースだけを履歴にして作る。
     as_of="day" は同じ日の行を互いの「過去」に含めないので、同じ日の予測対象レース
     どうしが混ざらない（例：同じ騎手が2つの対象レースに乗っても、もう片方を
     過去の騎乗として数えない）。日曜と月曜をまとめて予測しても、月曜のレースは
     日曜の対象レースを見ない（別々に作るため）
  5. ペース表（ラップ由来の値）も確定レースの分だけを渡す
  学習の経路（make_training_frame）と同じ関数で特徴量を作るので、
  過去のレースを両方の経路で作ると、馬体重以外は同じ値になる（compare_paths で確認する）。
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features, preprocess, trifecta

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
AS_OF = "day"
TRAIN_START = "2003-01-01"          # validate の fold と同じ学習開始日
# 全データで学習するときのラウンド数。5 fold の best iteration は 419/337/315/376/454。
# 学習データの量と期間が全データ（2003〜最新）に一番近いのは fold 5 なので、その 454 を使う
# （docs/v6_predict.md の判断2）。
DEFAULT_ROUNDS = 454
BOX_SIZE = 6

# 学習パラメータは validate と同じ。再現性のための設定だけ足す
# （deterministic は同じデータ・同じパラメータで同じモデルになるようにする設定。
#   LightGBM の説明に従い、force_col_wise と組み合わせる）
REPRO_PARAMS = {"deterministic": True, "force_col_wise": True}

# 予測対象で消す「結果」の列（race_result の列名）
RESULT_COLUMNS = [
    "着順", "タイム", "タイム差", "後３Ｆタイム", "単勝オッズ", "人気",
    "_1コーナー順位", "_2コーナー順位", "_3コーナー順位", "_4コーナー順位", "_脚質判定",
    "_前3F", "_後3F", "_前3F公式", "_後3F公式", "_前3F計算", "_後3F計算", "_ラップ1本目",
]
WEIGHT_COLUMNS = ("馬体重", "場体重増減")
GOING_COLUMN = "馬場状態1"
DEFAULT_GOING = "良"
TARGET_FLAG = "予測対象"   # `_` で始めない（add_all_features の最後で消されないように）


def data_root() -> Path:
    """data/（JV_DATA_DIR の1つ上）。モデルと予測はここに置く（.gitignore 済み）。"""
    return Path(config.JV_DATA_DIR).parent


def model_dir() -> Path:
    return data_root() / "models"


def prediction_dir() -> Path:
    return data_root() / "predictions"


# ---------------------------------------------------------------------------
# 学習の経路
# ---------------------------------------------------------------------------
def make_training_frame(race_result: pd.DataFrame, lap_df: pd.DataFrame,
                        until: str | None = None) -> pd.DataFrame:
    """確定したレースから、学習用の特徴量の表を作る（as_of="day"）。

    until を渡すと、その日までのレースだけで作る（特徴量も until より後を一切見ない）。
    """
    rr = race_result
    if until is not None:
        rr = rr.loc[pd.to_datetime(rr["レース日付"]) <= pd.Timestamp(until)]
        lap_df = lap_df.loc[lap_df["レースID"].isin(set(rr["レースID"]))]
    df = preprocess.basic_clean(rr)
    df = features.add_all_features(df, lap_df=lap_df, as_of=AS_OF)
    df = preprocess.downcast(df)
    return preprocess.to_category(df)


def train_params() -> dict:
    return {**config.LGB_PARAMS, **REPRO_PARAMS}


def train_model(frame: pd.DataFrame, rounds: int = DEFAULT_ROUNDS,
                start: str = TRAIN_START, until: str | None = None):
    """全期間（start〜until）で学習する。検証データは無いのでラウンド数は固定。"""
    import lightgbm as lgb

    dates = pd.to_datetime(frame["レース日付"])
    mask = dates >= pd.Timestamp(start)
    if until is not None:
        mask &= dates <= pd.Timestamp(until)
    train = frame.loc[mask]
    feature_cols = features.feature_columns(frame)
    data = lgb.Dataset(train[feature_cols], label=train[config.TARGET])
    booster = lgb.train(train_params(), data, num_boost_round=rounds)
    info = {
        "train_rows": int(len(train)),
        "train_races": int(train["レースID"].nunique()),
        "train_first_date": f"{dates[mask].min():%Y-%m-%d}",
        "train_last_date": f"{dates[mask].max():%Y-%m-%d}",
    }
    return booster, feature_cols, info


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("（src に未コミットの変更あり）" if dirty else "")
    except Exception:
        return "不明"


@dataclass
class SavedModel:
    booster: object
    meta: dict
    path: Path

    @property
    def feature_cols(self) -> list[str]:
        return self.meta["feature_cols"]

    @property
    def until(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["train_last_date"])


def save_model(booster, feature_cols: list[str], info: dict, rounds: int,
               until: str | None, data_info: dict, name: str | None = None) -> Path:
    """モデル本体と、再現に必要な情報（meta.json）を data/models/<name>/ に保存する。"""
    import lightgbm as lgb

    name = name or f"v6_until{info['train_last_date'].replace('-', '')}"
    out = model_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    meta = {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": AS_OF,
        "rounds": rounds,
        "params": train_params(),
        "train_start": TRAIN_START,
        "until_requested": until,
        **info,
        "feature_cols": feature_cols,
        **data_info,
        "git_commit": _git_commit(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "lightgbm": lgb.__version__},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir() / "latest.txt").write_text(name, encoding="utf-8")
    return out


def load_model(name: str | None = None) -> SavedModel:
    """保存したモデルを読む。name を省略すると最後に学習したもの（latest.txt）。"""
    import lightgbm as lgb

    if not name or name == "latest":
        latest = model_dir() / "latest.txt"
        if not latest.exists():
            raise FileNotFoundError("モデルがありません。先に `py -m src.v6 train` を実行してください")
        name = latest.read_text(encoding="utf-8").strip()
    path = model_dir() / name
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(path / "model.txt"))
    return SavedModel(booster, meta, path)


# ---------------------------------------------------------------------------
# 予測の経路
# ---------------------------------------------------------------------------
def blank_results(targets: pd.DataFrame) -> pd.DataFrame:
    """予測対象の結果の列をすべて消す（出力が結果に依存しないことの保証）。"""
    out = targets.copy()
    for col in RESULT_COLUMNS:
        if col in out.columns:
            out[col] = np.nan
    if "_3F出所" in out.columns:
        out["_3F出所"] = "なし"
    if config.TARGET in out.columns:
        out[config.TARGET] = np.nan
    return out


def previous_weight(df: pd.DataFrame) -> pd.Series:
    """各行について、その馬の「自分より前の行」で最後に分かっている馬体重。

    df は [日付, レースID, 馬番] 順に並んでいること。計量不能などで欠けている
    レースは飛ばして、その前の値を使う。前走が無ければ NaN。
    """
    horse = config.resolve_columns(df)["horse"]
    w = pd.to_numeric(df["馬体重"], errors="coerce")
    w = w.where(w > 0)
    known = w.groupby(df[horse], sort=False).ffill()
    return known.groupby(df[horse], sort=False).shift(1)


def substitute_weight(df: pd.DataFrame, rows: pd.Series) -> pd.DataFrame:
    """rows の行の馬体重を「前走の馬体重」、増減を 0 にする（発表前の予測用）。"""
    df = df.copy()
    prev = previous_weight(df)
    df.loc[rows, "馬体重"] = prev.loc[rows]
    df.loc[rows, "場体重増減"] = 0.0
    df["馬体重の扱い"] = np.where(
        rows, np.where(prev.notna(), "前走の値で代用", "前走なし（欠損）"), "")
    return df


def build_prediction_features(history: pd.DataFrame, targets: pd.DataFrame,
                              lap_df: pd.DataFrame, going: str | None = None,
                              keep_actual_going: bool = False,
                              substitute_weights: bool = True) -> pd.DataFrame:
    """予測の経路：確定したレース（history）の上に、予測対象（targets）の特徴量を作る。

    予測対象の日ごとに別々に作る。各日について、使う履歴は**その日より前**のレースだけ
    （同じ日・それ以降の行は渡されても捨てる）。こうすると
      - 同じ日の他のレース（未確定）を過去として数えない
      - 日曜と月曜をまとめて予測しても、月曜のレースが日曜の対象レース（未確定）を
        「前走」や「過去の騎乗」として数えない
    同じ日の予測対象どうしは as_of="day" の集計で互いに除かれる。

    Parameters
    ----------
    history : 確定したレース（race_result の形。jvmap.build_dataset の race_result）
    targets : 予測対象（race_result の形。結果の列は入っていてもここで消す）
    lap_df  : 確定したレースのペース表（予測対象のレースの分はここで除く）
    going   : 予測対象の馬場状態を指定する（例: "稍重"）
    keep_actual_going : True なら予測対象に入っている馬場状態をそのまま使う（練習モード用）。
                        False で going も無ければ、空欄を「良」と仮定する
    substitute_weights : False なら馬体重を代用しない（実際の値で予測したいとき）

    戻り値: 予測対象の行だけの特徴量の表（`馬体重の扱い`・`馬場状態の扱い` 列つき）
    """
    if targets.empty:
        return pd.DataFrame()
    target_ids = set(targets["レースID"])
    lap_hist = lap_df.loc[~lap_df["レースID"].isin(target_ids)]
    hist_all = history.loc[~history["レースID"].isin(target_ids)]
    hist_dates = pd.to_datetime(hist_all["レース日付"])
    tgt_dates = pd.to_datetime(targets["レース日付"])

    parts = []
    for day in sorted(tgt_dates.unique()):
        hist = hist_all.loc[hist_dates < day]
        parts.append(_build_one_day(hist, targets.loc[tgt_dates == day], lap_hist,
                                    going, keep_actual_going, substitute_weights))
    out = pd.concat(parts, ignore_index=True)
    return preprocess.to_category(out)


def _build_one_day(hist: pd.DataFrame, targets: pd.DataFrame, lap_hist: pd.DataFrame,
                   going: str | None, keep_actual_going: bool,
                   substitute_weights: bool) -> pd.DataFrame:
    """1日分の予測対象の特徴量（hist はその日より前の確定レースだけ）。"""
    hist = preprocess.basic_clean(hist)
    hist[TARGET_FLAG] = False

    tgt = preprocess.clean_entries(blank_results(targets))
    tgt[TARGET_FLAG] = True

    df = pd.concat([hist, tgt], ignore_index=True)
    df = df.sort_values(["レース日付", "レースID", "馬番"], kind="mergesort").reset_index(drop=True)
    rows = df[TARGET_FLAG].astype(bool)

    # 馬体重（発表前の代用）
    if substitute_weights:
        df = substitute_weight(df, rows)
    else:
        df["馬体重の扱い"] = np.where(rows, "実際の値", "")

    # 馬場状態（出馬表の段階では未発表）
    df[GOING_COLUMN] = df[GOING_COLUMN].astype(object)
    if going is not None:
        df.loc[rows, GOING_COLUMN] = going
        df["馬場状態の扱い"] = np.where(rows, f"指定（{going}）", "")
    elif keep_actual_going:
        df["馬場状態の扱い"] = np.where(rows, "実際の値", "")
    else:
        missing = rows & df[GOING_COLUMN].isna()
        df.loc[missing, GOING_COLUMN] = DEFAULT_GOING
        df["馬場状態の扱い"] = np.where(missing, f"未発表（{DEFAULT_GOING}と仮定）",
                                     np.where(rows, "発表済み", ""))

    df[features.DAY_COL] = df["レース日付"]
    out = features.add_all_features(df, lap_df=lap_hist, as_of=AS_OF)
    out = preprocess.downcast(out)
    return out.loc[out[TARGET_FLAG].astype(bool)].reset_index(drop=True)


def check_duplicate_horses(targets: pd.DataFrame) -> list[str]:
    """同じ馬が複数の予測対象レースに入っていないか（入っていると前走の扱いが崩れる）。"""
    dup = targets.loc[targets.duplicated("血統登録番号", keep=False)]
    return [f"{h}（{', '.join(g['レースID'])}）" for h, g in dup.groupby("血統登録番号")]


# ---------------------------------------------------------------------------
# 学習の経路と予測の経路の比較
# ---------------------------------------------------------------------------
def compare_paths(train_rows: pd.DataFrame, pred_rows: pd.DataFrame,
                  feature_cols: list[str], exclude=WEIGHT_COLUMNS,
                  rtol: float = 1e-5, atol: float = 1e-6) -> pd.DataFrame:
    """同じレースを2つの経路で作った特徴量を比べ、列ごとに一致しないセルの数を返す。

    数値は相対1e-5・絶対1e-6 の誤差まで一致とみなす（累積和の引き算の順序が違うと
    float32 の最後の桁がずれることがあるため）。欠損どうしは一致。
    """
    key = ["レースID", "血統登録番号"]
    a = train_rows.set_index(key)
    b = pred_rows.set_index(key)
    common = a.index.intersection(b.index)
    rows = []
    for col in feature_cols:
        if col in exclude:
            continue
        x, y = a.loc[common, col], b.loc[common, col]
        if isinstance(x.dtype, pd.CategoricalDtype) or isinstance(y.dtype, pd.CategoricalDtype) \
                or x.dtype == object or y.dtype == object:
            xs, ys = x.astype(object), y.astype(object)
            same = (xs == ys) | (xs.isna() & ys.isna())
        else:
            xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
            yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")
            same = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
        rows.append({"列": col, "不一致": int((~np.asarray(same)).sum()), "比較した行": len(common)})
    out = pd.DataFrame(rows)
    out.attrs["missing_rows"] = len(a.index.symmetric_difference(b.index))
    return out


# ---------------------------------------------------------------------------
# 予測と出力
# ---------------------------------------------------------------------------
def score(model: SavedModel, feats: pd.DataFrame) -> pd.DataFrame:
    """予測確率（レース内で合計1に正規化）と予測順位を付ける。"""
    out = feats.copy()
    raw = model.booster.predict(out[model.feature_cols])
    out["生スコア"] = raw
    total = out.groupby("レースID")["生スコア"].transform("sum")
    out["予測確率"] = out["生スコア"] / total
    out["予測順位"] = out.groupby("レースID")["生スコア"].rank(
        ascending=False, method="first").astype(int)
    return out.sort_values(["レース日付", "レースID", "予測順位"]).reset_index(drop=True)


def _pad(text: str, width: int) -> str:
    """全角を2桁として数えて、表示幅 width になるよう右に空白を足す（はみ出す分は切る）。"""
    import unicodedata

    out, used = "", 0
    for ch in str(text):
        w = 2 if unicodedata.east_asian_width(ch) in ("F", "W", "A") else 1
        if used + w > width:
            break
        out += ch
        used += w
    return out + " " * (width - used)


def _race_no(race_id: str) -> int:
    return int(str(race_id)[-2:])


def box_horses(race: pd.DataFrame, size: int = BOX_SIZE) -> list[str]:
    """上位 size 頭（馬番。馬番が未確定なら馬名）。"""
    top = race.nsmallest(size, "予測順位")
    if top["馬番"].notna().all():
        return [str(int(x)) for x in top["馬番"]]
    return list(top["馬名表示"].astype(str))


def render_race(race: pd.DataFrame, model: SavedModel, actual: dict | None = None) -> str:
    """1レース分をコンソール用の文字列にする。"""
    r = race.iloc[0]
    date = pd.Timestamp(r["レース日付"])
    status = ""
    if "データ区分" in race.columns and pd.notna(r.get("データ区分")):
        made = r.get("データ作成日")
        parts = [str(r["データ区分"])]
        if pd.notna(made):
            parts.append(f"{pd.Timestamp(made):%Y-%m-%d} 作成")
        if isinstance(r.get("データ時点"), str):
            parts.append(f"取得 {r['データ時点']} 時点")
        status = "［" + " ".join(parts) + "］"
    elif actual is not None:
        status = "［練習モード：結果を消して予測］"
    surface = r.get("芝・ダート区分") or ""
    lines = [
        f"{date:%Y-%m-%d} {r['競馬場名']}{_race_no(r['レースID'])}R （{r['競走名']}）"
        f"{r['リステッド・重賞競走']} {surface}{int(r['距離(m)'])}m {len(race)}頭  {status}",
    ]
    if "枠順確定" in race.columns and not bool(r["枠順確定"]):
        lines.append("  【注意】枠順が未確定（出走馬名表の段階）。馬番が決まったらデータを取り直して再予測すること")
    going_note = r.get("馬場状態の扱い", "")
    if isinstance(going_note, str) and going_note.startswith("未発表"):
        lines.append(f"  ※ 馬場状態は{going_note}。当日の発表を見て --going で指定し直せる")
    if (race.get("馬体重の扱い") == "前走の値で代用").any():
        lines.append("  ※ 馬体重は発表前のため前走の値で代用（増減は0）")

    header = f"  順位 馬番  {_pad('馬名', 18)}  {_pad('騎手', 8)}  予測確率"
    if actual is not None:
        header += "  着順 人気"
    lines.append(header)
    for _, h in race.iterrows():
        post = "--" if pd.isna(h["馬番"]) else f"{int(h['馬番']):>2}"
        name = _pad(h["馬名表示"], 18)
        jockey = _pad(h.get("騎手名略称", ""), 8)
        line = f"  {int(h['予測順位']):>3}   {post}  {name}  {jockey}  {100 * h['予測確率']:7.1f}%"
        if actual is not None:
            rank = h.get("実際の着順")
            pop = h.get("実際の人気")
            line += f"  {'' if pd.isna(rank) else int(rank):>4} {'' if pd.isna(pop) else int(pop):>4}"
        lines.append(line)

    box = box_horses(race)
    points = BOX_SIZE * (BOX_SIZE - 1) * (BOX_SIZE - 2)
    lines.append(f"  買い目: 3連単 {'・'.join(box)} の{BOX_SIZE}頭BOX {points}点 "
                 f"{points * trifecta.BET_UNIT:,}円")

    if actual is not None:
        lines.append(f"  結果: 1着 {actual['1着']} / 2着 {actual['2着']} / 3着 {actual['3着']}"
                     f"  3連単 {actual['払戻表示']}")
        lines.append(f"        {actual['判定']}")
    return "\n".join(lines)


def output_table(scored: pd.DataFrame, model: SavedModel) -> pd.DataFrame:
    """CSV 用の表。先頭7列はスプレッドシート「評価入力」シートと同じ順番。"""
    rows = scored.copy()
    rows["日付"] = pd.to_datetime(rows["レース日付"]).dt.strftime("%Y-%m-%d")
    box = {rid: "・".join(box_horses(g)) for rid, g in rows.groupby("レースID", sort=False)}
    out = pd.DataFrame({
        "レースID": rows["レースID"],
        "日付": rows["日付"],
        "レース名": rows["競走名"],
        "馬番": rows["馬番"].map(lambda x: "" if pd.isna(x) else int(x)),
        "馬名": rows["馬名表示"],
        "予測確率": rows["予測確率"].round(4),
        "予測順位": rows["予測順位"],
        "騎手": rows.get("騎手名略称"),
        "格付け": rows["リステッド・重賞競走"],
        "競馬場": rows["競馬場名"],
        "距離": rows["距離(m)"],
        "芝ダ": rows["芝・ダート区分"],
        "頭数": rows.groupby("レースID")["レースID"].transform("size"),
        "買い目（6頭BOX）": rows["レースID"].map(box),
        "データ区分": rows.get("データ区分", "練習（結果を消した確定データ）"),
        "データ作成日": rows.get("データ作成日"),
        "枠順確定": rows.get("枠順確定", True),
        "馬体重の扱い": rows.get("馬体重の扱い"),
        "馬場状態の扱い": rows.get("馬場状態の扱い"),
        "モデル": model.meta["name"],
        "モデル学習終了日": model.meta["train_last_date"],
    })
    for col in ("実際の着順", "実際の人気"):
        if col in rows.columns:
            out[col] = rows[col]
    return out


def save_predictions(table: pd.DataFrame, label: str) -> Path:
    out = prediction_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------------
# 練習モード用：実際の結果
# ---------------------------------------------------------------------------
def actual_result(race_rows: pd.DataFrame, payout: pd.DataFrame, scored_race: pd.DataFrame) -> dict:
    """そのレースの実際の1〜3着、3連単の払戻、上位6頭BOXが当たったか。"""
    rid = race_rows["レースID"].iloc[0]
    top3 = race_rows.loc[race_rows["着順"].isin([1, 2, 3])].sort_values("着順")

    def post_of(rank):
        hit = top3.loc[top3["着順"] == rank, "馬番"]
        return "/".join(str(int(x)) for x in hit) if len(hit) else "?"

    pay_row = payout.loc[payout["レースID"] == rid]
    pay = float(pay_row["3連単払戻"].iloc[0]) if len(pay_row) and pd.notna(pay_row["3連単払戻"].iloc[0]) else np.nan
    box = set(scored_race.nsmallest(BOX_SIZE, "予測順位")["血統登録番号"])
    hit = len(top3) >= 3 and set(top3["血統登録番号"].head(3)) <= box
    invest = BOX_SIZE * (BOX_SIZE - 1) * (BOX_SIZE - 2) * trifecta.BET_UNIT
    threshold = trifecta.hot_threshold(invest)
    if np.isnan(pay):
        verdict = "払戻データなし"
    elif hit:
        verdict = (f"的中 払戻 {pay:,.0f}円（投資 {invest:,}円）  熱い基準 {threshold:,}円を"
                   + ("超えた" if pay >= threshold else "超えなかった"))
    else:
        verdict = f"不的中（投資 {invest:,}円）"
    return {"1着": post_of(1), "2着": post_of(2), "3着": post_of(3),
            "払戻表示": "不明" if np.isnan(pay) else f"{pay:,.0f}円",
            "的中": bool(hit), "払戻": pay, "熱い": bool(hit and pay >= threshold),
            "判定": verdict}
