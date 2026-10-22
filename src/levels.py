"""v8 1.5：過去のレースのレベル（格）・着差・成績の推移・タイム指数の特徴量。

**検証用。本番の予測（predict）の特徴量には入れていない。**
features.add_all_features(..., extra={"levels": {}}) を渡したときだけ付く。

■ レースの格（RA の格付け・競走条件コードから。レースの前に分かる情報）
     0 新馬・未出走 / 1 未勝利 / 2 1勝（500万下）/ 3 2勝（1000万下）/ 4 3勝（1600万下）
     5 オープン / 6 リステッド / 7 G3・格付けなし重賞 / 8 G2 / 9 G1
  競走条件コード（2007）は年齢ごとに5つある（2歳・3歳・4歳・5歳以上・最若年）。
  例えば夏の「3歳以上」で 3歳は 500万下・4歳以上は 1000万下、という時代があったので、
  5つのうち一番上の格をそのレースの格とする。
  古い時代の「400万下」「900万下」なども、収得賞金の上限（万円）で 1勝/2勝/3勝 相当にまとめる:
     〜500万 → 1勝相当 / 〜1000万 → 2勝相当 / それ以上（〜1億）→ 3勝相当
  障害は J.G1〜3 を G1〜3 と同じ数字にする（平地とは別のレースだが、障害馬は障害の中で比べる）。

■ リーク対策
  馬ごとの値はすべて「前の行だけ」から作る（shift / leakfree）。
  タイム指数の基準（条件ごとの平均・標準偏差）は「前日まで」のレースで作る（同じ日を含めない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, leakfree

CLASS_LABELS = {0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝", 5: "OP",
                6: "L", 7: "G3", 8: "G2", 9: "G1"}
GRADE_LEVEL = {"A": 9, "B": 8, "C": 7, "D": 7, "F": 9, "G": 8, "H": 7, "L": 6}
JYOKEN_COLUMNS = [f"JyokenInfo.JyokenCD[{i}]" for i in range(1, 6)]
RECENT = 5                       # 「過去N走」の N
TIME_INDEX_MIN_COUNT = 30        # タイム指数の基準に要る、同じ条件の過去の頭数


def jyoken_level(code) -> float:
    """競走条件コード（3桁）→ 格（0〜5）。000・空は NaN。"""
    s = str(code).strip()
    if not s.isdigit():
        return np.nan
    n = int(s)
    if n == 0:
        return np.nan
    if n in (701, 702):
        return 0.0
    if n == 703:
        return 1.0
    if n == 999:
        return 5.0
    if 1 <= n <= 5:
        return 2.0
    if 6 <= n <= 10:
        return 3.0
    if 11 <= n <= 100:
        return 4.0
    return np.nan


def race_class_level(ra: pd.DataFrame) -> np.ndarray:
    """RA（1レース1行）から格（0〜9、分からなければ NaN）。"""
    if all(c in ra.columns for c in JYOKEN_COLUMNS):
        lv = np.column_stack([ra[c].map(jyoken_level).to_numpy(dtype="float64") for c in JYOKEN_COLUMNS])
        with np.errstate(all="ignore"):
            base = np.where(np.isnan(lv).all(axis=1), np.nan, np.nanmax(np.where(np.isnan(lv), -1, lv), axis=1))
    else:
        base = np.full(len(ra), np.nan)
    grade = ra["GradeCD"].astype(str).str.strip().map(GRADE_LEVEL).to_numpy(dtype="float64") \
        if "GradeCD" in ra.columns else np.full(len(ra), np.nan)
    return np.fmax(grade, base)          # どちらか片方が欠けていれば、もう片方


def parse_time_diff(s: pd.Series) -> pd.Series:
    """SE のタイム差（"+005" = 0.5秒遅れ、1着は "-003" = 2着に 0.3秒差）を秒にする。
    "9999"（取消・除外・中止）や空は NaN。1着の値は負（勝った着差）。"""
    t = s.astype(str).str.strip()
    ok = t.str.fullmatch(r"[+-]\d{3}")
    sign = np.where(t.str[0] == "-", -1.0, 1.0)
    val = pd.to_numeric(t.str[1:], errors="coerce") / 10.0
    return pd.Series(np.where(ok, sign * val, np.nan), index=s.index, dtype="float64")


# ---------------------------------------------------------------------------
# 特徴量
# ---------------------------------------------------------------------------
def level_feature_columns() -> list[str]:
    cols = ["今回の格", "前走の格", f"過去{RECENT}走平均格", "過去最高格", "格の差_前走", "格の差_過去最高",
            "同格以上_最高着順", "同格以上_平均相対着順", "同格以上_出走数",
            "前走タイム差", f"過去{RECENT}走平均タイム差",
            f"相対着順の傾き_{RECENT}走", "直近3走と通算の差_相対着順",
            "タイム指数_前走", f"タイム指数_過去{RECENT}走平均", "タイム指数_最高"]
    return cols


def _window_mean_valid(df: pd.DataFrame, keys: list[str], value: pd.Series, window: int):
    """直近 window 走のうち値がある回の (合計, 件数)。float64。前の行だけを足す。"""
    g = leakfree._grouper(df, keys)
    v = value.astype("float64")
    valid = v.notna().astype("float64")
    s = leakfree._exclusive_cumsum(v, g)
    c = leakfree._exclusive_cumsum(valid, g)
    s_lag = s.groupby(g, observed=True, sort=False).shift(window).fillna(0)
    c_lag = c.groupby(g, observed=True, sort=False).shift(window).fillna(0)
    return s - s_lag, c - c_lag


def add_level_features(df: pd.DataFrame, day_col: str | None = None) -> pd.DataFrame:
    """格・着差・推移・タイム指数の特徴量を足す。df は [日付, レースID, 馬番] 順。

    必要な列: 血統登録番号, 着順, _格レベル（jvmap が付ける）, _タイム差秒, タイム, 距離(m),
             芝・ダート区分, 競馬場コード, 馬場状態1, 公式出走頭数（無ければレースの頭数）
    """
    cols = config.resolve_columns(df)
    horse, rank, race_id = cols["horse"], cols["rank"], cols["race_id"]
    df = df.copy()
    g = df.groupby(horse, observed=True, sort=False)

    level = pd.to_numeric(df.get("_格レベル"), errors="coerce").astype("float64") \
        if "_格レベル" in df.columns else pd.Series(np.nan, index=df.index)
    df["今回の格"] = level.astype("float32")
    prev_level = level.groupby(df[horse], sort=False).shift(1)
    df["前走の格"] = prev_level.astype("float32")
    s, c = _window_mean_valid(df, [horse], level, RECENT)
    df[f"過去{RECENT}走平均格"] = (s / c.replace(0, np.nan)).astype("float32")
    best = level.groupby(df[horse], sort=False).cummax().groupby(df[horse], sort=False).ffill()
    best = best.groupby(df[horse], sort=False).shift(1)
    df["過去最高格"] = best.astype("float32")
    df["格の差_前走"] = (level - prev_level).astype("float32")
    df["格の差_過去最高"] = (level - best).astype("float32")

    # 相対着順（0=1着, 1=最下位）
    n = pd.to_numeric(df["公式出走頭数"], errors="coerce") if "公式出走頭数" in df.columns else \
        df.groupby(race_id, observed=True, sort=False)[race_id].transform("size")
    rel = ((pd.to_numeric(df[rank], errors="coerce") - 1) / (n - 1).replace(0, np.nan)).clip(0, 1)
    rk = pd.to_numeric(df[rank], errors="coerce").astype("float64")

    # 同格以上（今回の格以上）の過去のレースでの最高着順・平均相対着順・出走数
    best_rank = np.full(len(df), np.nan)
    mean_rel = np.full(len(df), np.nan)
    count = np.zeros(len(df))
    cur = level.to_numpy()
    for t in range(10):
        rows = cur == t
        if not rows.any():
            continue
        qualify = level >= t
        r_m = rk.where(qualify)
        cm = r_m.groupby(df[horse], sort=False).cummin()
        cm = cm.groupby(df[horse], sort=False).ffill().groupby(df[horse], sort=False).shift(1)
        mean_t = leakfree.past_mean_ignore_nan(df, [horse], rel.where(qualify).astype("float64"))
        cnt_t = leakfree.past_sum(df, [horse], qualify.astype("float64"))
        best_rank[rows] = cm.to_numpy()[rows]
        mean_rel[rows] = mean_t.to_numpy()[rows]
        count[rows] = cnt_t.to_numpy()[rows]
    df["同格以上_最高着順"] = pd.Series(best_rank, index=df.index, dtype="float32")
    df["同格以上_平均相対着順"] = pd.Series(mean_rel, index=df.index, dtype="float32")
    df["同格以上_出走数"] = pd.Series(np.where(np.isnan(cur), np.nan, count), index=df.index, dtype="float32")

    # 着差
    td = pd.to_numeric(df["_タイム差秒"], errors="coerce").clip(-3, 10) if "_タイム差秒" in df.columns \
        else pd.Series(np.nan, index=df.index)
    df["前走タイム差"] = td.groupby(df[horse], sort=False).shift(1).astype("float32")
    s, c = _window_mean_valid(df, [horse], td, RECENT)
    df[f"過去{RECENT}走平均タイム差"] = (s / c.replace(0, np.nan)).astype("float32")

    # 成績の推移：直近 N 走の相対着順の傾き（1走あたり。マイナス＝良くなっている）
    x = g.cumcount().astype("float64")
    yv = rel.astype("float64")
    xv = x.where(yv.notna())
    sx, cnt = _window_mean_valid(df, [horse], xv, RECENT)
    sy, _ = _window_mean_valid(df, [horse], yv, RECENT)
    sxx, _ = _window_mean_valid(df, [horse], xv * xv, RECENT)
    sxy, _ = _window_mean_valid(df, [horse], xv * yv, RECENT)
    with np.errstate(all="ignore"):
        var = sxx / cnt - (sx / cnt) ** 2
        cov = sxy / cnt - (sx / cnt) * (sy / cnt)
        slope = np.where((cnt >= 3) & (var > 1e-9), cov / var, np.nan)
    df[f"相対着順の傾き_{RECENT}走"] = pd.Series(slope, index=df.index, dtype="float32")
    s3, c3 = _window_mean_valid(df, [horse], yv, 3)
    all_mean = leakfree.past_mean_ignore_nan(df, [horse], yv)
    df["直近3走と通算の差_相対着順"] = ((s3 / c3.replace(0, np.nan)) - all_mean).astype("float32")

    # タイム指数：同じ 競馬場・距離・芝ダ・馬場 の過去（前日まで）の平均・標準偏差で標準化（速いほど大）
    df = _add_time_index(df, horse, day_col)
    return df


def _add_time_index(df: pd.DataFrame, horse: str, day_col: str | None) -> pd.DataFrame:
    needed = ["タイム", "距離(m)", "芝・ダート区分", "競馬場コード", "馬場状態1"]
    if not all(c in df.columns for c in needed):
        for c in ["タイム指数_前走", f"タイム指数_過去{RECENT}走平均", "タイム指数_最高"]:
            df[c] = np.float32(np.nan)
        return df
    t = pd.to_numeric(df["タイム"], errors="coerce").astype("float64")
    t = t.where(t > 0)
    keys = ["_cond_course", "_cond_dist", "_cond_surface", "_cond_going"]
    for k, c in zip(keys, ["競馬場コード", "距離(m)", "芝・ダート区分", "馬場状態1"]):
        df[k] = leakfree.label_encode(df[c].astype(str))
    group = day_col if day_col and day_col in df.columns else config.resolve_columns(df)["date"]
    mean = leakfree.past_mean_ignore_nan_excluding(df, keys, t, group, min_count=TIME_INDEX_MIN_COUNT)
    std = leakfree.past_std_ignore_nan_excluding(df, keys, t, group, min_count=TIME_INDEX_MIN_COUNT)
    z = (-(t - mean.astype("float64")) / std.astype("float64").replace(0, np.nan)).clip(-5, 5)
    df["_タイム指数"] = z
    df["タイム指数_前走"] = z.groupby(df[horse], sort=False).shift(1).astype("float32")
    s, c = _window_mean_valid(df, [horse], z, RECENT)
    df[f"タイム指数_過去{RECENT}走平均"] = (s / c.replace(0, np.nan)).astype("float32")
    best = z.groupby(df[horse], sort=False).cummax().groupby(df[horse], sort=False).ffill()
    df["タイム指数_最高"] = best.groupby(df[horse], sort=False).shift(1).astype("float32")
    return df
