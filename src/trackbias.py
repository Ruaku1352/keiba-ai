"""v8 3：当日の馬場の傾向（同じ日・同じ競馬場・同じ芝ダで、先に終わったレースの結果から）。

**検証用。本番の予測（predict）の特徴量には入れていない。**
features.add_all_features(..., extra={"trackbias": {...}}) を渡したときだけ付く。

■ 使えるレース（本番で間に合うものだけ）
  本番では、対象レースの発走の cutoff_min 分前までに買い終える。
  結果は発走から delay_min 分後に届くと見込む。したがって使えるのは
      その日・同じ競馬場・同じ芝ダで、発走時刻 ≤ 対象レースの発走時刻 − (cutoff_min + delay_min)
  のレースだけ（既定 20 + 20 = 40 分前までに発走したレース）。
  対象レースより後のレース・同じ時刻のレースは、この式で必ず外れる（tests/test_v8.py で確認）。

■ 1レースごとの傾向（結果から）
  内枠傾向  = 出走馬全体の平均馬番位置 − 3着以内の馬の平均馬番位置（0=最内, 1=大外）。正なら内が来ている
  前残り傾向 = 出走馬全体の平均4角位置 − 3着以内の馬の平均4角位置（0=先頭）。正なら前が残っている
  上がり傾向 = そのレースの後3Fが、同じ競馬場・距離・芝ダの過去（前日まで）の平均より何σ速いか

■ 当日の値（先に終わったレースが少ないときの扱い）
  使えたレースの傾向を「合計 ÷ (レース数 + SHRINK)」で平均する（SHRINK=1）。
  0 レースなら 0（傾向なし）、1レースなら半分に縮める。1R など使えるレースが無いときは 0 になり、
  「当日_使えたレース数」= 0 でモデルにも分かる。

■ 馬ごとの組み合わせ
  当日_内枠×枠位置  = 内枠傾向 × (0.5 − 自分の馬番位置)  … 内が来ていて自分が内なら正
  当日_前残り×脚質  = 前残り傾向 × (0.5 − 自分の脚質スコア) … 前が残っていて自分が先行タイプなら正
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, leakfree

SHRINK = 1.0
STATS = ["内枠傾向", "前残り傾向", "上がり傾向"]
L3_MIN_COUNT = 20


def track_bias_feature_columns() -> list[str]:
    return ["当日_使えたレース数"] + [f"当日_{s}" for s in STATS] + ["当日_内枠×枠位置", "当日_前残り×脚質"]


def start_minutes(s: pd.Series) -> pd.Series:
    """"1525" → 925（分）。読めなければ NaN。"""
    t = s.astype(str).str.strip()
    ok = t.str.fullmatch(r"\d{3,4}")
    t = t.str.zfill(4)
    return (pd.to_numeric(t.str[:2], errors="coerce") * 60 + pd.to_numeric(t.str[2:], errors="coerce")).where(ok)


def race_bias_stats(df: pd.DataFrame, day_col: str | None = None) -> pd.DataFrame:
    """1レース1行：レースの傾向（結果から作る）と、日付・場・芝ダ・発走時刻。"""
    cols = config.resolve_columns(df)
    race_id, rank, date = cols["race_id"], cols["rank"], cols["date"]
    n = pd.to_numeric(df["公式出走頭数"], errors="coerce") if "公式出走頭数" in df.columns else \
        df.groupby(race_id, observed=True, sort=False)[race_id].transform("size")
    post = (pd.to_numeric(df["馬番"], errors="coerce") - 1) / (n - 1).replace(0, np.nan)
    corner = pd.to_numeric(df["_4角相対位置"], errors="coerce") if "_4角相対位置" in df.columns \
        else pd.Series(np.nan, index=df.index)
    top3 = pd.to_numeric(df[rank], errors="coerce") <= 3
    h = pd.DataFrame({"race": df[race_id].to_numpy(), "post": post.to_numpy(), "corner": corner.to_numpy(),
                      "post3": post.where(top3).to_numpy(), "corner3": corner.where(top3).to_numpy()})
    agg = h.groupby("race", sort=False).mean()
    first = df.groupby(race_id, observed=True, sort=False).head(1)
    races = pd.DataFrame({
        "レースID": first[race_id].to_numpy(),
        "日付": pd.to_datetime(first[date]).to_numpy(),
        "場": first["競馬場コード"].astype(str).to_numpy(),
        "芝ダ": first["芝・ダート区分"].astype(str).to_numpy(),
        "距離": pd.to_numeric(first["距離(m)"], errors="coerce").to_numpy(),
        "発走分": start_minutes(first["発走時刻"]).to_numpy() if "発走時刻" in first.columns else np.nan,
        "後3F": pd.to_numeric(first["_後3F"], errors="coerce").to_numpy() if "_後3F" in first.columns
        else np.nan,
    })
    a = agg.reindex(races["レースID"].to_numpy())
    races["内枠傾向"] = (a["post"] - a["post3"]).to_numpy()
    races["前残り傾向"] = (a["corner"] - a["corner3"]).to_numpy()

    # 上がり傾向：同じ場・距離・芝ダの過去（前日まで）の後3F の平均・標準偏差で標準化
    races = races.sort_values(["日付", "レースID"], kind="mergesort").reset_index(drop=True)
    for k, c in [("_k1", "場"), ("_k2", "距離"), ("_k3", "芝ダ")]:
        races[k] = leakfree.label_encode(races[c].astype(str))
    l3 = races["後3F"].astype("float64")
    mean = leakfree.past_mean_ignore_nan_excluding(races, ["_k1", "_k2", "_k3"], l3, "日付",
                                                   min_count=L3_MIN_COUNT)
    std = leakfree.past_std_ignore_nan_excluding(races, ["_k1", "_k2", "_k3"], l3, "日付",
                                                 min_count=L3_MIN_COUNT)
    races["上がり傾向"] = ((mean.astype("float64") - l3) / std.astype("float64").replace(0, np.nan)).clip(-4, 4)
    return races.drop(columns=["_k1", "_k2", "_k3"])


def day_bias(races: pd.DataFrame, cutoff_min: float = 20, delay_min: float = 20) -> pd.DataFrame:
    """各レースについて、使えた先行レースの数と傾向（縮めた平均）。races は race_bias_stats の戻り値。"""
    lag = float(cutoff_min) + float(delay_min)
    r = races.copy()
    grp = r.groupby(["日付", "場", "芝ダ"], sort=False).ngroup().to_numpy()
    start = r["発走分"].to_numpy(dtype="float64")
    has_start = ~np.isnan(start)
    # 並べ替え用のキー（グループ × 1日の分）。発走時刻の無いレースは「使える側」に入れない
    key = grp * 10000.0 + np.where(has_start, start, 9999.0)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    out = pd.DataFrame(index=r.index)
    thresh = grp * 10000.0 + np.where(has_start, start - lag, -1.0)
    pos = np.searchsorted(key_sorted, thresh, side="right")      # 使える最後のレースの次の位置
    grp_first = np.searchsorted(key_sorted, grp * 10000.0, side="left")
    pos = np.where(has_start, np.maximum(pos, grp_first), grp_first)
    for s in STATS + ["_count"]:
        v = np.ones(len(r)) if s == "_count" else r[s].to_numpy(dtype="float64")
        valid = ~np.isnan(v) & has_start
        vs = np.where(valid, v, 0.0)[order]
        cs = valid[order].astype(float)
        P = np.concatenate([[0.0], np.cumsum(vs)])
        C = np.concatenate([[0.0], np.cumsum(cs)])
        total = P[pos] - P[grp_first]
        cnt = C[pos] - C[grp_first]
        if s == "_count":
            out["当日_使えたレース数"] = cnt
        else:
            out[f"当日_{s}"] = total / (cnt + SHRINK)
    out["レースID"] = r["レースID"].to_numpy()
    return out


def add_track_bias_features(df: pd.DataFrame, cutoff_min: float = 20, delay_min: float = 20) -> pd.DataFrame:
    """馬の行に当日の傾向を付ける。"""
    cols = config.resolve_columns(df)
    race_id = cols["race_id"]
    df = df.copy()
    races = race_bias_stats(df)
    bias = day_bias(races, cutoff_min, delay_min).set_index("レースID")
    rid = df[race_id].to_numpy()
    for c in track_bias_feature_columns()[:1 + len(STATS)]:
        df[c] = pd.Series(bias[c].reindex(rid).to_numpy(), index=df.index, dtype="float32")
    n = pd.to_numeric(df["公式出走頭数"], errors="coerce") if "公式出走頭数" in df.columns else \
        df.groupby(race_id, observed=True, sort=False)[race_id].transform("size")
    post = (pd.to_numeric(df["馬番"], errors="coerce") - 1) / (n - 1).replace(0, np.nan)
    df["当日_内枠×枠位置"] = (df["当日_内枠傾向"] * (0.5 - post)).astype("float32")
    style = df["脚質スコア"] if "脚質スコア" in df.columns else pd.Series(np.nan, index=df.index)
    df["当日_前残り×脚質"] = (df["当日_前残り傾向"] * (0.5 - style)).astype("float32")
    return df
