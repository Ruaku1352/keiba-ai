"""v8e：2・3着への来やすさ（平場で試す。本番の特徴量・予測には入れていない）。

仮説：「勝ち切れないけれど 2・3着には堅実に来る馬」がいる。今の 3連単の確率（Henery 型）は、2・3着の段の強さを
1着の確率から全馬一律（p^γ, p^δ）で決めているので、こうした馬ごとの個性が入らない。

■ 段ごとの強さを馬ごとに変えた Henery 型
    P(i, j, k) = p_i · w2_j / (Σ w2 − w2_i) · w3_k / (Σ w3 − w3_i − w3_j)
    w2 = exp(γ·log p + φ·x)、 w3 = exp(δ·log p + ψ·x)       （x = 0 なら今の Henery 型そのもの）
  2着の段は「1着の馬を除いた中から w2 に比例して選ぶ」、3着の段は「1・2着を除いた中から w3 に比例して選ぶ」なので、
  (γ, φ) と (δ, ψ) は、実際の 2着・3着の馬を選ぶ条件付きロジットそのもの。market.fit_conditional_logit（Newton 法）で
  格子ではなく連続値で推定する（実際の 1〜3着の並びの尤度の最大化と同じ）

■ P1 の来やすさ（過去走だけ。as_of=day）
  過去の各レースで、その馬が 2着か 3着に来たか（0/1）− Henery 型で予測した 2着か3着の確率（その日の単勝オッズから）。
  その合計 ÷ (有効な過去走の数 + 2)（少ないときは 0 に寄せる。3c と同じ縮め方）。
  「1着の確率から見込まれる以上に 2・3着に来ていた馬」ほど大きい。
  過去のレースの予測に使う γ・δ は、検証の最初の年より前のレースだけで1回推定した値（検証期間の結果を使わない）

■ P2 の市場の来やすさ
  3連複の確定オッズ（O5）から、組ごとの市場の確率 ∝ 1/オッズ。馬ごとの 3着以内の市場の確率 = その馬を含む組の合計。
  x = log(3連複から見た 3着以内の確率) − log(Henery 型で単勝から出した 3着以内の確率)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import leakfree, market

PLACE_SHRINK = 2.0
PLACE_TERM = "来やすさ"
MARKET_PLACE_TERM = "市場の来やすさ"


# ---------------------------------------------------------------------------
# 段ごとの強さを馬ごとに変えた Henery 型
# ---------------------------------------------------------------------------
def stage_tensor_batch(p: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    """(R, n) の 1着の確率と 2・3着の段の強さから (R, n, n, n) の 3連単の確率。"""
    p = p / p.sum(axis=1, keepdims=True)
    n = p.shape[1]
    s2, s3 = w2.sum(axis=1), w3.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        second = w2[:, None, :] / (s2[:, None, None] - w2[:, :, None])
        third = w3[:, None, None, :] / (s3[:, None, None, None] - w3[:, :, None, None] - w3[:, None, :, None])
        P = p[:, :, None, None] * second[:, :, :, None] * third
    idx = np.arange(n)
    P[:, idx, idx, :] = 0
    P[:, idx, :, idx] = 0
    P[:, :, idx, idx] = 0
    return np.nan_to_num(P, nan=0.0, posinf=0.0)


def stage_tensor(p: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    return stage_tensor_batch(np.asarray(p, "float64")[None], np.asarray(w2, "float64")[None],
                              np.asarray(w3, "float64")[None])[0]


def stage_weights(p: np.ndarray, coef: dict, x=None) -> tuple[np.ndarray, np.ndarray]:
    """2・3着の段の強さ。coef は fit_stages の戻り値（"γ", "δ", と x があれば "φ", "ψ"）。"""
    lp = np.log(np.clip(np.asarray(p, "float64") / np.sum(p, axis=-1, keepdims=True), 1e-12, None))
    u2, u3 = coef["γ"] * lp, coef["δ"] * lp
    if x is not None and "φ" in coef:
        x = np.nan_to_num(np.asarray(x, "float64"))
        u2, u3 = u2 + coef["φ"] * x, u3 + coef["ψ"] * x
    m2 = u2.max(axis=-1, keepdims=True)
    m3 = u3.max(axis=-1, keepdims=True)
    return np.exp(u2 - m2), np.exp(u3 - m3)       # 定数倍しても比は同じ


def marginals_batch(P: np.ndarray) -> np.ndarray:
    """(R, n, n, n) → (R, n, 3)：各馬が 1着・2着・3着になる確率。"""
    return np.stack([P.sum(axis=(2, 3)), P.sum(axis=(1, 3)), P.sum(axis=(1, 2))], axis=2)


def clean_races(t: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """1〜3着がちょうど1頭ずつで、確率がそろったレースの行（レースID→元の並び）。"""
    rank = pd.to_numeric(t["着順"], errors="coerce")
    p = pd.to_numeric(t[prob_col], errors="coerce")
    g = t["レースID"]
    ok = pd.Series(True, index=t.index)
    for k in (1, 2, 3):
        ok &= (rank == k).groupby(g).transform("sum") == 1
    ok &= (p.notna() & (p > 0)).groupby(g).transform("all")
    ok &= g.groupby(g).transform("size") >= 3
    return t.loc[ok]


def fit_stages(t: pd.DataFrame, prob_col: str, x_col: str | None = None) -> dict:
    """2着・3着の段の (γ, φ)・(δ, ψ) を条件付きロジットで推定する。t は推定に使う行（平場など）。"""
    c = clean_races(t, prob_col)
    if c.empty:
        return {"γ": np.nan, "δ": np.nan, "レース数": 0}
    c, _ = market.race_slices(c)
    rank = pd.to_numeric(c["着順"], errors="coerce").to_numpy()
    p = c[prob_col].to_numpy(dtype="float64")
    lp = np.log(p / c.groupby("レースID")[prob_col].transform("sum").to_numpy())
    cols = [lp] + ([np.nan_to_num(c[x_col].to_numpy(dtype="float64"))] if x_col else [])
    X = np.column_stack(cols)
    rid = c["レースID"].to_numpy()
    out = {"レース数": int(pd.unique(rid).size)}
    for stage, keep, pick, names in [(2, rank != 1, 2, ("γ", "φ")), (3, (rank != 1) & (rank != 2), 3, ("δ", "ψ"))]:
        r = rid[keep]
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1], True])
        beta = market.fit_conditional_logit(X[keep], starts, (rank[keep] == pick).astype(int))
        for name, b in zip(names, beta):
            out[name] = float(b)
    return out


def stage_loglik(t: pd.DataFrame, prob_col: str, coef: dict, x_col: str | None = None) -> float:
    """並びの −log 尤度（1レースあたり）：−log P(1着) − log P(2着|1着) − log P(3着|1,2着)。"""
    c, starts = market.race_slices(clean_races(t, prob_col))
    if len(starts) < 2:
        return float("nan")
    rank = pd.to_numeric(c["着順"], errors="coerce").to_numpy()
    p_all = c[prob_col].to_numpy(dtype="float64")
    x_all = c[x_col].to_numpy(dtype="float64") if x_col else None
    total = 0.0
    for a, b in zip(starts[:-1], starts[1:]):
        p = p_all[a:b] / p_all[a:b].sum()
        w2, w3 = stage_weights(p, coef, None if x_all is None else x_all[a:b])
        r = rank[a:b]
        i, j, k = (int(np.flatnonzero(r == v)[0]) for v in (1, 2, 3))
        total -= np.log(p[i]) + np.log(w2[j] / (w2.sum() - w2[i])) + np.log(w3[k] / (w3.sum() - w3[i] - w3[j]))
    return total / (len(starts) - 1)


# ---------------------------------------------------------------------------
# P1 過去の成績から見た 2・3着への来やすさ
# ---------------------------------------------------------------------------
def _batched_marginals(c: pd.DataFrame, prob_col: str, coef: dict, chunk: int = 1500):
    """clean_races の行ごとの (1着・2着・3着の確率)。同じ頭数のレースをまとめて計算する。

    戻り値: (元の表の index, (行, 3))
    """
    c, starts = market.race_slices(c.assign(_row=c.index))
    out = np.zeros((len(c), 3))
    p_all = c[prob_col].to_numpy(dtype="float64")
    lens = np.diff(starts)
    for n in np.unique(lens):
        sel = np.flatnonzero(lens == n)
        for s in range(0, len(sel), chunk):
            rs = starts[sel[s:s + chunk]]
            idx = rs[:, None] + np.arange(n)[None, :]
            p = p_all[idx]
            p = p / p.sum(axis=1, keepdims=True)
            w2, w3 = stage_weights(p, coef)
            out[idx] = marginals_batch(stage_tensor_batch(p, w2, w3))
    return c["_row"].to_numpy(), out


def past_place_feature(race_result: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """馬ごと・レースごとの P1 の来やすさ（その日より前のレースだけから）。

    race_result は preprocess.basic_clean の後（レース日付・レースID・血統登録番号・着順・馬番・単勝オッズ）。
    cutoff より前のレースだけで、過去のレースの予測に使う γ・δ を1回推定する。
    戻り値: (レースID・血統登録番号・来やすさ・過去の有効走数・過去の1着・過去の2・3着, 使った γ・δ)
    """
    df = race_result[["レースID", "レース日付", "血統登録番号", "着順", "馬番", "単勝オッズ"]].copy()
    df = df.sort_values(["レース日付", "レースID"], kind="mergesort").reset_index(drop=True)
    inv = 1.0 / pd.to_numeric(df["単勝オッズ"], errors="coerce").where(lambda s: s > 0)
    df["_p"] = inv / inv.groupby(df["レースID"]).transform("sum")
    c = clean_races(df, "_p")
    early = c.loc[pd.to_datetime(c["レース日付"]) < cutoff]
    base = fit_stages(early, "_p") if early["レースID"].nunique() >= 50 else {"γ": 1.0, "δ": 1.0, "レース数": 0}
    rows, marg = _batched_marginals(c, "_p", base)
    rank = pd.to_numeric(df["着順"], errors="coerce")
    resid = pd.Series(np.nan, index=df.index)
    expected23 = marg[:, 1] + marg[:, 2]
    resid.loc[rows] = rank.loc[rows].isin([2, 3]).astype("float64").to_numpy() - expected23
    df["_day"] = pd.to_datetime(df["レース日付"])
    key = ["血統登録番号"]
    s = leakfree.past_sum_excluding_race(df, key, resid, "_day")             # 前日まで（同じ日は除く）
    n = leakfree.past_sum_excluding_race(df, key, resid.notna().astype("float64"), "_day")
    w1 = leakfree.past_sum_excluding_race(df, key, (rank == 1).astype("float64"), "_day")
    w23 = leakfree.past_sum_excluding_race(df, key, rank.isin([2, 3]).astype("float64"), "_day")
    out = pd.DataFrame({"レースID": df["レースID"], "血統登録番号": df["血統登録番号"],
                        PLACE_TERM: (s / (n + PLACE_SHRINK)).to_numpy(), "過去の有効走数": n.to_numpy(),
                        "過去の1着": w1.to_numpy(), "過去の2・3着": w23.to_numpy()})
    return out, {"γ": base["γ"], "δ": base["δ"], "推定レース数": base["レース数"]}


# ---------------------------------------------------------------------------
# P2 市場（3連複）から見た 3着以内の確率
# ---------------------------------------------------------------------------
def trio_top3(horses: np.ndarray, pay: np.ndarray, posts: np.ndarray) -> np.ndarray:
    """3連複の確定オッズ（組の馬番 (組, 3) と 100円あたりの配当）から、posts の順の 3着以内の市場の確率。"""
    q = 1.0 / pay
    q = q / q.sum()
    idx = np.full(100, -1)
    idx[posts] = np.arange(len(posts))
    out = np.zeros(len(posts))
    ix = idx[np.clip(horses, 0, 99)]
    ok = (ix >= 0).all(axis=1)
    for col in range(3):
        np.add.at(out, ix[ok, col], q[ok])
    return out
