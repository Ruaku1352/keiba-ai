"""v8 2：モデルと市場（オッズ）の確率、3連単の確率、配当の推定。

■ 2b Benter (1994) の2段階
  1段目: モデル（オッズなし）の勝率 p。out-of-fold（その期間を学習に使っていないモデルの予測）
  2段目: レース内の条件付きロジット  P_i ∝ exp(α·log p_i + β·log q_i)
         q = 単勝オッズから出した市場の勝率（1/オッズ をレース内で正規化）
  2段目の係数は、**評価する fold より前の fold の平場のレース**だけで推定する（walk-forward）。
  1段目の予測はどれも out-of-fold なので、2段目の推定に使う期間は1段目の学習に入っていない。
  α が 0 に近ければ「モデルは市場以上のことを知らない」。

■ 2c 1着の確率から3連単の確率
  Harville:  P(i,j,k) = p_i · p_j/(1−p_i) · p_k/(1−p_i−p_j)
  Henery 型（Lo & Bacon-Shone の割引）: 2着・3着は p^γ・p^δ で計算し直す
             P(i,j,k) = p_i · p_j^γ/Σ_{m≠i} p_m^γ · p_k^δ/Σ_{m≠i,j} p_m^δ
             （γ=δ=1 が Harville。人気薄が2・3着に来やすいことを γ,δ<1 で表す）
  γ・δ は 2段目と同じく walk-forward で、平場の実際の2着・3着の尤度を最大にする値を格子で探す。

■ 2d 配当（100円あたり）の推定
  過去の3連単オッズ（O6）は保存していないので、市場の勝率 q から Henery 型で組の確率を出し、
      推定配当 = 100 × 払戻率 ÷ 組の確率      （払戻率: 3連単 72.5% / 3連複・馬単 75% / 馬連 77.5%）
  とする。実際の払戻（HR の的中組）と比べて、どのくらい合うかを見る。
  ※ 払戻率は現在のもの。2014年6月より前は券種によって違った可能性がある（年ごとの比も出す）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import trifecta

PAYOUT_RATE = {"3連単": 0.725, "3連複": 0.75, "馬単": 0.75, "馬連": 0.775}


# ---------------------------------------------------------------------------
# レース単位の配列
# ---------------------------------------------------------------------------
def race_slices(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """レースIDで並べ替えた表と、各レースの先頭位置（最後に len を足したもの）。"""
    t = table.sort_values(["レースID", "馬番"], kind="mergesort").reset_index(drop=True)
    codes = pd.factorize(t["レースID"])[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return t, np.r_[starts, len(t)]


def normalize(score: pd.Series, race: pd.Series) -> pd.Series:
    return score / score.groupby(race).transform("sum")


# ---------------------------------------------------------------------------
# 確率の当たり具合
# ---------------------------------------------------------------------------
def race_log_loss(table: pd.DataFrame, prob_col: str) -> float:
    """1着の馬に付けた確率の −log の平均（レースごと。1着が1頭のレースだけ）。小さいほど良い。"""
    w = table.loc[table["win"] == 1]
    w = w.loc[~w["レースID"].duplicated(keep=False)]
    p = w[prob_col].astype("float64").clip(lower=1e-6)
    return float(-np.log(p).mean()) if len(p) else float("nan")


def auc(y, p) -> float:
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(p, dtype="float64")).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


WIN_BINS = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0001]


def calibration_table(table: pd.DataFrame, prob_col: str, bins=WIN_BINS) -> pd.DataFrame:
    """予測した勝率の区間ごとに、予測の平均と実際の勝率。"""
    p = table[prob_col].astype("float64")
    b = pd.cut(p, bins=bins, right=False)
    g = pd.DataFrame({"区間": b, "p": p, "win": table["win"]}).groupby("区間", observed=True)
    out = g.agg(頭数=("win", "size"), 予測の平均=("p", "mean"), 実際の勝率=("win", "mean")).reset_index()
    out["区間"] = out["区間"].astype(str)
    return out


def expected_calibration_error(table: pd.DataFrame, prob_col: str, bins=WIN_BINS) -> float:
    c = calibration_table(table, prob_col, bins)
    w = c["頭数"] / c["頭数"].sum()
    return float((w * (c["予測の平均"] - c["実際の勝率"]).abs()).sum())


def summary_row(table: pd.DataFrame, prob_col: str, name: str) -> dict:
    return {"確率": name, "レース数": int(table["レースID"].nunique()),
            "log-loss（1着）": race_log_loss(table, prob_col),
            "AUC": auc(table["win"], table[prob_col]),
            "キャリブレーション誤差": expected_calibration_error(table, prob_col)}


# ---------------------------------------------------------------------------
# 2b 条件付きロジット
# ---------------------------------------------------------------------------
def _cl_terms(X: np.ndarray, starts: np.ndarray, beta: np.ndarray):
    u = X @ beta
    mx = np.maximum.reduceat(u, starts[:-1])
    lens = np.diff(starts)
    u = u - np.repeat(mx, lens)
    e = np.exp(u)
    s = np.add.reduceat(e, starts[:-1])
    p = e / np.repeat(s, lens)
    return u, s, p


def fit_conditional_logit(X: np.ndarray, starts: np.ndarray, y: np.ndarray,
                          max_iter: int = 50, tol: float = 1e-9) -> np.ndarray:
    """レース内の条件付きロジットを Newton 法で最尤推定する。各レースちょうど1頭が y=1。"""
    k = X.shape[1]
    beta = np.zeros(k)
    lens = np.diff(starts)

    def loglik(b):
        u, s, _ = _cl_terms(X, starts, b)
        return float(u[y == 1].sum() - np.log(s).sum())

    ll = loglik(beta)
    for _ in range(max_iter):
        _, _, p = _cl_terms(X, starts, beta)
        grad = ((y - p)[:, None] * X).sum(axis=0)
        m = np.add.reduceat(p[:, None] * X, starts[:-1], axis=0)          # レースごとの Σ p x
        hess = -(X.T @ (p[:, None] * X) - m.T @ m)
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-6:
            nb = beta + t * step
            nll = loglik(nb)
            if nll >= ll - 1e-12:
                break
            t /= 2
        if abs(nll - ll) < tol:
            beta, ll = nb, nll
            break
        beta, ll = nb, nll
    _ = lens
    return beta


def _cl_design(t: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.log(t[c].astype("float64").clip(lower=1e-6)) for c in cols])


def _one_winner(t: pd.DataFrame) -> pd.DataFrame:
    """1着がちょうど1頭で、オッズ・確率がそろっているレースだけ。"""
    ok = t.groupby("レースID")["win"].transform("sum") == 1
    full = t.groupby("レースID")["市場確率"].transform(lambda s: s.notna().all())
    return t.loc[ok & full]


def benter_walk_forward(table: pd.DataFrame, model_col: str = "prob",
                        market_col: str = "市場確率") -> tuple[pd.DataFrame, pd.DataFrame]:
    """fold ごとに、それより前の fold の平場で2段目を推定し、その fold を予測する。

    戻り値: (table に "blend" 列を足したもの（最初の fold は推定できないので除く）, 係数の表)
    """
    t = _one_winner(table)
    folds = list(dict.fromkeys(t["fold"]))
    parts, coefs = [], []
    for i, f in enumerate(folds):
        if i == 0:
            continue
        fit = t.loc[t["fold"].isin(folds[:i]) & ~t["重賞"]]
        fs, starts = race_slices(fit)
        beta = fit_conditional_logit(_cl_design(fs, [model_col, market_col]), starts,
                                     fs["win"].to_numpy())
        coefs.append({"検証fold": f, "推定に使ったfold": "+".join(folds[:i]),
                      "推定レース数": int(len(starts) - 1),
                      "α（モデル）": beta[0], "β（市場）": beta[1]})
        ev = t.loc[t["fold"] == f].copy()
        u = _cl_design(ev, [model_col, market_col]) @ beta
        e = np.exp(u - pd.Series(u, index=ev.index).groupby(ev["レースID"]).transform("max").to_numpy())
        ev["blend"] = e / pd.Series(e, index=ev.index).groupby(ev["レースID"]).transform("sum").to_numpy()
        parts.append(ev)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(coefs)


# ---------------------------------------------------------------------------
# 2c 3連単の確率
# ---------------------------------------------------------------------------
def trifecta_tensor(p: np.ndarray, gamma: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """P[i,j,k]（1着 i・2着 j・3着 k）。gamma=delta=1 で Harville。"""
    p = np.asarray(p, dtype="float64")
    p = p / p.sum()
    n = len(p)
    pg, pd_ = p ** gamma, p ** delta
    sg, sd = pg.sum(), pd_.sum()
    with np.errstate(divide="ignore", invalid="ignore"):          # 対角（同じ馬）は後で 0 にする
        second = pg[None, :] / (sg - pg[:, None])                   # [i,j]
        third = pd_[None, None, :] / (sd - pd_[:, None, None] - pd_[None, :, None])   # [i,j,k]
        P = p[:, None, None] * second[:, :, None] * third
    idx = np.arange(n)
    P[idx, idx, :] = 0
    P[idx, :, idx] = 0
    P[:, idx, idx] = 0
    return np.nan_to_num(P, nan=0.0, posinf=0.0)


GRID = np.round(np.arange(0.50, 1.201, 0.01), 2)


def _place_loglik(p: np.ndarray, first: int, second: int, third: int, grid=GRID):
    """格子の各値での log P(2着|1着)・log P(3着|1,2着)。"""
    p = p / p.sum()
    pw = p[None, :] ** grid[:, None]                      # (G, n)
    tot = pw.sum(axis=1)
    l2 = np.log(pw[:, second]) - np.log(tot - pw[:, first])
    l3 = np.log(pw[:, third]) - np.log(tot - pw[:, first] - pw[:, second])
    return l2, l3


def _finish_index(t: pd.DataFrame, starts: np.ndarray):
    """各レースの 1〜3着の位置（同着や欠けがあれば None）。"""
    rank = pd.to_numeric(t["着順"], errors="coerce").to_numpy()
    out = []
    for a, b in zip(starts[:-1], starts[1:]):
        r = rank[a:b]
        pos = []
        for k in (1, 2, 3):
            hit = np.flatnonzero(r == k)
            if len(hit) != 1:
                pos = None
                break
            pos.append(int(hit[0]))
        out.append(pos)
    return out


def fit_discount(table: pd.DataFrame, prob_col: str) -> tuple[float, float, float]:
    """γ・δ を最尤で（格子）。戻り値 (γ, δ, レース数)。"""
    t, starts = race_slices(table)
    fin = _finish_index(t, starts)
    p_all = t[prob_col].to_numpy(dtype="float64")
    s2 = np.zeros(len(GRID))
    s3 = np.zeros(len(GRID))
    n = 0
    for (a, b), pos in zip(zip(starts[:-1], starts[1:]), fin):
        if pos is None or b - a < 3:
            continue
        p = p_all[a:b]
        if not np.isfinite(p).all() or (p <= 0).any():
            continue
        l2, l3 = _place_loglik(p, *pos)
        s2 += l2
        s3 += l3
        n += 1
    return float(GRID[np.argmax(s2)]), float(GRID[np.argmax(s3)]), n


TRI_BINS = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0001]


def trifecta_calibration(table: pd.DataFrame, prob_col: str, gamma: float, delta: float,
                         bins=TRI_BINS) -> tuple[pd.DataFrame, float, int]:
    """全組の予測確率を区間に分け、予測の合計と実際の的中数を比べる。と、実際の組の平均 −log P。"""
    t, starts = race_slices(table)
    fin = _finish_index(t, starts)
    p_all = t[prob_col].to_numpy(dtype="float64")
    edges = np.asarray(bins)
    nb = len(edges) - 1
    pred_sum = np.zeros(nb)
    combos = np.zeros(nb)
    hits = np.zeros(nb)
    nll, n = 0.0, 0
    for (a, b), pos in zip(zip(starts[:-1], starts[1:]), fin):
        if pos is None or b - a < 3:
            continue
        p = p_all[a:b]
        if not np.isfinite(p).all() or (p <= 0).any():
            continue
        P = trifecta_tensor(p, gamma, delta)
        flat = P.ravel()
        valid = flat > 0
        idx = np.clip(np.searchsorted(edges, flat[valid], side="right") - 1, 0, nb - 1)
        np.add.at(pred_sum, idx, flat[valid])
        np.add.at(combos, idx, 1)
        actual = P[pos[0], pos[1], pos[2]]
        hits[np.clip(np.searchsorted(edges, actual, side="right") - 1, 0, nb - 1)] += 1
        nll += -math.log(max(actual, 1e-12))
        n += 1
    out = pd.DataFrame({"区間": [f"[{edges[i]:g}, {edges[i + 1]:g})" for i in range(nb)],
                        "組の数": combos.astype(int), "予測の合計": pred_sum, "実際の的中": hits.astype(int)})
    out["予測の平均"] = out["予測の合計"] / out["組の数"].replace(0, np.nan)
    out["実際の的中率"] = out["実際の的中"] / out["組の数"].replace(0, np.nan)
    return out, (nll / n if n else float("nan")), n


# ---------------------------------------------------------------------------
# 2d 配当の推定
# ---------------------------------------------------------------------------
def combo_probabilities(P: np.ndarray) -> dict[str, np.ndarray]:
    """3連単の確率の表から、各券種の組の確率（3連複・馬単・馬連）。"""
    exacta = P.sum(axis=2)                                   # 馬単 [i,j]
    quinella = exacta + exacta.T                             # 馬連 {i,j}（i<j を使う）
    trio = (P + P.transpose(0, 2, 1) + P.transpose(1, 0, 2) + P.transpose(1, 2, 0)
            + P.transpose(2, 0, 1) + P.transpose(2, 1, 0))   # 3連複 {i,j,k}（並びは問わない）
    return {"3連単": P, "3連複": trio, "馬単": exacta, "馬連": quinella}


def estimated_payout(prob: float, kind: str) -> float:
    """100円あたりの推定配当。"""
    return 100.0 * PAYOUT_RATE[kind] / prob if prob > 0 else float("nan")


def parse_kumi(kumi: str, kind: str) -> tuple[int, ...]:
    k = str(kumi)
    w = 2
    return tuple(int(k[i:i + w]) for i in range(0, len(k), w))


def payout_estimate_check(table: pd.DataFrame, payout: pd.DataFrame, gamma: float, delta: float,
                          market_col: str = "市場確率") -> pd.DataFrame:
    """実際の的中組について、市場の確率からの推定配当と実際の払戻を並べる（1レース・1券種1行）。"""
    t, starts = race_slices(table)
    pay = payout.set_index("レースID")
    posts = pd.to_numeric(t["馬番"], errors="coerce").to_numpy()
    q_all = t[market_col].to_numpy(dtype="float64")
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        rid = t["レースID"].iloc[a]
        if rid not in pay.index or b - a < 3:
            continue
        q = q_all[a:b]
        if not np.isfinite(q).all() or (q <= 0).any():
            continue
        P = trifecta_tensor(q, gamma, delta)
        probs = combo_probabilities(P)
        index_of = {int(pn): i for i, pn in enumerate(posts[a:b]) if np.isfinite(pn)}
        prow = pay.loc[rid]
        for kind in ("3連単", "3連複", "馬単", "馬連"):
            kumis = str(prow.get(f"{kind}組番一覧", "") or "").split(";")
            pays = str(prow.get(f"{kind}払戻一覧", "") or "").split(";")
            if not kumis or not kumis[0] or len(kumis) != 1:      # 同着（的中組が複数）は除く
                continue
            nums = parse_kumi(kumis[0], kind)
            if not all(x in index_of for x in nums):
                continue
            ix = [index_of[x] for x in nums]
            pr = probs[kind][tuple(ix)] if kind in ("3連単", "3連複") else probs[kind][ix[0], ix[1]]
            rows.append({"レースID": rid, "年": pd.Timestamp(t["レース日付"].iloc[a]).year,
                         "重賞": bool(t["重賞"].iloc[a]), "券種": kind,
                         "推定配当": estimated_payout(float(pr), kind), "実際の払戻": float(pays[0])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["比（実際÷推定）"] = out["実際の払戻"] / out["推定配当"]
    return out


def payout_check_summary(check: pd.DataFrame, by: str = "券種") -> pd.DataFrame:
    def agg(g):
        r = np.log(g["比（実際÷推定）"])
        return pd.Series({
            "件数": len(g),
            "比の中央値": float(np.exp(r.median())),
            "比の25%": float(np.exp(r.quantile(0.25))), "比の75%": float(np.exp(r.quantile(0.75))),
            "±25%以内": float((np.abs(r) <= np.log(1.25)).mean()),
            "±50%以内": float((r.between(np.log(1 / 1.5), np.log(1.5))).mean()),
            "相関（log）": float(np.corrcoef(np.log(g["推定配当"]), np.log(g["実際の払戻"]))[0, 1])
            if len(g) > 2 else np.nan,
        })
    return check.groupby(by, sort=False).apply(agg, include_groups=False).reset_index()


def odds_band(est: pd.Series) -> pd.Series:
    return pd.cut(est, [0, 1000, 5000, 20000, 50000, 200000, np.inf],
                  labels=["〜1千", "1千〜5千", "5千〜2万", "2万〜5万", "5万〜20万", "20万〜"], right=False)


def wilson(h: int, n: int) -> tuple[float, float]:
    return trifecta.wilson_interval(h, n)
