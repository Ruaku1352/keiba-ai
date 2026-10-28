"""v8c レーティング（Weng-Lin の Plackett-Luce を逐次で）のテスト。

- 1レースの更新が式どおり（素直なループの実装と一致）・上位は上がり下位は下がる
- リーク：その日のレースのレーティングは前日までの結果だけから（その日以降の結果を変えても変わらない）
- 不確かさ：出走が増えると小さく、休むと大きく。伸び = μ − 3走前の μ
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ratings  # noqa: E402


def _reference_update(mu, sig2):
    """OpenSkill の PlackettLuce（同着なし・1頭1チーム）をそのままループで書いたもの。"""
    n = len(mu)
    c = np.sqrt(np.sum(sig2 + ratings.BETA ** 2))
    e = np.exp(mu / c)
    S = np.array([e[q:].sum() for q in range(n)])
    new_mu, new_s2 = mu.copy(), sig2.copy()
    for i in range(n):
        omega = delta = 0.0
        for q in range(i + 1):
            r = e[i] / S[q]
            omega += (1 - r) if q == i else -r
            delta += r * (1 - r)
        gamma = np.sqrt(sig2[i]) / c
        new_mu[i] = mu[i] + sig2[i] / c * omega
        new_s2[i] = sig2[i] * max(1 - gamma * sig2[i] / c ** 2 * delta, ratings.KAPPA)
    return new_mu, new_s2


def test_update_matches_reference_and_direction():
    rng = np.random.default_rng(0)
    for n in (2, 5, 18):
        mu = rng.normal(25, 3, n)
        s2 = rng.uniform(1, ratings.SIGMA0 ** 2, n)
        a = ratings.update_race(mu, s2)
        b = _reference_update(mu, s2)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    mu = np.full(8, 25.0)
    s2 = np.full(8, ratings.SIGMA0 ** 2)
    new_mu, new_s2 = ratings.update_race(mu, s2)
    assert new_mu[0] > 25 > new_mu[-1]
    assert np.all(np.diff(new_mu) < 0)              # 着順どおりに並ぶ
    assert np.all(new_s2 < s2)


def test_beating_strong_horses_counts_more():
    """同じ1着でも、強い相手（μ が高い）に勝った方が大きく上がる。"""
    s2 = np.full(6, 4.0)
    weak = ratings.update_race(np.array([25.0] + [20.0] * 5), s2)[0][0] - 25
    strong = ratings.update_race(np.array([25.0] + [30.0] * 5), s2)[0][0] - 25
    assert strong > weak > 0


def make_results(n_days=60, races_per_day=4, horses=60, field=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, horses)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2015-01-03") + pd.Timedelta(days=7 * d)
        runners = rng.permutation(horses)[: races_per_day * field]
        for r in range(races_per_day):
            hs = runners[r * field:(r + 1) * field]
            order = hs[np.argsort(-(ability[hs] + rng.normal(0, 1, field)))]
            rid = f"{date:%Y%m%d}05{r + 1:02d}"
            for rank, h in enumerate(order, 1):
                rows.append({"レースID": rid, "レース日付": date, "血統登録番号": f"H{h:04d}", "着順": rank,
                             "馬番": int(np.where(hs == h)[0][0]) + 1})
    return pd.DataFrame(rows)


def test_no_leak_as_of_day():
    df = make_results()
    base = ratings.compute_ratings(df)
    dates = sorted(df["レース日付"].unique())
    D = dates[30]
    on_d = df.loc[df["レース日付"] == D, "レースID"].unique()

    # D 以降（D を含む）の着順を逆にしても、D のレースのレーティングは変わらない
    alt = df.copy()
    m = alt["レース日付"] >= D
    alt.loc[m, "着順"] = alt.loc[m].groupby("レースID")["着順"].transform(lambda s: s.max() + 1 - s)
    changed = ratings.compute_ratings(alt)
    k = ["レースID", "血統登録番号"]
    a = base.loc[base["レースID"].isin(on_d)].sort_values(k).reset_index(drop=True)
    b = changed.loc[changed["レースID"].isin(on_d)].sort_values(k).reset_index(drop=True)
    cols = ["μ", "σ", "出走数"] + ratings.RATING_TERMS
    pd.testing.assert_frame_equal(a[k + cols], b[k + cols])

    # 前日（前の開催日）の着順を変えると変わる（テストが空振りしていないことの確認）
    alt2 = df.copy()
    m = alt2["レース日付"] == dates[29]
    alt2.loc[m, "着順"] = alt2.loc[m].groupby("レースID")["着順"].transform(lambda s: s.max() + 1 - s)
    c = ratings.compute_ratings(alt2)
    c = c.loc[c["レースID"].isin(on_d)].sort_values(k).reset_index(drop=True)
    assert not np.allclose(a["μ"], c["μ"])


def test_same_day_results_not_used():
    """同じ日の前のレースの結果は使わない（as_of=day）。同じ日に2回走る馬がいても、その日のうちは更新しない。"""
    rows = []
    for r, order in enumerate([["A", "B", "C", "D"], ["A", "B", "C", "D"]], 1):
        for rank, h in enumerate(order, 1):
            rows.append({"レースID": f"2015010305{r:02d}", "レース日付": pd.Timestamp("2015-01-03"),
                         "血統登録番号": h, "着順": rank, "馬番": rank})
    t = ratings.compute_ratings(pd.DataFrame(rows))
    assert np.allclose(t["μ"], ratings.MU0) and (t["出走数"] == 0).all()


def test_uncertainty_and_growth():
    df = make_results()
    t = ratings.compute_ratings(df).merge(df[["レースID", "血統登録番号", "レース日付"]])
    # 出走数が多い馬ほど σ が小さい
    first = t.loc[t["出走数"] == 0, "σ"]
    many = t.loc[t["出走数"] >= 10, "σ"]
    assert np.allclose(first, ratings.SIGMA0) and many.mean() < first.mean()
    # 伸び = μ − 3走前の μ（3走未満は 0）
    h = t.loc[t["血統登録番号"] == t["血統登録番号"].iloc[0]].sort_values("レース日付").reset_index(drop=True)
    assert (h.loc[h["出走数"] < 3, "レーティング_伸び"] == 0).all()
    for i in range(3, len(h)):
        assert h.loc[i, "レーティング_伸び"] == pytest.approx(h.loc[i, "μ"] - h.loc[i - 3, "μ"])
    # レース内の相対値は平均 0
    assert np.allclose(t.groupby("レースID")["レーティング_相対"].sum(), 0, atol=1e-9)


def test_layoff_increases_uncertainty():
    def run(gap_days):
        rows = []
        for j, day in enumerate([0, 7, 7 + gap_days]):
            date = pd.Timestamp("2015-01-03") + pd.Timedelta(days=day)
            for rank, h in enumerate(["A", "B", "C", "D"], 1):
                rows.append({"レースID": f"{date:%Y%m%d}0501", "レース日付": date, "血統登録番号": h,
                             "着順": rank, "馬番": rank})
        t = ratings.compute_ratings(pd.DataFrame(rows))
        return t.loc[t["出走数"] == 2, "σ"].mean()
    assert run(365) > run(7)


def test_speed_estimate():
    """1レースあたりの時間（1986年からの約14万レースの見積もりに使う）。"""
    df = make_results(n_days=250, races_per_day=12, horses=1500, field=14)
    t0 = time.perf_counter()
    ratings.compute_ratings(df)
    per_race = (time.perf_counter() - t0) / df["レースID"].nunique()
    print(f"\n1レース {per_race * 1e3:.3f} ミリ秒 → 14万レースで約 {per_race * 140_000:.0f} 秒")
    assert per_race < 0.01
