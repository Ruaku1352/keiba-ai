"""v8e 2・3着への来やすさのテスト。

- 段ごとの強さを馬ごとに変えた Henery 型：x = 0 なら今の Henery 型と同じ、確率の合計 1
- (γ, φ, δ, ψ) の推定：合成データで元の値に戻る
- P1 の来やすさ：その日より前のレースだけから（as_of=day）
- P2：3連複の確率から 3着以内の確率
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import market, place  # noqa: E402


def test_stage_tensor_matches_henery_without_term():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(9))
    coef = {"γ": 0.83, "δ": 0.71, "φ": 0.5, "ψ": -0.3}
    w2, w3 = place.stage_weights(p, coef)                      # x なし → p^γ, p^δ
    assert np.allclose(place.stage_tensor(p, w2, w3), market.trifecta_tensor(p, 0.83, 0.71))
    w2, w3 = place.stage_weights(p, coef, np.zeros(9))
    assert np.allclose(place.stage_tensor(p, w2, w3), market.trifecta_tensor(p, 0.83, 0.71))
    x = rng.normal(0, 1, 9)
    w2, w3 = place.stage_weights(p, coef, x)
    P = place.stage_tensor(p, w2, w3)
    assert P.sum() == pytest.approx(1.0)
    m = place.marginals_batch(P[None])[0]
    assert np.allclose(m[:, 0], p)                             # 1着の段は変わらない
    assert np.allclose(m.sum(axis=0), 1.0)
    # φ > 0 なら x の大きい馬ほど 2着の確率が上がる
    hi = np.argmax(x)
    w2b, w3b = place.stage_weights(p, {"γ": 0.83, "δ": 0.71})
    assert m[hi, 1] > place.marginals_batch(place.stage_tensor(p, w2b, w3b)[None])[0][hi, 1]


def _simulate(n_races=3000, field=10, coef=None, seed=1):
    rng = np.random.default_rng(seed)
    coef = coef or {"γ": 0.8, "δ": 0.7, "φ": 0.6, "ψ": 0.4}
    rows = []
    for r in range(n_races):
        p = rng.dirichlet(np.ones(field) * 1.5)
        x = rng.normal(0, 1, field)
        w2, w3 = place.stage_weights(p, coef, x)
        P = place.stage_tensor(p, w2, w3).ravel()
        i, j, k = np.unravel_index(rng.choice(len(P), p=P / P.sum()), (field,) * 3)
        rank = np.full(field, 9)
        rank[[i, j, k]] = [1, 2, 3]
        for h in range(field):
            rows.append({"レースID": f"R{r:05d}", "馬番": h + 1, "着順": int(rank[h]), "p": p[h], "x": x[h]})
    return pd.DataFrame(rows), coef


def test_fit_stages_recovers_coefficients():
    t, coef = _simulate()
    fit = place.fit_stages(t, "p", "x")
    for k in ("γ", "δ", "φ", "ψ"):
        assert fit[k] == pytest.approx(coef[k], abs=0.12), k
    # x の効果がある分、並びの −log 尤度は x を入れた方が小さい
    plain = place.fit_stages(t, "p")
    assert place.stage_loglik(t, "p", fit, "x") < place.stage_loglik(t, "p", plain)


def test_fit_stages_without_term_close_to_grid():
    t, _ = _simulate(coef={"γ": 0.8, "δ": 0.7, "φ": 0.0, "ψ": 0.0}, seed=2)
    fit = place.fit_stages(t, "p")
    g, d, _ = market.fit_discount(t.rename(columns={"p": "prob"}).assign(win=(t["着順"] == 1).astype(int)), "prob")
    assert fit["γ"] == pytest.approx(g, abs=0.02) and fit["δ"] == pytest.approx(d, abs=0.02)


def make_results(n_days=50, races_per_day=4, horses=60, field=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, horses)
    placer = rng.normal(0, 1, horses)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2012-01-07") + pd.Timedelta(days=7 * d)
        runners = rng.permutation(horses)[: races_per_day * field]
        for r in range(races_per_day):
            hs = runners[r * field:(r + 1) * field]
            score = ability[hs] + rng.normal(0, 1, field)
            order = np.argsort(-score)
            odds = 1 / np.exp(ability[hs]) / np.exp(ability[hs]).sum() ** -1 * 1.25
            for rank, pos in enumerate(order, 1):
                rows.append({"レースID": f"{date:%Y%m%d}05{r + 1:02d}", "レース日付": date,
                             "血統登録番号": f"H{hs[pos]:04d}", "着順": rank, "馬番": int(pos) + 1,
                             "単勝オッズ": float(odds[pos])})
    return pd.DataFrame(rows)


def test_place_feature_no_leak_as_of_day():
    df = make_results()
    cutoff = pd.Timestamp("2012-04-01")
    base, gd = place.past_place_feature(df, cutoff)
    assert gd["推定レース数"] > 0
    dates = sorted(df["レース日付"].unique())
    D = dates[30]
    on_d = set(df.loc[df["レース日付"] == D, "レースID"])
    k = ["レースID", "血統登録番号"]
    cols = [place.PLACE_TERM, "過去の有効走数", "過去の1着", "過去の2・3着"]

    def on(t):
        return t.loc[t["レースID"].isin(on_d)].sort_values(k).reset_index(drop=True)[k + cols]

    alt = df.copy()
    m = alt["レース日付"] >= D                          # その日以降の着順を逆にしても変わらない
    alt.loc[m, "着順"] = alt.loc[m].groupby("レースID")["着順"].transform(lambda s: s.max() + 1 - s)
    pd.testing.assert_frame_equal(on(base), on(place.past_place_feature(alt, cutoff)[0]))
    alt2 = df.copy()
    m = alt2["レース日付"] == dates[29]                 # 前の開催日の着順を変えると変わる
    alt2.loc[m, "着順"] = alt2.loc[m].groupby("レースID")["着順"].transform(lambda s: s.max() + 1 - s)
    assert not np.allclose(on(base)[place.PLACE_TERM], on(place.past_place_feature(alt2, cutoff)[0])[place.PLACE_TERM])
    # 初出走は 0
    first = base.merge(df[k + ["レース日付"]]).sort_values("レース日付").groupby("血統登録番号").head(1)
    assert (first[place.PLACE_TERM] == 0).all() and (first["過去の有効走数"] == 0).all()


def test_place_feature_cutoff_only_uses_early_races():
    """過去のレースの予測に使う γ・δ は cutoff より前のレースだけで推定（後ろの結果を変えても同じ）。"""
    df = make_results()
    cutoff = pd.Timestamp("2012-04-01")
    _, a = place.past_place_feature(df, cutoff)
    alt = df.copy()
    m = alt["レース日付"] >= cutoff
    alt.loc[m, "着順"] = alt.loc[m].groupby("レースID")["着順"].transform(lambda s: s.max() + 1 - s)
    _, b = place.past_place_feature(alt, cutoff)
    assert a == b


def test_place_feature_counts_and_sign():
    """毎回 2・3着に来る（1着の確率から見込まれるより多い）馬は、来やすさがプラス。"""
    rows = []
    for d in range(8):
        date = pd.Timestamp("2012-01-07") + pd.Timedelta(days=7 * d)
        order = ["A", "P", "B", "C", "D", "E"]          # P は毎回2着（単勝は5番人気）
        odds = {"A": 2.0, "B": 4.0, "C": 6.0, "D": 8.0, "P": 12.0, "E": 20.0}
        for rank, h in enumerate(order, 1):
            rows.append({"レースID": f"{date:%Y%m%d}0501", "レース日付": date, "血統登録番号": h,
                         "着順": rank, "馬番": "ABCDEP".index(h) + 1, "単勝オッズ": odds[h]})
    f, _ = place.past_place_feature(pd.DataFrame(rows), pd.Timestamp("2012-01-01"))
    last = f.loc[f["レースID"] == f["レースID"].max()].set_index("血統登録番号")
    assert last.loc["P", "過去の有効走数"] == 7 and last.loc["P", "過去の2・3着"] == 7 and last.loc["P", "過去の1着"] == 0
    assert last.loc["P", place.PLACE_TERM] > 0 > last.loc["E", place.PLACE_TERM]


def test_trio_top3():
    rng = np.random.default_rng(3)
    posts = np.array([1, 2, 4, 5, 7, 8])
    p = rng.dirichlet(np.ones(6))
    P = market.trifecta_tensor(p, 0.8, 0.7)
    trio = market.combo_probabilities(P)["3連複"]
    combos = list(itertools.combinations(range(6), 3))
    horses = np.array([[posts[i] for i in c] for c in combos])
    pay = np.array([100 * 0.75 / trio[c] for c in combos])
    top3 = place.trio_top3(horses, pay, posts)
    assert top3.sum() == pytest.approx(3.0)
    assert np.allclose(top3, place.marginals_batch(P[None])[0].sum(axis=1))
