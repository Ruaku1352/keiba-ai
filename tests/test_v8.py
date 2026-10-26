"""v8（有馬記念に向けた最後の改良の検証）のテスト。

- 1.5 格・着差・推移・タイム指数：コード表の対応、過去の行だけから作られること
- 3 当日の馬場の傾向：発走時刻の打ち切り（後のレース・同時刻のレースを使わない）
- 2 市場の確率・3連単の確率・配当の推定・買い目
- 重賞のガード：新しい案の成績を重賞で出そうとするとエラー
- 本番の特徴量（extra なし）は変わらない
- CLI（py -m src.v8 ...）が合成データで最後まで動く
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (bets, config, experiment, features, jvmap, levels, market, pace,  # noqa: E402
                 predict, preprocess, trackbias, trifecta, validate)

pace.PACE_Z_MIN_COUNT = 5
pace.EXPECTED_PACE_MIN_COUNT = 5
levels.TIME_INDEX_MIN_COUNT = 5
trackbias.L3_MIN_COUNT = 3


# ===========================================================================
# 合成の生データ（1日に2場 × 6R、発走時刻は 30分おき。能力の高い馬ほど勝ちやすく、人気にもなる）
# ===========================================================================
JYOKEN = ["701", "703", "005", "010", "016", "999"]      # 新馬・未勝利・1勝・2勝・3勝・OP


def _key(day, jyo, kai, nichi, race_no):
    return {"id.Year": f"{day.year:04d}", "id.MonthDay": day.strftime("%m%d"), "id.JyoCD": jyo,
            "id.Kaiji": f"{kai:02d}", "id.Nichiji": f"{nichi:02d}", "id.RaceNum": f"{race_no:02d}",
            "head.MakeDate.Year": f"{day.year:04d}", "head.MakeDate.Month": day.strftime("%m"),
            "head.MakeDate.Day": day.strftime("%d")}


def _time_str(seconds):
    m = int(seconds // 60)
    return f"{m}{round((seconds - 60 * m) * 10):03d}"


def make_raw_v8(n_days=70, venues=("05", "08"), races=6, n_horses=10, start="2012-01-08", seed=0):
    rng = np.random.default_rng(seed)
    pool = [f"2010{i:06d}" for i in range(n_horses * races * len(venues) * 2)]
    ability = {h: rng.normal() for h in pool}
    jockeys = [f"0{i:04d}" for i in range(n_horses + 6)]
    trainers = [f"1{i:04d}" for i in range(8)]
    ra, se, hr = [], [], []
    seq = 0
    for d in range(n_days):
        day = pd.Timestamp(start) + pd.Timedelta(days=7 * d)
        runners_today = rng.choice(pool, size=n_horses * races * len(venues), replace=False)
        slot = 0
        for v_i, jyo in enumerate(venues):
            for r in range(1, races + 1):
                key = _key(day, jyo, 1 + d // 8, 1 + d % 8, r)
                graded = r == races and d % 3 == 0
                grade = ["A", "B", "C"][d % 3] if graded else " "
                jyoken = "999" if graded else JYOKEN[(r - 1) % len(JYOKEN)]
                track = ["11", "23"][(r + v_i) % 2]
                dist = [1200, 1600, 1800, 2000][(r + d) % 4]
                start_hm = 1000 + 30 * (r - 1)
                hh, mm = divmod(start_hm - 1000 + 600, 60)
                ra.append({"_seq": str(seq), "head.RecordSpec": "RA", "head.DataKubun": "7", **key,
                           "RaceInfo.Hondai": f"テスト{d}-{jyo}-{r}", "RaceInfo.Ryakusyo10": f"テ{d}{r}",
                           "GradeCD": grade, "JyokenInfo.SyubetuCD": "13",
                           **{f"JyokenInfo.JyokenCD[{i}]": jyoken for i in range(1, 6)},
                           "Kyori": str(dist), "TrackCD": track, "TenkoBaba.TenkoCD": "1",
                           "TenkoBaba.SibaBabaCD": str(1 + d % 3), "TenkoBaba.DirtBabaCD": str(1 + d % 2),
                           "SyussoTosu": f"{n_horses:02d}", "HassoTime": f"{hh:02d}{mm:02d}",
                           "HaronTimeS3": str(345 + rng.integers(-15, 15)),
                           "HaronTimeL3": str(350 + rng.integers(-15, 15))})
                seq += 1
                runners = runners_today[slot * n_horses:(slot + 1) * n_horses]
                slot += 1
                strength = np.array([ability[h] for h in runners]) + rng.gumbel(size=n_horses) * 0.9
                order = np.argsort(np.argsort(-strength)) + 1
                mkt = np.exp(np.array([ability[h] for h in runners]))
                mkt = mkt / mkt.sum()
                odds = np.round(0.8 / mkt, 1)
                ninki = np.argsort(np.argsort(odds)) + 1
                corner = rng.permutation(n_horses) + 1
                race_jockeys = rng.choice(jockeys, size=n_horses, replace=False)
                for i, (h, rank) in enumerate(zip(runners, order)):
                    td = "-003" if rank == 1 else f"+{min(999, (rank - 1) * 3 + rng.integers(0, 3)):03d}"
                    se.append({"_seq": str(seq), "head.RecordSpec": "SE", "head.DataKubun": "7", **key,
                               "Wakuban": str(1 + i // 2), "Umaban": f"{i + 1:02d}", "KettoNum": h,
                               "Bamei": f"ウマ{h[-4:]}", "SexCD": "1", "Barei": "04",
                               "ChokyosiCode": rng.choice(trainers), "ChokyosiRyakusyo": "調教師",
                               "KisyuCode": race_jockeys[i], "KisyuRyakusyo": "騎手", "Futan": "550",
                               "BaTaijyu": str(470 + rng.integers(-20, 20)), "ZogenFugo": "+",
                               "ZogenSa": "002", "IJyoCD": "0", "KakuteiJyuni": f"{rank:02d}",
                               "Time": _time_str(dist / 16.5 + rank * 0.2), "Jyuni1c": "00", "Jyuni2c": "00",
                               "Jyuni3c": f"{corner[i]:02d}", "Jyuni4c": f"{corner[i]:02d}",
                               "Odds": f"{int(min(9999, odds[i] * 10)):04d}", "Ninki": f"{ninki[i]:02d}",
                               "HaronTimeL4": "000", "HaronTimeL3": str(340 + rank), "TimeDiff": td,
                               "DMTime": "13450", "DMJyuni": "01",
                               "KyakusituKubun": str(1 + (corner[i] - 1) * 4 // n_horses)})
                    seq += 1
                top = [i + 1 for i in np.argsort(order)[:3]]
                row = {"_seq": str(seq), "head.RecordSpec": "HR", "head.DataKubun": "2", **key}
                for j in range(1, 10):
                    row[f"FuseirituFlag[{j}]"] = row[f"TokubaraiFlag[{j}]"] = row[f"HenkanFlag[{j}]"] = "0"
                for prefix, n in (("PaySanrentan", 6), ("PaySanrenpuku", 3), ("PayUmatan", 6), ("PayUmaren", 3)):
                    for j in range(1, n + 1):
                        row[f"{prefix}[{j}].Kumi"] = "000000" if "Sanren" in prefix else "0000"
                        row[f"{prefix}[{j}].Pay"] = "000000000"
                p3 = 0.725 / (mkt[top[0] - 1] * mkt[top[1] - 1] / (1 - mkt[top[0] - 1])
                              * mkt[top[2] - 1] / (1 - mkt[top[0] - 1] - mkt[top[1] - 1]))
                row["PaySanrentan[1].Kumi"] = "".join(f"{w:02d}" for w in top)
                row["PaySanrentan[1].Pay"] = f"{int(min(p3 * 100, 99999999)):09d}"
                row["PaySanrenpuku[1].Kumi"] = "".join(f"{w:02d}" for w in sorted(top))
                row["PaySanrenpuku[1].Pay"] = f"{int(min(p3 * 100 / 5, 99999999)):09d}"
                row["PayUmatan[1].Kumi"] = "".join(f"{w:02d}" for w in top[:2])
                row["PayUmatan[1].Pay"] = f"{int(min(p3 * 100 / 20, 99999999)):09d}"
                row["PayUmaren[1].Kumi"] = "".join(f"{w:02d}" for w in sorted(top[:2]))
                row["PayUmaren[1].Pay"] = f"{int(min(p3 * 100 / 40, 99999999)):09d}"
                hr.append(row)
                seq += 1
    return {"RA": pd.DataFrame(ra), "SE": pd.DataFrame(se), "HR": pd.DataFrame(hr)}


@pytest.fixture(scope="module")
def ds8():
    return jvmap.build_dataset(raw=make_raw_v8())


FOLDS = [validate.Fold("1", (2012, 2012), (2013, 2013))]


# ===========================================================================
# 1.5 格・着差
# ===========================================================================
def test_jyoken_level_mapping():
    assert levels.jyoken_level("701") == 0 and levels.jyoken_level("702") == 0
    assert levels.jyoken_level("703") == 1
    assert levels.jyoken_level("005") == 2 and levels.jyoken_level("004") == 2
    assert levels.jyoken_level("010") == 3 and levels.jyoken_level("009") == 3
    assert levels.jyoken_level("016") == 4 and levels.jyoken_level("015") == 4
    assert levels.jyoken_level("999") == 5
    assert np.isnan(levels.jyoken_level("000")) and np.isnan(levels.jyoken_level(" "))


def test_race_class_level_uses_grade_and_max_condition():
    cols = [f"JyokenInfo.JyokenCD[{i}]" for i in range(1, 6)]
    rows = [  # GradeCD, 2歳・3歳・4歳・5歳以上・最若年 の条件
        (" ", ["000", "005", "010", "010", "005"]),   # 3歳 500万下・4歳以上 1000万下 → 2勝相当（3）
        ("A", ["000", "999", "999", "999", "999"]),   # G1 → 9
        ("L", ["000", "999", "999", "999", "999"]),   # リステッド → 6
        (" ", ["000", "000", "000", "000", "000"]),   # 不明 → NaN
        ("C", ["000", "999", "999", "999", "999"]),   # G3 → 7
        (" ", ["701", "000", "000", "000", "701"]),   # 新馬 → 0
        (" ", ["000", "703", "000", "000", "703"]),   # 未勝利 → 1
        (" ", ["000", "999", "999", "999", "999"]),   # オープン → 5
    ]
    ra = pd.DataFrame([{"GradeCD": g, **dict(zip(cols, c))} for g, c in rows])
    lv = levels.race_class_level(ra)
    assert lv[[0, 1, 2, 4, 5, 6, 7]].tolist() == [3.0, 9.0, 6.0, 7.0, 0.0, 1.0, 5.0]
    assert np.isnan(lv[3])


def test_time_diff_parsing():
    s = pd.Series(["+005", "-003", "9999", "", "+123"])
    got = levels.parse_time_diff(s)
    assert got[0] == 0.5 and got[1] == -0.3 and np.isnan(got[2]) and np.isnan(got[3]) and got[4] == 12.3


def _level_frame(ds, scramble_race=None, seed=0):
    rr = ds.race_result.copy()
    if scramble_race is not None:
        m = rr["レースID"] == scramble_race
        rng = np.random.default_rng(seed)
        rr.loc[m, "着順"] = rng.permutation(rr.loc[m, "着順"].to_numpy())
        rr.loc[m, "_タイム差秒"] = rng.uniform(-1, 5, m.sum())
        rr.loc[m, "タイム"] = rr.loc[m, "タイム"] + rng.uniform(-3, 3, m.sum())
    df = preprocess.basic_clean(rr)
    return features.add_all_features(df, lap_df=ds.lap_df, as_of="day", extra={"levels": {}})


def test_level_features_do_not_use_own_result(ds8):
    rid = ds8.race_result["レースID"].unique()[500]
    a = _level_frame(ds8)
    b = _level_frame(ds8, scramble_race=rid)
    cols = levels.level_feature_columns()
    ra = a.loc[a["レースID"] == rid].set_index("血統登録番号")[cols].sort_index()
    rb = b.loc[b["レースID"] == rid].set_index("血統登録番号")[cols].sort_index()
    pd.testing.assert_frame_equal(ra, rb)
    assert ra["今回の格"].notna().all() and ra["前走の格"].notna().any()


def test_level_features_values(ds8):
    df = _level_frame(ds8)
    h = df.groupby("血統登録番号").filter(lambda g: len(g) >= 4)
    hid = h["血統登録番号"].iloc[0]
    g = df.loc[df["血統登録番号"] == hid]
    lv = g["今回の格"].to_numpy()
    assert np.isnan(g["前走の格"].iloc[0]) and g["前走の格"].iloc[1] == lv[0]
    assert g["過去最高格"].iloc[2] == max(lv[:2])
    assert g["格の差_前走"].iloc[1] == lv[1] - lv[0]
    rr = ds8.race_result.set_index(["レースID", "血統登録番号"])
    first_td = rr.loc[(g["レースID"].iloc[0], hid), "_タイム差秒"]
    assert g["前走タイム差"].iloc[1] == pytest.approx(first_td)


def test_production_features_unchanged_without_extra(ds8):
    df = preprocess.basic_clean(ds8.race_result)
    plain = features.add_all_features(df, lap_df=ds8.lap_df, as_of="day")
    cols = features.feature_columns(plain)
    for c in levels.level_feature_columns() + trackbias.track_bias_feature_columns() + ["市場確率_log"]:
        assert c not in plain.columns and c not in cols


# ===========================================================================
# 3 当日の馬場の傾向
# ===========================================================================
def _tb(ds, lag_cutoff=20, delay=20, rr=None):
    rr = ds.race_result if rr is None else rr
    df = preprocess.basic_clean(rr)
    return features.add_all_features(df, lap_df=ds.lap_df, as_of="day",
                                     extra={"trackbias": {"cutoff_min": lag_cutoff, "delay_min": delay}})


def test_start_minutes():
    assert list(trackbias.start_minutes(pd.Series(["1025", "0930", "930", "", "xx"])).fillna(-1)) == \
        [625, 570, 570, -1, -1]


def test_track_bias_uses_only_races_started_early_enough(ds8):
    df = _tb(ds8)
    first = df.loc[df["レースID"].str[-2:] == "01"]
    assert (first["当日_使えたレース数"] == 0).all() and (first["当日_内枠傾向"] == 0).all()
    # 30分おき・ラグ40分：nR（発走 10:00 + 30(n−1)）で使えるのは 発走 ≤ 発走(n) − 40分 のレース。
    # 芝ダが交互なので、同じ芝ダの先行レースは 2R おき
    races = df.drop_duplicates("レースID").copy()
    races["R"] = races["レースID"].str[-2:].astype(int)
    for _, r in races.iterrows():
        n = r["R"]
        same_surface_before = [m for m in range(1, n) if (m - n) % 2 == 0 and 30 * (n - m) >= 40]
        assert r["当日_使えたレース数"] == len(same_surface_before), (n, r["当日_使えたレース数"])


def test_track_bias_boundary_minutes():
    races = pd.DataFrame({"レースID": ["a", "b", "c", "d"], "日付": pd.to_datetime(["2020-01-01"] * 4),
                          "場": "05", "芝ダ": "芝", "発走分": [600, 640, 641, 640],
                          "内枠傾向": [1.0, 2.0, 4.0, 8.0], "前残り傾向": 0.0, "上がり傾向": 0.0})
    out = trackbias.day_bias(races, cutoff_min=20, delay_min=20).set_index("レースID")
    assert out.loc["b", "当日_使えたレース数"] == 1          # 600 ≤ 640 − 40
    assert out.loc["c", "当日_使えたレース数"] == 1          # 641 − 40 = 601 → a だけ（b,d は 640 > 601）
    assert out.loc["a", "当日_使えたレース数"] == 0
    out60 = trackbias.day_bias(races, cutoff_min=60, delay_min=20).set_index("レースID")
    assert (out60["当日_使えたレース数"] == 0).all()
    assert out.loc["b", "当日_内枠傾向"] == pytest.approx(1.0 / 2)   # 合計 / (件数 + 1)


def test_track_bias_ignores_later_same_time_and_own_results(ds8):
    base = _tb(ds8)
    rr = ds8.race_result.copy()
    day = rr["レース日付"].iloc[len(rr) // 2]
    jyo = "05"
    target = rr.loc[(rr["レース日付"] == day) & (rr["競馬場コード"] == jyo) & (rr["レースID"].str[-2:] == "04"),
                    "レースID"].iloc[0]
    # 4R 自身・4R より後・別の場のレースの結果を書き換えても、4R の傾向は変わらない
    m = (rr["レース日付"] == day) & ((rr["レースID"].str[-2:].astype(int) >= 4) | (rr["競馬場コード"] != jyo))
    rng = np.random.default_rng(3)
    rr.loc[m, "着順"] = rr.loc[m].groupby("レースID")["着順"].transform(lambda s: rng.permutation(s.to_numpy()))
    rr.loc[m, "_4コーナー順位"] = rng.integers(1, 11, m.sum())
    rr.loc[m, "_後3F"] = rr.loc[m, "_後3F"] + 2.0
    after = _tb(ds8, rr=rr)
    cols = trackbias.track_bias_feature_columns()
    a = base.loc[base["レースID"] == target].set_index("血統登録番号")[cols].sort_index()
    b = after.loc[after["レースID"] == target].set_index("血統登録番号")[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)
    # 2R（4R の 60 分前、同じ芝ダ）を書き換えると変わる（＝ちゃんと使っている）
    rr2 = ds8.race_result.copy()
    r2 = target[:-2] + "02"
    m2 = rr2["レースID"] == r2
    rr2.loc[m2, "着順"] = rr2.loc[m2, "着順"].to_numpy()[::-1]
    c = _tb(ds8, rr=rr2).loc[lambda d: d["レースID"] == target].set_index("血統登録番号")[cols].sort_index()
    assert not c.equals(a)


# ===========================================================================
# 2 市場・3連単・配当・買い目
# ===========================================================================
def test_trifecta_tensor_sums_and_harville():
    p = np.array([0.5, 0.3, 0.2])
    P = market.trifecta_tensor(p)
    assert P.sum() == pytest.approx(1.0)
    assert P[0, 1, 2] == pytest.approx(0.5 * 0.3 / 0.5 * 1.0)
    Pd = market.trifecta_tensor(np.array([0.4, 0.3, 0.2, 0.1]), 0.8, 0.7)
    assert Pd.sum() == pytest.approx(1.0)
    assert np.allclose(Pd.sum(axis=(1, 2)), [0.4, 0.3, 0.2, 0.1])


def test_conditional_logit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X, y, starts = [], [], [0]
    for _ in range(3000):
        n = rng.integers(8, 15)
        a, b = np.log(rng.dirichlet(np.ones(n))), np.log(rng.dirichlet(np.ones(n)))
        u = 0.5 * a + 1.0 * b
        pr = np.exp(u - u.max())
        w = rng.choice(n, p=pr / pr.sum())
        X.append(np.c_[a, b])
        yy = np.zeros(n)
        yy[w] = 1
        y.append(yy)
        starts.append(starts[-1] + n)
    beta = market.fit_conditional_logit(np.vstack(X), np.array(starts), np.concatenate(y))
    assert beta[0] == pytest.approx(0.5, abs=0.12) and beta[1] == pytest.approx(1.0, abs=0.12)


def test_combo_probabilities_consistency():
    P = market.trifecta_tensor(np.array([0.4, 0.3, 0.2, 0.1]))
    c = market.combo_probabilities(P)
    assert c["馬単"][0, 1] == pytest.approx(P[0, 1, :].sum())
    assert c["馬連"][0, 1] == pytest.approx(P[0, 1, :].sum() + P[1, 0, :].sum())
    assert c["3連複"][0, 1, 2] == pytest.approx(sum(P[x] for x in itertools.permutations((0, 1, 2))))
    assert market.estimated_payout(0.01, "3連単") == pytest.approx(7250)


def test_bets_plans_respect_budget_and_target():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(14) * 0.8)
    q = rng.dirichlet(np.ones(14) * 0.8)
    posts = np.arange(1, 15)
    P = market.trifecta_tensor(p, 0.85, 0.75)
    Q = market.combo_probabilities(market.trifecta_tensor(q, 0.85, 0.75))
    box = bets.box(p, posts)
    assert len(box.bets) == 120 and box.invest == 12_000
    hot = bets.hot_top(P, Q["3連単"], posts)
    assert len(hot.bets) <= 120
    for b in hot.bets:
        ix = tuple(x - 1 for x in b.combo)
        assert market.estimated_payout(Q["3連単"][ix], "3連単") >= bets.TARGET
    form, cards = bets.formation_cards(P, Q["3連単"], p, posts)
    assert len(cards) <= bets.MAX_CARDS_FORMATION and form.invest <= 12_000
    covered = {b.combo for b in form.bets}
    for a, b2, c in cards:
        for x in itertools.product(a, b2, c):
            if len(set(x)) == 3:
                assert x in covered
    for kinds, cap in [(bets.KINDS, None), (bets.KINDS, 2), (("3連単",), None)]:
        plan = bets.mixed(P, Q, p, posts, kinds=kinds, max_per_kind=cap)
        assert plan.invest <= bets.BUDGET
        assert all(b.kind in kinds for b in plan.bets)
        for b in plan.bets:
            assert b.stake % 100 == 0
            ix = [x - 1 for x in b.combo]
            q_ = Q[b.kind][tuple(ix)] if b.kind in ("3連単", "3連複") else Q[b.kind][ix[0], ix[1]]
            assert market.estimated_payout(q_, b.kind) * b.stake / 100 >= bets.TARGET - 1e-6
        if cap:
            assert max(pd.Series([b.kind for b in plan.bets]).value_counts(), default=0) <= cap
        assert 0 <= plan.predicted <= 1


def test_realized_payout_with_dead_heat():
    row = pd.Series({"3連単組番一覧": "010203;010204", "3連単払戻一覧": "50000;80000",
                     "3連複組番一覧": "010203", "3連複払戻一覧": "9000",
                     "馬単組番一覧": "0102", "馬単払戻一覧": "1500", "馬連組番一覧": "0102", "馬連払戻一覧": "800"})
    wins = bets.winning_combos(row)
    plan = bets.Plan("x", [bets.Bet("3連単", (1, 2, 4), 200), bets.Bet("3連複", (1, 2, 3), 100),
                           bets.Bet("馬連", (1, 2), 300), bets.Bet("馬単", (2, 1), 100)])
    assert bets.realized(plan, wins) == 80000 * 2 + 9000 + 800 * 3


# ===========================================================================
# 重賞のガード
# ===========================================================================
def test_exploration_guard_rejects_graded():
    t = pd.DataFrame({"リステッド・重賞競走": ["G1", None], "レースID": ["a", "b"]})
    with pytest.raises(experiment.GradedResultsError):
        experiment.exploration_only(t, "test")
    assert len(experiment.exploration_only(t.iloc[1:], "test")) == 1


# ===========================================================================
# 実行の流れ（合成データ・小さな fold・少ないラウンド）
# ===========================================================================
@pytest.fixture()
def v8_env(tmp_path, monkeypatch):
    # 3 fold にする：2段目（混ぜた確率）が fold 2・3 の2つになり、fold をまたいで表をつなぐ処理まで通る
    # （v8 の最初の版は、ここで pd.concat が attrs の DataFrame を比べて落ちた）
    raw = make_raw_v8(n_days=170, races=8)
    d = tmp_path / "jvlink"
    d.mkdir()
    for rt, df in raw.items():
        df.to_csv(d / f"{rt}.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(predict, "data_root", lambda: tmp_path)
    monkeypatch.setattr(validate, "V5_FOLDS", [validate.Fold("1", (2012, 2012), (2013, 2013)),
                                               validate.Fold("2", (2012, 2013), (2014, 2014)),
                                               validate.Fold("3", (2012, 2014), (2015, 2015))])
    return d


def test_oof_runs_and_marks_graded(v8_env):
    ds = jvmap.build_dataset(v8_env)
    oof = experiment.run_oof(ds, "levels", rounds=10, verbose=False)
    t = oof.table
    assert set(t["fold"]) == {"1", "2", "3"}
    assert t["重賞"].any() and (~t["重賞"]).any()
    assert np.allclose(t.groupby("レースID")["prob"].sum(), 1.0)
    assert set(levels.level_feature_columns()) <= set(oof.feature_cols)


def _no_frame_in_attrs(df: pd.DataFrame) -> bool:
    return not any(isinstance(v, (pd.DataFrame, pd.Series)) for v in df.attrs.values())


def test_cli_v8_commands_run(v8_env, capsys):
    """全コマンドを、表の組み立て・表示・保存まで通す（重い計算のあとで落ちないように）。"""
    from src import v8
    common = ["--data", str(v8_env), "--rounds", "10"]
    assert v8.main(common + ["baseline", "--reps", "3"]) == 0
    out = capsys.readouterr().out
    assert "A 今のモデル" in out and "B 単勝人気" in out and "C ランダム" in out and "重賞" in out
    assert v8.main(common + ["odds-coverage"]) == 0
    assert "3連単の払戻" in capsys.readouterr().out
    assert v8.main(common + ["levels"]) == 0
    out = capsys.readouterr().out
    assert "log-loss" in out and "平場で比べた案: 1 個" in out
    assert v8.main(common + ["market"]) == 0
    out = capsys.readouterr().out
    assert "α（モデル）" in out and "混ぜたもの（2段階）" in out

    assert v8.main(common + ["trifecta"]) == 0
    out = capsys.readouterr().out
    assert "Harville" in out and "推定配当" in out and "3連単：年ごと" in out
    assert "−log P（Henery）" in out
    # 2回目は γ・δ と 2c・2d の結果を保存から読む（推定をやり直さない）
    assert v8.main(common + ["trifecta"]) == 0
    out = capsys.readouterr().out
    assert "保存済みの γ・δ と 2c・2d の結果を使います" in out and "γ・δ を推定中" not in out

    assert v8.main(common + ["bets", "--sample", "30"]) == 0
    out = capsys.readouterr().out
    assert "保存済みの γ・δ と 2c・2d の結果を使います" in out
    for name, _ in v8.PROB_SOURCES:
        assert name in out
    assert "達成率の一覧" in out and "熱さの一覧" in out and "S4 券種と金額を混ぜる" in out
    n_trials = len(v8.PROB_SOURCES) * len(v8.STRATEGIES)
    assert f"平場で比べた案: {n_trials} 個" in out

    assert v8.main(common + ["trackbias"]) == 0
    out = capsys.readouterr().out
    assert "7R 以降" in out and "当日_使えたレース数" in out
    saved = {p.name for p in (v8_env.parent / "v8").glob("*.csv")}
    assert {"baseline.csv", "levels_compare.csv", "market_compare.csv", "trifecta_nll.csv",
            "payout_estimate.csv", "bets_summary.csv", "bets_heat.csv", "trackbias_compare.csv"} <= saved
    summary = pd.read_csv(v8_env.parent / "v8" / "bets_summary.csv", encoding="utf-8-sig")
    assert len(summary) == n_trials and set(summary["確率"]) == {n for n, _ in v8.PROB_SOURCES}


def test_market_tables_keep_no_frames_in_attrs(v8_env):
    """DataFrame を attrs に入れない（pd.concat が attrs を == で比べて落ちる）。"""
    import argparse
    from src import v8
    ds = jvmap.build_dataset(v8_env)
    args = argparse.Namespace(folds=None, refresh=False, rounds=10, stage1="levels", data=str(v8_env))
    mt = v8.market_tables(args, ds)
    for df in (mt.blended, mt.coefs, mt.stage1, mt.odds_oof.table):
        assert _no_frame_in_attrs(df)
    assert mt.blended["fold"].nunique() == 2
    pd.concat([mt.blended.loc[mt.blended["fold"] == f] for f in mt.blended["fold"].unique()])
    res = v8.trifecta_results(args, ds, mt, verbose=False)
    assert set(res["disc_b"]) == set(res["disc_q"]) == set(mt.blended["fold"])


def test_mix_columns():
    from src import v8
    t = pd.DataFrame({"レースID": ["a"] * 3 + ["b"] * 2, "prob": [0.6, 0.3, 0.1, 0.5, 0.5],
                      "市場確率": [0.2, 0.3, 0.5, 0.9, 0.1]})
    m = v8.add_mix_columns(t, weights=(0.0, 0.5, 1.0))
    assert np.allclose(m["mix0"], t["市場確率"]) and np.allclose(m["mix100"], t["prob"])
    assert np.allclose(m.groupby("レースID")["mix50"].sum(), 1.0)
    g = np.sqrt(np.array([0.6, 0.3, 0.1]) * np.array([0.2, 0.3, 0.5]))
    assert np.allclose(m["mix50"].iloc[:3], g / g.sum())


def test_place_loglik_matches_per_race():
    """γ・δ の推定（まとめて計算する版）が、1レースずつの計算と同じになる。"""
    rng = np.random.default_rng(0)
    rows = []
    for r in range(300):
        n = int(rng.integers(5, 16))
        p = rng.dirichlet(np.ones(n))
        order = rng.permutation(n)
        rank = np.full(n, 9.0)
        rank[order[:3]] = [1, 2, 3]
        rows += [{"レースID": f"r{r:04d}", "馬番": h + 1, "着順": rank[h], "prob": p[h]} for h in range(n)]
    t = pd.DataFrame(rows)
    _, L2, L3 = market.place_loglik(t, "prob")
    ts, st = market.race_slices(t)
    s2 = np.zeros(len(market.GRID))
    s3 = np.zeros(len(market.GRID))
    for (a, b), pos in zip(zip(st[:-1], st[1:]), market._finish_index(ts, st)):
        l2, l3 = market._place_loglik(ts["prob"].to_numpy()[a:b], *pos)
        s2 += l2
        s3 += l3
    assert np.allclose(L2.sum(0), s2) and np.allclose(L3.sum(0), s3)
    cal, nll, n = market.trifecta_calibration(t, "prob", 0.9, 0.8)
    ref = np.mean([-np.log(market.trifecta_tensor(ts["prob"].to_numpy()[a:b], 0.9, 0.8)[tuple(pos)])
                   for (a, b), pos in zip(zip(st[:-1], st[1:]), market._finish_index(ts, st))])
    assert n == 300 and np.isclose(nll, ref) and cal["実際の的中"].sum() == 300


def _mixed_reference(P, Q, p, posts, budget=bets.BUDGET, target=bets.TARGET, kinds=bets.KINDS, h=bets.TOP_H):
    """毎回すべての候補の増え方を計算し直す、素直な版（速くした版と同じ選び方になるかの確認用）。"""
    order = np.argsort(-p, kind="mergesort")[:min(h, len(p))]
    hh = len(order)
    outcomes, cands, M = bets.coverage_structure(hh)
    po = bets._subtensor(P, order, outcomes)
    out = []
    for kind, c in cands:
        ix = tuple(order[list(c)])
        q = Q[kind][ix] if kind in ("3連単", "3連複") else Q[kind][ix[0], ix[1]]
        out.append(market.estimated_payout(q, kind) if q > 0 else np.nan)
    est = np.array(out)
    stake = np.ceil(target / est) * 100
    usable = np.isfinite(stake) & (stake <= budget) & np.isin([k for k, _ in cands], list(kinds))
    unc = np.ones(len(outcomes))
    chosen, spent = [], 0
    while True:
        gains = M @ (po * unc)
        ok = usable & (stake <= budget - spent) & (gains > 1e-15)
        if not ok.any():
            break
        best = int(np.argmax(np.where(ok, gains / stake, -1)))
        chosen.append(best)
        spent += int(stake[best])
        unc = unc * (1 - M[best])
        usable[best] = False
    return [(cands[c][0], tuple(int(posts[order[x]]) for x in cands[c][1]), int(stake[c])) for c in chosen]


def test_fast_mixed_matches_reference():
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(6, 18))
        p, q = rng.dirichlet(np.ones(n) * 0.8), rng.dirichlet(np.ones(n) * 0.8)
        posts = np.arange(1, n + 1)
        P = market.trifecta_tensor(p, 0.86, 0.75)
        Q = market.combo_probabilities(market.trifecta_tensor(q, 0.86, 0.75))
        for kinds in (bets.KINDS, ("3連単",)):
            fast = bets.mixed(P, Q, p, posts, kinds=kinds)
            ref = _mixed_reference(P, Q, p, posts, kinds=kinds)
            ref_plan = bets.Plan("ref", [bets.Bet(k, tuple(sorted(c)) if k in ("3連複", "馬連") else c, st)
                                         for k, c, st in ref])
            # 増え方がまったく同じ候補（例: 片方の並びを押さえ済みの馬連と、残りの並びの馬単）が並んだときは、
            # 丸めの差でどちらを選ぶかが変わることがある。どちらでも予測の達成確率・金額は同じ
            assert fast.invest == ref_plan.invest and len(fast.bets) == len(ref_plan.bets)
            assert bets.predicted_success(fast, P, Q, posts, bets.TARGET) == pytest.approx(
                bets.predicted_success(ref_plan, P, Q, posts, bets.TARGET), abs=1e-12)
