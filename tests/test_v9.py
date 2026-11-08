"""v9 重賞の確認のテスト（F・F2・B）のテスト。合成データで、判定のルール・同じレースで比べること・やり直しの扱いを確かめる。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import bets, v9  # noqa: E402
from test_v8 import v8_env  # noqa: E402,F401  （同じ合成データの環境を使う）


def _rows(hot_f, hot_f2, hot_b):
    rows = []
    for i, (a, b, c) in enumerate(zip(hot_f, hot_f2, hot_b)):
        for key, h in (("F", a), ("F2", b), ("B", c)):
            rows.append({"レースID": f"R{i:04d}", "案": v9.NAMES[key], "払戻": 60_000 if h else 0, "投資": 12_000,
                         "点数": 10, "予測の達成確率": 0.1})
    return pd.DataFrame(rows)


def test_judge_rule():
    n = 1000
    none = np.zeros(n, bool)
    many = np.zeros(n, bool)
    many[:150] = True
    few = np.zeros(n, bool)
    few[:60] = True
    assert v9.judge(_rows(many, few, none))[0] == "F"             # F だけ合格
    assert v9.judge(_rows(few, many, none))[0] == "F2"            # F2 だけ合格
    more = many.copy()
    more[150:170] = True
    assert v9.judge(_rows(many, more, none))[0] == "F2"           # 両方合格なら達成率の高いほう
    assert v9.judge(_rows(few, few, few))[0] == "B"               # どちらも合格しない
    assert v9.CONFIRM_Z == pytest.approx(2.2414, abs=1e-3)       # 97.5% 区間


def test_add_f2_is_normalized_geometric_mean():
    t = pd.DataFrame({"レースID": ["A"] * 3 + ["B"] * 2, "prob": [0.5, 0.3, 0.2, 0.6, 0.4],
                      "市場確率": [0.2, 0.3, 0.5, 0.5, 0.5]})
    out = v9.add_f2(t)
    assert np.allclose(out.groupby("レースID")["F2確率"].sum(), 1.0)
    g = np.sqrt(np.array([0.5 * 0.2, 0.3 * 0.3, 0.2 * 0.5]))
    assert np.allclose(out["F2確率"].iloc[:3], g / g.sum())


def test_box_is_120_points():
    plan = bets.box(np.arange(10, 0, -1, dtype=float), np.arange(1, 11), size=v9.BOX_SIZE)
    assert len(plan.bets) == 120 and plan.invest == 12_000


def test_cli_confirm_and_rerun(v8_env, capsys):
    from src import v8
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "tilt", "--sample", "5"]) == 0
    capsys.readouterr()
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "confirm"]) == 0
    out = capsys.readouterr().out
    assert out.index("■ 判定") < out.index("判定に使った比較") and "97.5% 区間" in out
    assert "F − F2" in out and "グレード別" in out and "払戻の分布" in out
    d = v8_env.parent / "v9"
    runs = pd.read_csv(d / "confirm_runs.csv", encoding="utf-8-sig")
    assert len(runs) == 1
    first = Path(runs["保存先"].iloc[0])
    races = pd.read_csv(first / "races.csv", encoding="utf-8-sig")
    assert set(races["案"]) == set(v9.NAMES.values())               # F・F2・B だけ
    assert races.groupby("案")["レースID"].nunique().nunique() == 1  # 同じレースで比べている
    assert races["グレード"].isin(["G1", "G2", "G3", "G"]).all()
    assert (races["fold"].astype(str) != "1").all()                 # fold 2 以降
    f = races.loc[races["案"] != v9.NAMES["B"]]
    assert (f["点数"] <= v9.MAX_POINTS).all() and (f["投資"] <= bets.BUDGET).all()
    b = races.loc[races["案"] == v9.NAMES["B"]]
    assert (b["投資"] <= 12_000).all()
    # 2回目は理由が無いと実行しない
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "confirm"]) == 1
    assert "すでに実行されています" in capsys.readouterr().out
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "confirm", "--rerun-reason", "テスト"]) == 0
    out = capsys.readouterr().out
    assert "前回までの判定" in out
    runs = pd.read_csv(d / "confirm_runs.csv", encoding="utf-8-sig")
    assert len(runs) == 2 and first.exists()                          # 前の結果は消さない
    second = Path(runs["保存先"].iloc[1])
    a = pd.read_csv(first / "judgement.csv", encoding="utf-8-sig")
    b2 = pd.read_csv(second / "judgement.csv", encoding="utf-8-sig")
    assert np.allclose(a["差"], b2["差"])                             # 同じデータなら同じ結果


def test_cli_curse(v8_env, capsys):
    """v11 2：点数ごと・確定÷推定ごと・集中度ごとの表（記述だけ）。"""
    from src import v8
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "tilt", "--sample", "5"]) == 0
    capsys.readouterr()
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "curse"]) == 0
    out = capsys.readouterr().out
    assert "記述だけ" in out and "最終の点数ごと" in out and "集中度" in out
    d = v8_env.parent / "v9" / "curse"
    races = pd.read_csv(d / "races.csv", encoding="utf-8-sig")
    assert set(races["区分"]) == {"重賞", "平場"}
    assert races["点数"].between(1, 15).all() and races["集中度"].between(0, 1.0001).all()
    t1 = pd.read_csv(d / "by_points.csv", encoding="utf-8-sig")
    assert t1.groupby("区分")["レース数"].sum().to_dict() == races.groupby("区分").size().to_dict()
    bets_df = pd.read_csv(d / "bets.csv", encoding="utf-8-sig")
    assert (bets_df["当たる確率（予測）"] > 0).all()


def test_curse_matches_confirm(v8_env, capsys):
    from src import v8
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "tilt", "--sample", "5"]) == 0
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "confirm"]) == 0
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "curse"]) == 0
    capsys.readouterr()
    d = v8_env.parent / "v9"
    runs = pd.read_csv(d / "confirm_runs.csv", encoding="utf-8-sig")
    conf = pd.read_csv(Path(runs["保存先"].iloc[-1]) / "races.csv", encoding="utf-8-sig", dtype={"レースID": str})
    f = conf.loc[conf["案"] == v9.NAMES["F"]].set_index("レースID")
    c = pd.read_csv(d / "curse" / "races.csv", encoding="utf-8-sig", dtype={"レースID": str})
    c = c.loc[c["区分"] == "重賞"].set_index("レースID")
    assert set(c.index) == set(f.index)
    assert (c["点数"] == f.loc[c.index, "点数"]).all()
    assert (c["熱い"] == (f.loc[c.index, "払戻"] >= v9.TARGET)).all()


# ---------------------------------------------------------------------------
# v12 付け替えの効果の試算
# ---------------------------------------------------------------------------
def _random_races(n=300, seed=0):
    rng = np.random.default_rng(seed)
    races = []
    for i in range(n):
        k = int(rng.integers(1, 8))
        stakes = rng.multinomial(120 - k, np.ones(k) / k) + 1                  # 合計 12,000円・各 100円以上
        rows = []
        for st in stakes:
            odds = float(np.round(np.exp(rng.uniform(np.log(3), np.log(300))), 1))
            win = rng.random() < 0.15
            rows.append({"券種": str(rng.choice(["馬連", "馬単", "3連複", "3連単"])), "金額": int(st) * 100,
                         "確定オッズ": odds if rng.random() < 0.95 else None, "実際の払戻": odds * 100 if win else 0.0})
        races.append({"区分": "平場", "レースID": f"R{i}", "買い目": rows,
                      "プラン": {1.4: {"買い目": rows, "予測": 0.12, "点数": len(rows), "投資": 12_000}}})
    return races


def test_rebalance_without_drift_is_never_worse():
    """ずれなし（確定オッズ＝20分前）なら、付け替えで熱い当たりは減らない（増えることはある）。"""
    races = _random_races()
    for m in v9.REBALANCE_M:
        for r in races:
            base = v9.evaluate_f(r)
            sim = v9.simulate_rebalance(r, m, None)
            assert not (base["熱い"] and not sim["熱い"])
            assert sum(b["金額"] for b in r["買い目"]) == 12_000                      # 元の買い目は書き換えない
    table, _ = v9.rebalance_summary(races, "none", 1, 0, None)
    assert (table["差（同じ k の付け替えなしとの）"] >= 0).all()


def test_drift_sampler_data_and_fallback():
    samples = pd.DataFrame({"券種": "馬連", "帯": "10〜50倍", "r": np.full(50, 0.8)})
    s = v9.DriftSampler("data", samples, seed=1)
    assert s.draw("馬連", 20.0) == pytest.approx(0.8)                              # データのある組
    r = [s.draw("3連単", 500.0) for _ in range(2000)]                              # データの無い組は仮の分布
    assert abs(np.mean(np.log(r))) < 0.01 and np.std(np.log(r)) == pytest.approx(0.10, abs=0.01)
    assert s.used == {"集めたデータ": 1, "仮の分布": 2000}
    a = [v9.DriftSampler("assumed", None, seed=3).draw("馬連", 5.0) for _ in range(3)]
    assert len(set(a)) == 1                                                        # 種を固定すれば同じ


def test_cli_rebalance(v8_env, capsys):
    from src import v8
    from test_v8 import _write_synthetic_odds
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "tilt", "--sample", "5"]) == 0
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "odds-targets", "--n-flat", "60"]) == 0
    targets = pd.read_csv(v8_env.parent / "v8" / "odds_targets.csv", dtype={"レースID": str}, encoding="utf-8-sig")
    _write_synthetic_odds(v8_env, targets)
    capsys.readouterr()
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "rebalance", "--draws", "2"]) == 0
    out = capsys.readouterr().out
    assert "仮の分布（log r ~ 正規(0, 0.10)）" in out and "■ 決め方" in out and "記述だけ" in out
    t = pd.read_csv(v8_env.parent / "v9" / "rebalance" / "rebalance_assumed.csv", encoding="utf-8-sig")
    assert set(t["区分"]) == {"重賞", "平場"}
    for kind, g in t.groupby("区分"):
        assert len(g) == 1 + 2 * len(v9.REBALANCE_M)                              # F・m ごとに ずれなし／あり
        none = g.loc[g["ずれ"].fillna("").str.startswith("なし")]
        assert (none["差（同じ k の付け替えなしとの）"] >= -1e-12).all()             # ずれなしは悪くならない
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "rebalance", "--drift", "data", "--draws", "1"]) == 0
    assert "仮の分布で引きます" in capsys.readouterr().out                           # 集めたデータがまだ無い



# ---------------------------------------------------------------------------
# v13：最初の金額の余裕 k × 付け替えの目標 m
# ---------------------------------------------------------------------------
def _races_k(n=200, seed=0, ks=(1.0, 1.4)):
    base = _random_races(n, seed)
    rng = np.random.default_rng(seed + 1)
    for r in base:
        for k in ks:
            rows = [dict(b, 組=(i,)) for i, b in enumerate(r["買い目"])]
            if k != 1.4:                                                           # 余裕が小さいほど1点の金額は小さい（同じ組で）
                rows = [dict(b, 金額=max(100, int(b["金額"] * k / 1.4) // 100 * 100)) for b in rows]
            r["プラン"][float(k)] = {"買い目": rows, "予測": 0.12 * 1.4 / k, "点数": len(rows),
                                    "投資": sum(b["金額"] for b in rows)}
        r["買い目"] = r["プラン"][1.4]["買い目"]
    return base


def test_common_random_numbers_same_bet_same_r():
    s = v9.DriftSampler("assumed", None, seed=7)
    a = s.draw("馬連", 12.0, key=("R1", 0, "馬連", (1, 2)))
    assert s.draw("馬連", 12.0, key=("R1", 0, "馬連", (1, 2))) == a                  # 同じ買い目・同じ回は同じ r
    assert v9.DriftSampler("assumed", None, seed=7).draw("馬連", 12.0, key=("R1", 0, "馬連", (1, 2))) == a
    assert s.draw("馬連", 12.0, key=("R1", 1, "馬連", (1, 2))) != a                  # 回が違えば別


def test_kxm_summary_shapes_and_baseline():
    races = _races_k()
    table, pivots, used = v9.kxm_summary(races, "assumed", 3, 0, None, ks=(1.0, 1.4))
    g = table.loc[table["区分"] == "平場"]
    assert len(g) == 2 * (1 + 2 * len(v9.REBALANCE_M))
    base = g.loc[(g["k"] == 1.4) & (g["付け替え"] == "m = 1.00") & g["ずれ"].str.startswith("あり")].iloc[0]
    assert base["差（今の形との）"] == pytest.approx(0.0) and base["今の形だけ熱い"] == 0     # 今の形どうしは差 0
    piv = pivots["平場"]
    assert list(piv.index) == ["k = 1.0", "k = 1.4"] and list(piv.columns) == ["付け替えなし"] + [f"m = {m:.2f}" for m in v9.REBALANCE_M]
    assert piv.loc["k = 1.4", "m = 1.00"] == pytest.approx(base["熱い当たりの割合"])
    for col in ("15点の上限に当たった割合", "予算を使い切れなかった割合", "補いきれなかった組があった割合"):
        assert g[col].between(0, 1).all()
    assert used["仮の分布"] > 0


def test_decide_k_m_rules():
    def table(best_k, best_m, m14):
        rows = []
        for k in (1.0, 1.2, 1.4):
            for m in (1.0, 1.1):
                rate = 0.10
                if (k, m) == (best_k, best_m):
                    rate = 0.13
                elif k == 1.4 and m == m14:
                    rate = 0.12
                rows.append({"区分": "平場", "k": k, "付け替え": f"m = {m:.2f}", "ずれ": "あり（集めたデータ）", "熱い当たりの割合": rate})
            rows.append({"区分": "平場", "k": k, "付け替え": "付け替えなし", "ずれ": "", "熱い当たりの割合": 0.2})
        return pd.DataFrame(rows)
    c = v9.decide_k_m(table(1.2, 1.1, 1.0))
    assert (c["k*"], c["m*"], c["m1.4"], c["重賞で確かめる"]) == (1.2, 1.1, 1.0, True)    # 付け替えなしの行は選ばない
    c = v9.decide_k_m(table(1.4, 1.1, 1.1))
    assert (c["k*"], c["m*"], c["重賞で確かめる"]) == (1.4, 1.1, False)


def test_cli_rebalance_k_and_decide(v8_env, capsys):
    from src import v8, raceday
    from test_v8 import _write_synthetic_odds
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "tilt", "--sample", "5"]) == 0
    assert v8.main(["--data", str(v8_env), "--rounds", "10", "odds-targets", "--n-flat", "60"]) == 0
    targets = pd.read_csv(v8_env.parent / "v8" / "odds_targets.csv", dtype={"レースID": str}, encoding="utf-8-sig")
    _write_synthetic_odds(v8_env, targets)
    capsys.readouterr()
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "rebalance", "--k", "1.0,1.2", "--draws", "2"]) == 0
    out = capsys.readouterr().out
    assert "k を行・m を列" in out and "何も決めない" in out
    t = pd.read_csv(v8_env.parent / "v9" / "rebalance" / "rebalance_assumed.csv", encoding="utf-8-sig")
    assert set(t["k"]) == {1.0, 1.2, 1.4}                                           # 1.4 は必ず入る
    plan_pts = t.groupby("k")["点数（平均）"].first()
    assert (t["点数（平均）"] <= 15).all() and plan_pts.notna().all()
    # 決定：集めたデータが無ければ実行しない
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "rebalance-decide", "--draws", "1"]) == 1
    samples = pd.DataFrame({"日付": "20261101", "レースID": "x", "券種": "馬連", "帯": "10〜50倍", "r": np.full(40, 0.95)})
    samples.to_csv(raceday.drift_samples_path(), index=False, encoding="utf-8-sig", compression="gzip")
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "rebalance-decide", "--draws", "1"]) == 0
    out = capsys.readouterr().out
    assert "■ 決定：k = " in out and "freeze --k" in out
    assert v9.main(["--data", str(v8_env), "--rounds", "10", "rebalance-decide", "--draws", "1"]) == 1   # 1回だけ
    assert "すでに行われています" in capsys.readouterr().out
