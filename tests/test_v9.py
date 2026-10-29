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
