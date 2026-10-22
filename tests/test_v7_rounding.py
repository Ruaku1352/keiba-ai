"""過去集計が「自分の値」に丸めの桁でも依存しないことのテスト（v7 の replay のずれの再発防止）。

v7 までは「自分を含む累積和 − 自分の値」で過去の合計を出していた。浮動小数では
(前の合計 + 自分) − 自分 が前の合計と一致せず、丸めの向きが自分の値（＝そのレースの結果）で変わる。
脚質カテゴリは脚質スコアを 0.10 などの境界で切るので、ちょうど境界にいる馬のカテゴリが
自分の結果で変わり、学習の経路と予測の経路で特徴量がずれた（replay 2026-10-10 で 69 セル）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, features, jvmap, leakfree, predict  # noqa: E402
from test_v6_predict import make_raw  # noqa: E402


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    # 1レース4行（日付→レース順に並んでいる）。同じキー（馬・騎手）は1レースに1行だけ
    df = pd.DataFrame({
        "key": np.concatenate([rng.choice(12, 4, replace=False) for _ in range(n // 4)]),
        "race": np.repeat(np.arange(n // 4), 4),
    })
    df["day"] = df["race"] // 3                             # 1日3レース
    v = pd.Series(rng.integers(0, 11, n) / 10.0, dtype="float32")   # 0.0, 0.1, …, 1.0（11頭立ての相対位置）
    v[rng.random(n) < 0.1] = np.nan
    return df, v


FUNCS = {
    "past_sum": lambda df, v: leakfree.past_sum(df, ["key"], v),
    "past_mean_ignore_nan": lambda df, v: leakfree.past_mean_ignore_nan(df, ["key"], v),
    "past_std_ignore_nan": lambda df, v: leakfree.past_std_ignore_nan(df, ["key"], v),
    "past_window_mean": lambda df, v: leakfree.past_window_mean(df, ["key"], v, 3),
    "past_conditional_mean": lambda df, v: leakfree.past_conditional_mean(
        df, ["key"], v, (df["race"] % 2 == 0).astype("float32")),
    "past_sum_excluding_race": lambda df, v: leakfree.past_sum_excluding_race(df, ["key"], v, "day"),
    "past_mean_ignore_nan_excluding": lambda df, v: leakfree.past_mean_ignore_nan_excluding(
        df, ["key"], v, "day"),
    "past_std_ignore_nan_excluding": lambda df, v: leakfree.past_std_ignore_nan_excluding(
        df, ["key"], v, "day"),
    "past_window_mean_excluding": lambda df, v: leakfree.past_window_mean_excluding(
        df, ["key"], v, 3, "day"),
}


@pytest.mark.parametrize("name", list(FUNCS))
def test_own_rows_do_not_change_past_values_bitwise(name):
    """あるレース（または日）の値を書き換えても、その行の過去集計はビット単位で変わらない。"""
    df, v = _frame()
    fn = FUNCS[name]
    base = fn(df, v).to_numpy()
    group = "day" if "excluding" in name else "race"
    rng = np.random.default_rng(1)
    for g in rng.choice(df[group].unique(), size=15, replace=False):
        rows = (df[group] == g).to_numpy()
        w = v.copy()
        w.iloc[np.flatnonzero(rows)] = (rng.integers(0, 11, rows.sum()) / 10.0).astype("float32")
        got = fn(df, w).to_numpy()
        a, b = base[rows], got[rows]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        assert same.all(), (name, g, a[~same], b[~same])


def test_exclusive_sums_match_brute_force():
    df, v = _frame(n=200, seed=3)
    vals = v.to_numpy(dtype="float64")
    got_sum = leakfree.past_sum(df, ["key"], v).to_numpy()
    got_ex = leakfree.past_sum_excluding_race(df, ["key"], v, "day").to_numpy()
    for i in range(len(df)):
        prev = (df["key"].to_numpy()[:i] == df["key"].iloc[i])
        assert got_sum[i] == pytest.approx(np.nansum(vals[:i][prev]), abs=1e-5)
        prev_day = prev & (df["day"].to_numpy()[:i] != df["day"].iloc[i])
        assert got_ex[i] == pytest.approx(np.nansum(vals[:i][prev_day]), abs=1e-5)


def test_style_category_on_boundary_does_not_depend_on_own_result():
    """過去平均がちょうど 0.10 の馬は、今回の4角位置が何であっても同じカテゴリ。"""
    for p2 in (0.0, 0.2, 0.3, 0.7, 0.9, 1.0):
        df = pd.DataFrame({"h": ["a", "a"]})
        m = leakfree.past_mean_ignore_nan(df, ["h"], pd.Series([0.1, p2], dtype="float32")).iloc[1]
        assert m == np.float32(0.1)


@pytest.fixture(scope="module")
def ds11():
    # 11頭立て：4角の相対位置が 0.1 刻みになり、脚質カテゴリの境界（0.10・0.70）にちょうど乗る馬が出る
    return jvmap.build_dataset(raw=make_raw(n_days=60, n_horses=11, seed=5))


def test_replay_paths_match_with_eleven_runners(ds11):
    """実データの replay で出たずれ（脚質カテゴリ→前傾度→想定ペース）が出ないこと。全日で確認。"""
    rr = ds11.race_result
    dates = pd.to_datetime(rr["レース日付"])
    graded = rr["リステッド・重賞競走"].isin(config.GRADED_VALUES)
    total = 0
    for day in sorted(dates.loc[graded].unique())[20:]:
        day = pd.Timestamp(day)
        train = predict.make_training_frame(rr, ds11.lap_df, until=f"{day:%Y-%m-%d}")
        rows = train.loc[(train["レース日付"] == day)
                         & train["リステッド・重賞競走"].isin(config.GRADED_VALUES)]
        pred = predict.build_prediction_features(rr.loc[dates < day], rr.loc[(dates == day) & graded],
                                                 ds11.lap_df, keep_actual_going=True)
        diff = predict.compare_paths(rows, pred, features.feature_columns(train))
        total += int(diff["不一致"].sum())
        assert diff["不一致"].sum() == 0, (day, diff.loc[diff["不一致"] > 0].to_string())
    assert total == 0


def test_categorical_style_compared_exactly(ds11):
    """compare_paths は脚質カテゴリ（カテゴリ列）を許容誤差なしで比べている。"""
    rr = ds11.race_result
    day = pd.Timestamp(sorted(pd.to_datetime(rr["レース日付"]).unique())[40])
    train = predict.make_training_frame(rr, ds11.lap_df, until=f"{day:%Y-%m-%d}")
    rows = train.loc[train["レース日付"] == day].copy()
    flipped = rows.copy()
    cat = flipped["脚質カテゴリ"].astype(object)
    flipped["脚質カテゴリ"] = cat.where(cat != "先行", "逃げ").where(cat == "先行", "先行")
    diff = predict.compare_paths(rows, flipped, ["脚質カテゴリ"], exclude=())
    assert int(diff["不一致"].iloc[0]) > 0
