"""v6（実戦用の予測）のテスト。

いちばん大事なのは次の3つ:
  1. 学習の経路と予測の経路で、同じレースの特徴量が一致する（馬体重・増減を除く）
  2. 予測の出力が、予測対象レースの結果の列に一切依存しない
  3. 同じ日（同じ週末）の予測対象レースどうしが混ざらない
     （同じ騎手が2つの対象レースに乗っても、もう片方を過去の騎乗として数えない）

合成データは「1日に3レース、うち2つが重賞の日がある」「騎手・調教師は同じ日の
レースで重なる」「同じ馬は1日に1回しか走らない」ように作る。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, features, jvmap, pace, predict  # noqa: E402

pace.PACE_Z_MIN_COUNT = 5
pace.EXPECTED_PACE_MIN_COUNT = 5

RACES_PER_DAY = 3


# ===========================================================================
# 合成の生データ（jvlink が保存する CSV と同じ列名・すべて文字列）
# ===========================================================================
def _key(day: pd.Timestamp, jyo: str, kai: int, nichi: int, race_no: int, make: pd.Timestamp):
    return {"id.Year": f"{day.year:04d}", "id.MonthDay": day.strftime("%m%d"), "id.JyoCD": jyo,
            "id.Kaiji": f"{kai:02d}", "id.Nichiji": f"{nichi:02d}", "id.RaceNum": f"{race_no:02d}",
            "head.MakeDate.Year": f"{make.year:04d}", "head.MakeDate.Month": make.strftime("%m"),
            "head.MakeDate.Day": make.strftime("%d")}


def _time_str(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m}{round((seconds - 60 * m) * 10):03d}"


def make_raw(n_days: int = 60, n_horses: int = 8, seed: int = 0, kubun: str = "7",
             start: str = "2015-01-04", entry_days: tuple = ()) -> dict[str, pd.DataFrame]:
    """1日 RACES_PER_DAY レース。偶数日は2レースが重賞（G1 と G3）、奇数日は1レース（G2）。

    entry_days に入れた日（0始まりの番号）は出馬表の段階（データ区分 2、結果なし）にする。
    """
    rng = np.random.default_rng(seed)
    horses = [f"2012{i:06d}" for i in range(n_horses * RACES_PER_DAY * 2)]
    jockeys = [f"0{i:04d}" for i in range(n_horses + 4)]      # 同じ日のレースで重なる
    trainers = [f"1{i:04d}" for i in range(5)]
    ra, se, hr = [], [], []
    seq = 0
    for d in range(n_days):
        day = pd.Timestamp(start) + pd.Timedelta(days=7 * d)
        entry = d in entry_days
        k = "2" if entry else kubun
        made = day - pd.Timedelta(days=1) if entry else day
        runners_today = rng.choice(horses, size=n_horses * RACES_PER_DAY, replace=False)
        for r in range(RACES_PER_DAY):
            jyo = ["05", "08", "09"][r]
            key = _key(day, jyo, 1 + d // 8, 1 + d % 8, 11, made)
            if d % 2 == 0:
                grade = ["A", "C", " "][r]
            else:
                grade = ["B", " ", " "][r]
            track = ["11", "17", "23"][(d + r) % 3]
            ra.append({"_seq": str(seq), "head.RecordSpec": "RA", "head.DataKubun": k, **key,
                       "RaceInfo.Hondai": f"テスト{d}-{r}", "RaceInfo.Ryakusyo10": f"テ{d}-{r}",
                       "GradeCD": grade, "Kyori": str([1200, 1600, 2000][(d + r) % 3]),
                       "TrackCD": track, "TenkoBaba.TenkoCD": "0" if entry else "1",
                       "TenkoBaba.SibaBabaCD": "0" if entry else str(1 + d % 3),
                       "TenkoBaba.DirtBabaCD": "0" if entry else str(1 + d % 2),
                       "SyussoTosu": f"{n_horses:02d}", "HassoTime": "1530",
                       "HaronTimeS3": "000" if entry else str(345 + rng.integers(-15, 15)),
                       "HaronTimeL3": "000" if entry else str(350 + rng.integers(-15, 15))})
            seq += 1
            runners = runners_today[r * n_horses:(r + 1) * n_horses]
            race_jockeys = rng.choice(jockeys, size=n_horses, replace=False)
            order = rng.permutation(n_horses) + 1
            corner = rng.permutation(n_horses) + 1
            for i, (h, rank) in enumerate(zip(runners, order)):
                res = (lambda v, blank: blank if entry else v)
                se.append({"_seq": str(seq), "head.RecordSpec": "SE", "head.DataKubun": k, **key,
                           "Wakuban": str(1 + i // 2), "Umaban": f"{i + 1:02d}", "KettoNum": h,
                           "Bamei": f"ウマ{h[-4:]}", "SexCD": "1", "Barei": "04",
                           "ChokyosiCode": rng.choice(trainers), "ChokyosiRyakusyo": "調教師",
                           "KisyuCode": race_jockeys[i], "KisyuRyakusyo": f"騎手{race_jockeys[i][-2:]}",
                           "Futan": str(540 + 10 * rng.integers(0, 3)),
                           "BaTaijyu": res(str(470 + rng.integers(-20, 20)), "000"),
                           "ZogenFugo": res(rng.choice(["+", "-"]), " "),
                           "ZogenSa": res(f"{rng.integers(0, 12):03d}", "000"), "IJyoCD": "0",
                           "KakuteiJyuni": res(f"{rank:02d}", "00"),
                           "Time": res(_time_str(94.0 + rank * 0.2), "0000"),
                           "Jyuni1c": "00", "Jyuni2c": "00",
                           "Jyuni3c": res(f"{corner[i]:02d}", "00"),
                           "Jyuni4c": res(f"{corner[i]:02d}", "00"),
                           "Odds": res(f"{int(rank * 25 + rng.integers(0, 30)):04d}", "0000"),
                           "Ninki": res(f"{rank:02d}", "00"), "HaronTimeL4": "000",
                           "HaronTimeL3": res(str(340 + rank), "000"), "TimeDiff": "+005",
                           "DMTime": "13450", "DMJyuni": "01",
                           "KyakusituKubun": res(str(1 + (corner[i] - 1) * 4 // n_horses), "0")})
                seq += 1
            if entry:
                continue
            win = [i + 1 for i in np.argsort(order)[:3]]
            hr_row = {"_seq": str(seq), "head.RecordSpec": "HR", "head.DataKubun": "2", **key}
            for j in range(1, 10):
                hr_row[f"FuseirituFlag[{j}]"] = "0"
                hr_row[f"TokubaraiFlag[{j}]"] = "0"
                hr_row[f"HenkanFlag[{j}]"] = "0"
            for prefix, n in (("PaySanrentan", 6), ("PaySanrenpuku", 3)):
                for j in range(1, n + 1):
                    hr_row[f"{prefix}[{j}].Kumi"] = "000000"
                    hr_row[f"{prefix}[{j}].Pay"] = "000000000"
            hr_row["PaySanrentan[1].Kumi"] = "".join(f"{w:02d}" for w in win)
            hr_row["PaySanrentan[1].Pay"] = f"{int(np.exp(rng.normal(9.3, 1.4))):09d}"
            hr_row["PaySanrenpuku[1].Kumi"] = "".join(f"{w:02d}" for w in sorted(win))
            hr_row["PaySanrenpuku[1].Pay"] = f"{int(np.exp(rng.normal(7.2, 1.1))):09d}"
            hr.append(hr_row)
            seq += 1
    return {"RA": pd.DataFrame(ra), "SE": pd.DataFrame(se), "HR": pd.DataFrame(hr)}


@pytest.fixture(scope="module")
def ds():
    return jvmap.build_dataset(raw=make_raw())


def _graded(rr: pd.DataFrame) -> pd.Series:
    return rr["リステッド・重賞競走"].isin(config.GRADED_VALUES)


def _replay_frames(ds, day: pd.Timestamp, **kw):
    """練習モードと同じ組み立て：前日までを履歴に、その日の重賞を予測対象にする。"""
    rr = ds.race_result
    dates = pd.to_datetime(rr["レース日付"])
    history = rr.loc[dates < day]
    targets = rr.loc[(dates == day) & _graded(rr)]
    kw.setdefault("keep_actual_going", True)
    pred = predict.build_prediction_features(history, targets, ds.lap_df, **kw)
    return history, targets, pred


def _feature_cols(frame):
    return features.feature_columns(frame)


# ===========================================================================
# 1. 学習の経路と予測の経路の一致
# ===========================================================================
@pytest.mark.parametrize("day_index", [40, 51, 59])
def test_train_and_predict_paths_match_except_weight(ds, day_index):
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * day_index)
    train = predict.make_training_frame(ds.race_result, ds.lap_df, until=f"{day:%Y-%m-%d}")
    train_rows = train.loc[(train["レース日付"] == day) & _graded(train)]
    _, targets, pred = _replay_frames(ds, day)
    assert len(pred) == len(targets) == len(train_rows) > 0

    cols = _feature_cols(train)
    diff = predict.compare_paths(train_rows, pred, cols)
    assert diff.attrs["missing_rows"] == 0
    bad = diff.loc[diff["不一致"] > 0]
    assert bad.empty, bad.to_string()
    # 比べた列に、騎手・調教師・ペース・脚質の過去集計が入っていること
    for col in ["騎手通算勝率", "騎手直近100走勝率", "調教師通算勝率", "通算勝率"]:
        assert col in set(diff["列"])

    # 馬体重だけは意図的に違う（前走の値・増減0）
    assert (pred["場体重増減"] == 0).all()


def test_weight_is_previous_known_weight(ds):
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 50)
    history, targets, pred = _replay_frames(ds, day)
    h = history.sort_values(["レース日付", "レースID"])
    h = h.loc[h["馬体重"].notna()]
    last = h.groupby("血統登録番号")["馬体重"].last()
    expect = pred["血統登録番号"].map(last)
    got = pred["馬体重"].astype("float64")
    assert np.allclose(got.fillna(-1), expect.fillna(-1))
    assert set(pred["馬体重の扱い"]) <= {"前走の値で代用", "前走なし（欠損）"}


def test_previous_weight_skips_missing_and_first_race_is_nan():
    df = pd.DataFrame({
        "レースID": ["1", "2", "3", "4"], "レース日付": pd.to_datetime(["2020-01-01"] * 4),
        "血統登録番号": ["a", "a", "a", "a"], "馬番": [1, 1, 1, 1], "着順": [1, 2, 3, 4],
        "馬体重": [480, np.nan, 0, 490],
    })
    prev = predict.previous_weight(df)
    assert np.isnan(prev.iloc[0])
    assert prev.iloc[1] == 480 and prev.iloc[2] == 480 and prev.iloc[3] == 480


def test_actual_weight_option_matches_training_path_completely(ds):
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 44)
    train = predict.make_training_frame(ds.race_result, ds.lap_df, until=f"{day:%Y-%m-%d}")
    train_rows = train.loc[(train["レース日付"] == day) & _graded(train)]
    _, _, pred = _replay_frames(ds, day, substitute_weights=False)
    diff = predict.compare_paths(train_rows, pred, _feature_cols(train), exclude=())
    assert (diff["不一致"] == 0).all(), diff.loc[diff["不一致"] > 0].to_string()


# ===========================================================================
# 2. 結果の列に依存しない
# ===========================================================================
def _scramble_results(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = targets.copy()
    n = len(t)
    t["着順"] = rng.integers(1, 19, n)
    t["タイム"] = rng.uniform(50, 200, n)
    t["後３Ｆタイム"] = rng.uniform(30, 45, n)
    t["単勝オッズ"] = rng.uniform(1, 500, n)
    t["人気"] = rng.integers(1, 19, n)
    for c in ["_1コーナー順位", "_2コーナー順位", "_3コーナー順位", "_4コーナー順位"]:
        t[c] = rng.integers(1, 19, n)
    t["_脚質判定"] = rng.integers(1, 5, n)
    t["_前3F"] = rng.uniform(30, 40, n)
    t["_後3F"] = rng.uniform(30, 40, n)
    t["馬体重"] = rng.integers(400, 560, n)       # 代用されるので出力に効かない
    t["場体重増減"] = rng.integers(-20, 20, n)
    return t


def test_prediction_does_not_depend_on_target_results(ds):
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 56)
    history, targets, base = _replay_frames(ds, day)
    for seed in range(3):
        other = predict.build_prediction_features(
            history, _scramble_results(targets, seed), ds.lap_df, keep_actual_going=True)
        cols = _feature_cols(base)
        diff = predict.compare_paths(base, other, cols, exclude=())
        assert (diff["不一致"] == 0).all(), diff.loc[diff["不一致"] > 0].to_string()


def test_target_rows_in_history_and_laps_are_ignored(ds):
    """履歴に予測対象レースの結果が紛れ込んでいても使わない（race_result 全体を渡してもよい）。"""
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 56)
    _, targets, base = _replay_frames(ds, day)
    rr = ds.race_result
    history_with_targets = rr.loc[pd.to_datetime(rr["レース日付"]) <= day]
    other = predict.build_prediction_features(history_with_targets, targets, ds.lap_df,
                                              keep_actual_going=True)
    diff = predict.compare_paths(base, other, _feature_cols(base), exclude=())
    assert (diff["不一致"] == 0).all()


def test_scores_do_not_depend_on_target_results(ds):
    model = _tiny_model(ds)
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 56)
    history, targets, base = _replay_frames(ds, day)
    other = predict.build_prediction_features(history, _scramble_results(targets, 9), ds.lap_df,
                                              keep_actual_going=True)
    a = predict.score(model, base).set_index("血統登録番号")["予測確率"]
    b = predict.score(model, other).set_index("血統登録番号")["予測確率"]
    assert np.allclose(a.sort_index(), b.sort_index())


# ===========================================================================
# 3. 同じ日・同じ週末の予測対象どうしが混ざらない
# ===========================================================================
def test_same_day_targets_are_isolated(ds):
    """同じ日の重賞2つ。片方だけで予測しても、両方まとめて予測しても、同じ特徴量になる。"""
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 58)
    history, targets, both = _replay_frames(ds, day)
    race_ids = sorted(targets["レースID"].unique())
    assert len(race_ids) == 2
    shared = set(targets.loc[targets["レースID"] == race_ids[0], "騎手コード"]) \
        & set(targets.loc[targets["レースID"] == race_ids[1], "騎手コード"])
    assert shared, "テストデータで騎手が重なっていない"

    cols = _feature_cols(both)
    for rid in race_ids:
        alone = predict.build_prediction_features(
            history, targets.loc[targets["レースID"] == rid], ds.lap_df, keep_actual_going=True)
        diff = predict.compare_paths(both.loc[both["レースID"] == rid], alone, cols, exclude=())
        assert (diff["不一致"] == 0).all(), diff.loc[diff["不一致"] > 0].to_string()


def test_jockey_count_does_not_include_other_target_race(ds):
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 58)
    history, targets, pred = _replay_frames(ds, day)
    h = predict.preprocess.basic_clean(history)
    rides = h.groupby("騎手コード").size()
    expect = pred["騎手コード"].astype(str).map(rides).fillna(0)
    assert (pred["騎手通算騎乗数"].astype(int).to_numpy() == expect.astype(int).to_numpy()).all()


def test_weekend_targets_do_not_see_each_other(ds):
    """日曜と月曜の重賞をまとめて予測するとき、月曜のレースは日曜の（未確定の）レースを見ない。"""
    rr = ds.race_result
    dates = pd.to_datetime(rr["レース日付"])
    sun = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 57)
    mon = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 58)
    history = rr.loc[dates < sun]
    targets = rr.loc[dates.isin([sun, mon]) & _graded(rr)]
    both = predict.build_prediction_features(history, targets, ds.lap_df, keep_actual_going=True)
    mon_alone = predict.build_prediction_features(
        history, targets.loc[pd.to_datetime(targets["レース日付"]) == mon], ds.lap_df,
        keep_actual_going=True)
    diff = predict.compare_paths(both.loc[both["レース日付"] == mon], mon_alone,
                                 _feature_cols(both), exclude=())
    assert (diff["不一致"] == 0).all(), diff.loc[diff["不一致"] > 0].to_string()


# ===========================================================================
# 出馬表（データ区分 1・2）の取り込み
# ===========================================================================
def test_prepare_entries_reads_unconfirmed_graded_races():
    raw = make_raw(n_days=12, entry_days=(10, 11))
    entries = jvmap.prepare_entries(raw)
    days = sorted(entries["レース日付"].unique())
    assert len(days) == 2
    assert set(entries["リステッド・重賞競走"]) <= set(config.GRADED_VALUES)
    assert (entries["データ区分"] == "出馬表").all()
    assert entries["枠順確定"].all()
    assert entries["着順"].isna().all() and entries["馬体重"].isna().all()
    # 確定レース側には入らない
    ds = jvmap.build_dataset(raw=raw)
    assert not set(entries["レースID"]) & set(ds.race_result["レースID"])


def test_prepare_entries_excludes_scratched_and_flags_unfixed_posts():
    raw = make_raw(n_days=12, entry_days=(10,))
    se = raw["SE"]
    day_mask = se["id.MonthDay"] == (pd.Timestamp("2015-01-04") + pd.Timedelta(days=70)).strftime("%m%d")
    first = se.index[day_mask & (se["id.JyoCD"] == "05")]
    se.loc[first[0], "IJyoCD"] = "1"                      # 出走取消
    second = se.index[day_mask & (se["id.JyoCD"] == "08")]
    se.loc[second, "Umaban"] = "00"                       # 出走馬名表の段階（馬番未定）
    se.loc[second, "head.DataKubun"] = "1"
    ra = raw["RA"]
    ra.loc[(ra["id.MonthDay"] == se.loc[second[0], "id.MonthDay"]) & (ra["id.JyoCD"] == "08"),
           "head.DataKubun"] = "1"
    entries = jvmap.prepare_entries(raw)
    assert se.loc[first[0], "KettoNum"] not in set(entries["血統登録番号"])
    by_race = entries.groupby("競馬場コード")
    assert by_race.size()["05"] == 7
    assert not entries.loc[entries["競馬場コード"] == "08", "枠順確定"].any()
    assert (entries.loc[entries["競馬場コード"] == "08", "データ区分"] == "出走馬名表").all()
    assert entries.loc[entries["競馬場コード"] == "05", "枠順確定"].all()


def test_prepare_entries_drops_race_once_results_start():
    raw = make_raw(n_days=12, entry_days=(10,))
    ra = raw["RA"]
    row = ra.loc[(ra["head.DataKubun"] == "2") & (ra["id.JyoCD"] == "05")].iloc[0].copy()
    row["head.DataKubun"] = "3"                            # 速報（結果が出始めた）
    row["_seq"] = str(10 ** 6)
    raw["RA"] = pd.concat([ra, row.to_frame().T], ignore_index=True)
    entries = jvmap.prepare_entries(raw)
    assert "05" not in set(entries["競馬場コード"])


def test_entries_go_through_prediction_path():
    raw = make_raw(n_days=40, entry_days=(39,))
    ds = jvmap.build_dataset(raw=raw)
    entries = jvmap.prepare_entries(raw)
    pred = predict.build_prediction_features(ds.race_result, entries, ds.lap_df)
    assert len(pred) == len(entries)
    assert (pred["馬場状態の扱い"] == "未発表（良と仮定）").all()
    assert (pred["馬場状態1"].astype(str) == "良").all()
    assert pred["騎手通算騎乗数"].gt(0).any()
    pred2 = predict.build_prediction_features(ds.race_result, entries, ds.lap_df, going="重")
    assert (pred2["馬場状態1"].astype(str) == "重").all()


# ===========================================================================
# モデル・出力
# ===========================================================================
_MODEL_CACHE = {}


def _tiny_model(ds, rounds=20):
    if rounds not in _MODEL_CACHE:
        frame = predict.make_training_frame(ds.race_result, ds.lap_df)
        booster, cols, info = predict.train_model(frame, rounds=rounds, start="2015-01-01")
        meta = {"name": "test", "feature_cols": cols, "train_last_date": info["train_last_date"]}
        _MODEL_CACHE[rounds] = predict.SavedModel(booster, meta, Path("."))
    return _MODEL_CACHE[rounds]


def test_training_is_reproducible(ds):
    frame = predict.make_training_frame(ds.race_result, ds.lap_df)
    b1, cols, _ = predict.train_model(frame, rounds=15, start="2015-01-01")
    b2, _, _ = predict.train_model(frame, rounds=15, start="2015-01-01")
    x = frame[cols]
    assert np.array_equal(b1.predict(x), b2.predict(x))


def test_until_limits_training_and_features(ds):
    frame = predict.make_training_frame(ds.race_result, ds.lap_df, until="2015-06-30")
    assert frame["レース日付"].max() <= pd.Timestamp("2015-06-30")
    _, _, info = predict.train_model(frame, rounds=5, start="2015-01-01", until="2015-06-30")
    assert info["train_last_date"] <= "2015-06-30"


def test_save_and_load_model(ds, tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "data_root", lambda: tmp_path)
    frame = predict.make_training_frame(ds.race_result, ds.lap_df)
    booster, cols, info = predict.train_model(frame, rounds=5, start="2015-01-01")
    path = predict.save_model(booster, cols, info, 5, None, {"data_last_race_date": "x"})
    assert (path / "model.txt").exists() and (path / "meta.json").exists()
    m = predict.load_model()
    assert m.feature_cols == cols and m.meta["rounds"] == 5
    assert m.meta["train_first_date"] >= "2015-01-01"
    assert "git_commit" in m.meta and m.meta["as_of"] == "day"
    x = frame[cols].head(50)
    assert np.allclose(m.booster.predict(x), booster.predict(x))


def test_output_table_first_seven_columns(ds):
    model = _tiny_model(ds)
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 58)
    _, _, pred = _replay_frames(ds, day)
    scored = predict.score(model, pred)
    table = predict.output_table(scored, model)
    assert list(table.columns[:7]) == ["レースID", "日付", "レース名", "馬番", "馬名", "予測確率", "予測順位"]
    assert "騎手" in table.columns and "格付け" in table.columns and "データ区分" in table.columns
    for _, g in scored.groupby("レースID"):
        assert np.isclose(g["予測確率"].sum(), 1.0)
        assert sorted(g["予測順位"]) == list(range(1, len(g) + 1))


def test_render_race_shows_box_and_warning(ds):
    model = _tiny_model(ds)
    raw = make_raw(n_days=40, entry_days=(39,))
    ds2 = jvmap.build_dataset(raw=raw)
    entries = jvmap.prepare_entries(raw)
    entries.loc[entries["競馬場コード"] == "05", "枠順確定"] = False
    scored = predict.score(model, predict.build_prediction_features(ds2.race_result, entries, ds2.lap_df))
    race = scored.loc[scored["競馬場コード"] == "05"]
    text = predict.render_race(race, model)
    assert "の6頭BOX 120点 12,000円" in text
    assert "枠順が未確定" in text
    assert "良と仮定" in text


def test_actual_result_hit_and_hot(ds):
    model = _tiny_model(ds)
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 58)
    _, targets, pred = _replay_frames(ds, day)
    scored = predict.score(model, pred)
    rid = scored["レースID"].iloc[0]
    race_rows = targets.loc[targets["レースID"] == rid]
    s = scored.loc[scored["レースID"] == rid].copy()
    # 実際の1〜3着を上位に置けば必ず的中
    top3 = race_rows.nsmallest(3, "着順")["血統登録番号"]
    s["予測順位"] = s["予測順位"] + 10
    s.loc[s["血統登録番号"].isin(top3), "予測順位"] = [1, 2, 3]
    res = predict.actual_result(race_rows, ds.payout_df, s)
    assert res["的中"]
    pay = ds.payout_df.set_index("レースID").loc[rid, "3連単払戻"]
    assert res["熱い"] == (pay >= 50000)


# ===========================================================================
# CLI（py -m src.v6）を合成データで通す
# ===========================================================================
@pytest.fixture()
def jv_dir(tmp_path, monkeypatch):
    raw = make_raw(n_days=60, entry_days=(59,))
    d = tmp_path / "jvlink"
    d.mkdir()
    for rt, df in raw.items():
        df.to_csv(d / f"{rt}.csv", index=False, encoding="utf-8-sig")
    (d / "state.json").write_text('{"RACE": {"last_file_timestamp": "20160220112818"}}',
                                  encoding="utf-8")
    monkeypatch.setattr(predict, "data_root", lambda: tmp_path)
    return d


def test_cli_train_predict_replay(jv_dir, capsys):
    from src import v6

    assert v6.main(["--data", str(jv_dir), "train", "--rounds", "10", "--until", "2015-12-31"]) == 0
    out = capsys.readouterr().out
    assert "保存しました" in out
    model = predict.load_model()
    assert model.meta["train_last_date"] <= "2015-12-31"
    assert model.meta["jvlink_last_file_timestamp"]["RACE"] == "20160220112818"

    assert v6.main(["--data", str(jv_dir), "predict", "--from-date", "2015-01-01"]) == 0
    out = capsys.readouterr().out
    assert "の6頭BOX 120点 12,000円" in out
    assert "取得 2016-02-20 11:28 時点" in out
    csv = sorted((jv_dir.parent / "predictions").glob("predict_*.csv"))[-1]
    table = pd.read_csv(csv, encoding="utf-8-sig")
    assert list(table.columns[:7]) == ["レースID", "日付", "レース名", "馬番", "馬名", "予測確率", "予測順位"]

    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 40)     # 2015-10-11
    assert v6.main(["--data", str(jv_dir), "replay", "--date", f"{day:%Y-%m-%d}"]) == 0
    out = capsys.readouterr().out
    assert "一致しなかったセル（馬体重・増減を除く）: 0" in out
    assert "学習終了日" in out          # モデルは 2015-12-31 まで見ている → 警告

    later = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 56)   # モデルが見ていない日
    assert v6.main(["--data", str(jv_dir), "replay", "--date", f"{later:%Y-%m-%d}"]) == 0
    assert "学習終了日" not in capsys.readouterr().out
    assert "まとめ:" in out

    assert v6.main(["--data", str(jv_dir), "replay", "--date", f"{day:%Y-%m-%d}",
                    "--assume-going"]) == 0
    out = capsys.readouterr().out
    assert "一致しなかったセル（馬体重・増減を除く）: 0" in out


def test_cli_replay_without_graded_races(jv_dir, capsys):
    from src import v6
    assert v6.main(["--data", str(jv_dir), "train", "--rounds", "5"]) == 0
    assert v6.main(["--data", str(jv_dir), "replay", "--date", "2015-01-05"]) == 1
    assert "重賞がありません" in capsys.readouterr().out


def test_cli_diagnose_small_commands(jv_dir, capsys):
    from src import v6
    assert v6.main(["--data", str(jv_dir), "diagnose", "payout-trend"]) == 0
    assert v6.main(["--data", str(jv_dir), "diagnose", "check-3f"]) == 0
    assert v6.main(["--data", str(jv_dir), "diagnose", "style-zero"]) == 0
    out = capsys.readouterr().out
    assert "重賞_中央値" in out and "前3F_一致率" in out and "確定前の区分" in out


# ===========================================================================
# 診断
# ===========================================================================
def test_substituted_weight_frame_matches_prediction_path(ds):
    from src import diagnose
    day = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 50)
    train = predict.make_training_frame(ds.race_result, ds.lap_df, until=f"{day:%Y-%m-%d}")
    sub = diagnose.substituted_weight_frame(train)
    rows = sub.loc[(sub["レース日付"] == day) & _graded(sub)]
    _, _, pred = _replay_frames(ds, day)
    diff = predict.compare_paths(rows, pred, _feature_cols(train), exclude=())
    assert (diff["不一致"] == 0).all(), diff.loc[diff["不一致"] > 0].to_string()


def test_weight_impact_cv_runs(ds):
    from src import diagnose, validate
    folds = [validate.Fold("t", (2015, 2015), (2016, 2016))]
    out = diagnose.weight_impact_cv(ds, folds=folds, verbose=False)
    t = out["table"]
    assert set(t["馬体重"]) == {"実際の馬体重", "代用（前走・増減0）"}
    assert t["上位6頭の単勝人気の平均"].notna().all()
    assert len(out["pooled"]) == 2


def test_three_f_agreement_on_laps():
    from src import diagnose
    raw = make_raw(n_days=8)
    ra = raw["RA"]
    for i in range(1, 26):
        ra[f"LapTime[{i}]"] = "000"
    # 1700m のレース：端数 100m + 200m×8。前3ハロン = 先頭3本（仕様書: 端数＋400m）
    ra["Kyori"] = "1700"
    laps = [65, 110, 115, 120, 122, 124, 121, 118, 119]
    for i, v in enumerate(laps, start=1):
        ra[f"LapTime[{i}]"] = f"{v:03d}"
    ra["HaronTimeS3"] = f"{sum(laps[:3]):03d}"
    ra["HaronTimeL3"] = f"{sum(laps[-3:]) + 1:03d}"   # 公式と0.1秒ずらす
    table = diagnose.three_f_agreement(jvmap.build_dataset(raw=raw), since="2015-01-01")
    row = table.loc[table["距離"] == 1700].iloc[0]
    assert row["前3F_一致率"] == 1.0
    assert row["後3F_一致率"] == 0.0
    assert row["後3F_差の平均"] == pytest.approx(-0.1, abs=1e-4)


def test_style_zero_table_breakdown():
    from src import diagnose
    raw = make_raw(n_days=12, entry_days=(11,))
    se = raw["SE"]
    se.loc[se.index[:3], "KyakusituKubun"] = "0"           # 最終データの中の 0
    table = diagnose.style_zero_table(raw)
    assert table["最終データ"].sum() == 3
    assert table["確定前の区分"].sum() == (se["head.DataKubun"] == "2").sum()


def test_auc_matches_definition():
    from src import diagnose
    y = [0, 0, 1, 1, 0, 1]
    p = [0.1, 0.4, 0.35, 0.8, 0.4, 0.9]
    # 正例と負例の全ペア（9組）のうち、正例の方が高い組 + 同点は0.5
    pairs = [(a, b) for a, ya in zip(p, y) if ya for b, yb in zip(p, y) if not yb]
    expect = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
    assert diagnose._auc(y, p) == pytest.approx(expect)
