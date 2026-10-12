"""v7 のテスト：スプレッドシートへの書き込み（偽物のワークシート）と、結果の照合。

本物の Google シートには一切つながない。
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import jvmap, predict, results, sheets  # noqa: E402
from test_v6_predict import make_raw  # noqa: E402


# ===========================================================================
# 偽物のワークシート（gspread.Worksheet の使う部分だけ）
# ===========================================================================
class FakeWorksheet:
    """値はシートと同じく文字列で持つ。USER_ENTERED の先頭 ' は文字列扱いの印なので外して保存する。"""

    def __init__(self, title: str, header: list[str], rows=()):
        self.title = title
        self.values = [list(header)] + [list(r) for r in rows]
        self.batch_calls = 0
        self.append_calls = 0

    def get_all_values(self):
        return [list(r) for r in self.values]

    @staticmethod
    def _store(v) -> str:
        return str(v)[1:] if isinstance(v, str) and v.startswith("'") else str(v)

    def batch_update(self, data, value_input_option=None):
        assert value_input_option == "USER_ENTERED"
        self.batch_calls += 1
        for d in data:
            m = re.fullmatch(r"([A-Z]+)(\d+)", d["range"])
            col = 0
            for ch in m.group(1):
                col = col * 26 + ord(ch) - 64
            row = int(m.group(2))
            while len(self.values) < row:
                self.values.append([])
            line = self.values[row - 1]
            line.extend([""] * (col - len(line)))
            line[col - 1] = self._store(d["values"][0][0])

    def append_rows(self, rows, value_input_option=None, table_range=None):
        assert value_input_option == "USER_ENTERED"
        self.append_calls += 1
        for r in rows:
            self.values.append([self._store(v) for v in r])

    def table(self) -> pd.DataFrame:
        width = len(self.values[0])
        return pd.DataFrame([r + [""] * (width - len(r)) for r in self.values[1:]], columns=self.values[0])


ENTRY_HEADER = ["レースID", "日付", "レース名", "馬番", "馬名", "予測確率", "予測順位", "発汗", "落着き", "メモ", "馬体重"]
RECORD_HEADER = ["レースID", "日付", "レース名", "馬番", "馬名", "予測確率", "予測順位", "発汗", "落着き", "メモ",
                 "着順", "単勝オッズ", "馬体重"]


def _pred_table(probs=None, horses=("アアア", "イイイ", "ウウウ"), rid="202610110511"):
    probs = probs or [0.5, 0.3, 0.2]
    return pd.DataFrame({"レースID": [rid] * len(horses), "日付": ["2026-10-11"] * len(horses),
                         "レース名": ["アイルランドトロフィー"] * len(horses),
                         "馬番": list(range(1, len(horses) + 1)), "馬名": list(horses),
                         "予測確率": probs[:len(horses)], "予測順位": list(range(1, len(horses) + 1))})


# ===========================================================================
# スプレッドシート
# ===========================================================================
def test_writes_by_header_name_not_position():
    ws = FakeWorksheet("評価入力", ENTRY_HEADER)
    rec = FakeWorksheet("記録", RECORD_HEADER)
    rows = _pred_table()
    sheets.upsert_rows(ws, rows, sheets.ENTRY_COLUMNS)
    t = ws.table()
    assert list(t["馬名"]) == ["アアア", "イイイ", "ウウウ"]
    assert list(t["レースID"]) == ["202610110511"] * 3
    assert (t["馬体重"] == "").all()                      # 評価入力の馬体重には書かない
    # 列の順番が違うシートでも、同じ名前の列に入る
    shuffled = FakeWorksheet("評価入力", ["メモ", "馬名", "予測順位", "レースID", "予測確率", "日付",
                                         "レース名", "馬番", "発汗", "落着き", "馬体重"])
    sheets.upsert_rows(shuffled, rows, sheets.ENTRY_COLUMNS)
    assert list(shuffled.table()["予測順位"]) == ["1", "2", "3"]
    assert list(shuffled.table()["メモ"]) == ["", "", ""]
    assert rec.table().empty


def test_repredict_overwrites_without_duplicates_and_keeps_manual_columns():
    ws = FakeWorksheet("評価入力", ENTRY_HEADER)
    sheets.upsert_rows(ws, _pred_table(), sheets.ENTRY_COLUMNS)
    # スマホで手入力した欄
    ws.values[1][ENTRY_HEADER.index("発汗")] = "多い"
    ws.values[1][ENTRY_HEADER.index("メモ")] = "イレ込み"
    ws.values[2][ENTRY_HEADER.index("落着き")] = "○"
    res = sheets.upsert_rows(ws, _pred_table(probs=[0.25, 0.45, 0.30]).assign(
        予測順位=[3, 1, 2]), sheets.ENTRY_COLUMNS)
    t = ws.table()
    assert len(t) == 3 and res.appended_rows == 0 and res.updated_rows == 3
    assert list(t["予測順位"]) == ["3", "1", "2"]
    assert t.loc[0, "発汗"] == "多い" and t.loc[0, "メモ"] == "イレ込み" and t.loc[1, "落着き"] == "○"


def test_scratched_horse_row_is_kept_and_marked():
    ws = FakeWorksheet("評価入力", ENTRY_HEADER)
    sheets.upsert_rows(ws, _pred_table(), sheets.ENTRY_COLUMNS)
    ws.values[3][ENTRY_HEADER.index("メモ")] = "良く見えた"
    res = sheets.upsert_rows(ws, _pred_table(probs=[0.6, 0.4], horses=("アアア", "イイイ")),
                             sheets.ENTRY_COLUMNS)
    t = ws.table()
    assert len(t) == 3 and res.scratched_rows == 1
    row = t.loc[t["馬名"] == "ウウウ"].iloc[0]
    assert row["予測順位"] == sheets.SCRATCHED and row["予測確率"] == "" and row["メモ"] == "良く見えた"


def test_other_races_are_untouched():
    other = ["202610110411", "2026-10-11", "別のレース", "1", "エエエ", "0.9", "1", "", "", "メモ", ""]
    ws = FakeWorksheet("評価入力", ENTRY_HEADER, [other])
    sheets.upsert_rows(ws, _pred_table(), sheets.ENTRY_COLUMNS)
    assert ws.values[1] == other


def test_protected_columns_are_never_written():
    ws = FakeWorksheet("評価入力", ENTRY_HEADER)
    with pytest.raises(ValueError):
        sheets.upsert_rows(ws, _pred_table().assign(メモ="x"), sheets.ENTRY_COLUMNS + ["メモ"])


def test_missing_header_is_an_error():
    ws = FakeWorksheet("評価入力", ["レースID", "馬名", "予測確率"])
    with pytest.raises(SystemExit, match="見出しに無い列"):
        sheets.upsert_rows(ws, _pred_table(), sheets.ENTRY_COLUMNS)


def test_existing_numeric_or_quoted_race_id_matches():
    rows = [["202610110511.0", "2026/10/11", "x", "1", "アアア", "0.1", "9", "", "", "", ""],
            ["'202610110511", "2026/10/11", "x", "2", "イイイ", "0.1", "9", "", "", "", ""]]
    ws = FakeWorksheet("評価入力", ENTRY_HEADER, rows)
    res = sheets.upsert_rows(ws, _pred_table(horses=("アアア", "イイイ"), probs=[0.6, 0.4]),
                             sheets.ENTRY_COLUMNS)
    assert res.appended_rows == 0 and len(ws.table()) == 2


def test_no_write_when_nothing_changed_and_updated_at_column():
    ws = FakeWorksheet("評価入力", ENTRY_HEADER + ["更新日時"])
    sheets.upsert_rows(ws, _pred_table(), sheets.ENTRY_COLUMNS, now=datetime(2026, 10, 10, 20, 0))
    assert set(ws.table()["更新日時"]) == {"2026-10-10 20:00:00"}
    calls = ws.batch_calls
    res = sheets.upsert_rows(ws, _pred_table(), sheets.ENTRY_COLUMNS, now=datetime(2026, 10, 11, 9, 0))
    assert res.changed_cells == 0 and ws.batch_calls == calls
    assert set(ws.table()["更新日時"]) == {"2026-10-10 20:00:00"}


def test_settings_from_env_and_file(tmp_path, monkeypatch):
    key = tmp_path / "key.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("KEIBA_GOOGLE_KEY", raising=False)
    monkeypatch.delenv("KEIBA_SHEET_ID", raising=False)
    cfg = tmp_path / "local_settings.json"
    cfg.write_text('{"google_service_account_file": "%s", "spreadsheet_id": "abc"}'
                   % str(key).replace("\\", "\\\\"), encoding="utf-8")
    s = sheets.load_settings(cfg)
    assert s.spreadsheet_id == "abc" and s.entry_sheet == "評価入力" and s.record_sheet == "記録"
    monkeypatch.setenv("KEIBA_SHEET_ID", "from-env")
    assert sheets.load_settings(cfg).spreadsheet_id == "from-env"
    monkeypatch.delenv("KEIBA_SHEET_ID")
    with pytest.raises(SystemExit, match="設定がありません"):
        sheets.load_settings(tmp_path / "none.json")


def test_local_settings_are_gitignored():
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "local_settings.json" in gi.splitlines()


def test_cli_predict_sheet_writes_entry_sheet(monkeypatch, capsys):
    from src import v6
    ws = FakeWorksheet("評価入力", ENTRY_HEADER)
    table = _pred_table().assign(騎手="x")
    v6._write_entry_sheet(table, ws=ws)
    assert len(ws.table()) == 3
    assert "書き込みました" in capsys.readouterr().out


# ===========================================================================
# 結果の照合
# ===========================================================================
START = pd.Timestamp("2015-01-04")


def _day(i: int) -> pd.Timestamp:
    return START + pd.Timedelta(days=7 * i)


def _write_pred(pred_dir: Path, rows: pd.DataFrame, kind: str, made: datetime) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{kind}_x_{made:%Y%m%d_%H%M%S}.csv"
    rows.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _race_pred(raw, day_index: int, jyo: str = "05", order=None, until="2014-12-31"):
    """合成データの1レースの予測（order: 予測順位を付ける馬番の並び）。"""
    rr = jvmap.build_dataset(raw=raw).race_result
    race = rr.loc[(rr["レース日付"] == _day(day_index)) & (rr["競馬場コード"] == jyo)].sort_values("馬番")
    order = order or list(race["馬番"])
    rank = {int(p): i + 1 for i, p in enumerate(order)}
    return pd.DataFrame({
        "レースID": race["レースID"], "日付": f"{_day(day_index):%Y-%m-%d}", "レース名": race["競走名"],
        "馬番": race["馬番"].astype(int), "馬名": race["馬名表示"],
        "予測確率": [1 / len(race)] * len(race), "予測順位": race["馬番"].astype(int).map(rank),
        "格付け": race["リステッド・重賞競走"], "モデル学習終了日": until,
    }).sort_values("予測順位"), race


def test_hit_payout_and_balance(tmp_path):
    raw = make_raw(n_days=12)
    pred, race = _race_pred(raw, 10)
    top3 = list(race.sort_values("着順")["馬番"].astype(int)[:3])
    others = [int(x) for x in race["馬番"] if int(x) not in top3]
    pred, _ = _race_pred(raw, 10, order=top3 + others)     # 1〜3着を BOX に入れる → 必ず的中
    _write_pred(tmp_path, pred, "predict", datetime(2015, 3, 14, 20, 0))
    out = results.reconcile(results.load_predictions(tmp_path), raw)
    r = out["races"].iloc[0]
    pay = jvmap.prepare_payout(raw["HR"]).set_index("レースID").loc[r["レースID"], "3連単払戻"]
    assert r["的中"] and r["払戻"] == pay and r["投資"] == 12_000
    assert r["収支"] == pay - 12_000 and r["熱い"] == (pay >= 50_000)
    assert r["1着"] == str(top3[0])

    pred_miss, _ = _race_pred(raw, 10, order=others + top3)  # 1〜3着を BOX の外に
    _write_pred(tmp_path, pred_miss, "predict", datetime(2015, 3, 15, 9, 0))
    out = results.reconcile(results.load_predictions(tmp_path), raw)
    r = out["races"].iloc[0]                                   # 発走前の最後（9:00）の予測を使う
    assert not r["的中"] and r["収支"] == -12_000 and r["予測ファイル"].endswith("20150315_090000.csv")


def test_prediction_after_post_time_is_not_used(tmp_path):
    raw = make_raw(n_days=12)
    pred, _ = _race_pred(raw, 10)
    _write_pred(tmp_path, pred, "predict", datetime(2015, 3, 15, 16, 0))   # 発走 15:30 の後
    out = results.reconcile(results.load_predictions(tmp_path), raw)
    assert out["races"].empty and len(out["skipped"]) == 1
    _write_pred(tmp_path, pred, "predict", datetime(2015, 3, 15, 15, 0))
    out = results.reconcile(results.load_predictions(tmp_path), raw)
    assert out["races"].iloc[0]["予測ファイル"].endswith("20150315_150000.csv")


def test_scratched_box_horse_is_refunded(tmp_path):
    raw = make_raw(n_days=12)
    pred, race = _race_pred(raw, 10)
    rid = race["レースID"].iloc[0]
    box_last = int(pred.loc[pred["予測順位"] == 6, "馬番"].iloc[0])
    se = raw["SE"]
    rid_of = jvmap.make_race_id(se)
    i = se.index[(rid_of == rid) & (se["Umaban"].astype(int) == box_last)][0]
    se.loc[i, ["IJyoCD", "KakuteiJyuni"]] = ["1", "00"]
    _write_pred(tmp_path, pred, "predict", datetime(2015, 3, 14, 20, 0))
    out = results.reconcile(results.load_predictions(tmp_path), raw)
    r = out["races"].iloc[0]
    assert r["投資"] == 5 * 4 * 3 * 100 and r["返還された馬"]
    rec = results.record_rows(out["horses"])
    assert "取消" in set(rec["着順"].astype(str))


def test_dead_heat_combos_any_hit(tmp_path):
    raw = make_raw(n_days=12)
    pred, race = _race_pred(raw, 10)
    rid = race["レースID"].iloc[0]
    box = sorted(int(x) for x in pred.loc[pred["予測順位"] <= 6, "馬番"])
    hr = raw["HR"]
    j = hr.index[jvmap.make_race_id(hr) == rid][0]
    hr.loc[j, "PaySanrentan[1].Kumi"] = "080706"            # BOX 外（合成データは8頭、BOX は 1〜6番）
    hr.loc[j, "PaySanrentan[2].Kumi"] = "".join(f"{x:02d}" for x in box[:3])
    hr.loc[j, "PaySanrentan[2].Pay"] = "000060000"
    _write_pred(tmp_path, pred, "predict", datetime(2015, 3, 14, 20, 0))
    r = results.reconcile(results.load_predictions(tmp_path), raw)["races"].iloc[0]
    assert r["的中"] and r["払戻"] == 60_000 and r["熱い"]


def test_pending_and_practice_are_separated(tmp_path):
    raw = make_raw(n_days=12, entry_days=(11,))
    pred_live, _ = _race_pred(raw, 10)
    _write_pred(tmp_path, pred_live, "predict", datetime(2015, 3, 14, 20, 0))
    # 結果がまだ無いレース（出馬表の段階）
    entries = jvmap.prepare_entries(raw, graded_only=False)
    e = entries.loc[entries["競馬場コード"] == "05"]
    pend = pd.DataFrame({"レースID": e["レースID"], "日付": f"{_day(11):%Y-%m-%d}", "レース名": e["競走名"],
                         "馬番": e["馬番"].astype(int), "馬名": e["馬名表示"], "予測確率": 0.125,
                         "予測順位": range(1, len(e) + 1), "格付け": e["リステッド・重賞競走"]})
    _write_pred(tmp_path, pend, "predict", datetime(2015, 3, 21, 20, 0))
    # 練習：学習終了日がレース日より前（有効）と後（参考外）
    p1, _ = _race_pred(raw, 9, until="2015-03-01")
    _write_pred(tmp_path, p1, "replay", datetime(2026, 10, 1, 12, 0))
    p2, _ = _race_pred(raw, 8, until="2015-12-31")
    _write_pred(tmp_path, p2, "replay", datetime(2026, 10, 1, 12, 5))

    out = results.reconcile(results.load_predictions(tmp_path), raw)
    kinds = set(out["races"]["区分"])
    assert kinds == {results.LIVE, results.PRACTICE, results.PRACTICE + "・参考外"}
    assert list(out["pending"]["状態"]) == ["結果待ち"]
    summ = results.summary(out["races"]).set_index("区分")
    assert summ.loc[results.LIVE, "レース数"] == 1
    assert summ.loc[results.PRACTICE, "レース数"] == 1
    assert summ.loc["validate の期待値（as_of=day・馬体重代用）", "達成率"] == pytest.approx(0.047)
    rec = results.record_rows(out["horses"])
    assert list(rec.columns) == sheets.RECORD_COLUMNS
    assert rec["レースID"].nunique() == 2                    # 参考外は書かない


def test_record_sheet_write(tmp_path, capsys):
    from src import v6
    raw = make_raw(n_days=12)
    pred, _ = _race_pred(raw, 10)
    _write_pred(tmp_path, pred, "predict", datetime(2015, 3, 14, 20, 0))
    out = results.reconcile(results.load_predictions(tmp_path), raw)
    ws = FakeWorksheet("記録", RECORD_HEADER, [["x", "", "", "", "", "", "", "多い", "", "メモ", "", "", ""]])
    v6._write_record_sheet(results.record_rows(out["horses"]), ws=ws)
    t = ws.table()
    assert len(t) == 9 and t.loc[0, "発汗"] == "多い"
    got = t.iloc[1:]
    assert set(got["着順"]) == {str(i) for i in range(1, 9)}
    assert (got["単勝オッズ"] != "").all() and (got["馬体重"] != "").all()


def test_cli_results(tmp_path, monkeypatch, capsys):
    from src import v6
    raw = make_raw(n_days=12)
    d = tmp_path / "jvlink"
    d.mkdir()
    for rt, df in raw.items():
        df.to_csv(d / f"{rt}.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(predict, "data_root", lambda: tmp_path)
    pred, _ = _race_pred(raw, 10)
    _write_pred(tmp_path / "predictions", pred, "predict", datetime(2015, 3, 14, 20, 0))
    assert v6.main(["--data", str(d), "results"]) == 0
    out = capsys.readouterr().out
    assert "通算の成績表" in out and "validate の期待値" in out and "収支" in out
    assert list((tmp_path / "results").glob("summary_*.csv"))


# ===========================================================================
# 週末の結果欠落の影響・学習開始年の比較
# ===========================================================================
def make_weekend_raw(n_weeks: int = 40) -> dict:
    """土曜と日曜に開催がある合成データ（騎手は土日で重なる。馬は土日で別）。"""
    sun = make_raw(n_days=n_weeks, start="2015-01-04", seed=1)
    sat = make_raw(n_days=n_weeks, start="2015-01-03", seed=2)
    for rt in ("RA", "SE", "HR"):
        sat[rt]["id.RaceNum"] = "10"                         # レースIDが日曜とぶつからないように
        sat[rt]["_seq"] = (sat[rt]["_seq"].astype(int) + 10 ** 6).astype(str)
    sat["SE"]["KettoNum"] = sat["SE"]["KettoNum"].str.replace("2012", "2013", n=1)
    return {rt: pd.concat([sat[rt], sun[rt]], ignore_index=True) for rt in ("RA", "SE", "HR")}


def test_weekend_block_start():
    from src import diagnose
    d = pd.Series(pd.to_datetime(["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-18",
                                   "2026-10-17", "2026-12-28", "2027-01-05"]))
    got = diagnose.weekend_block_start(d).dt.strftime("%m-%d").tolist()
    assert got == ["10-10", "10-10", "10-10", "10-17", "10-17", "12-28", "01-05"]


def test_weekend_frame_equals_prediction_path_without_same_weekend_results():
    """週末より前の特徴量 = 土曜の結果が無いまま日曜を予測の経路で作ったもの（本番と同じ）。"""
    from src import diagnose
    ds = jvmap.build_dataset(raw=make_weekend_raw())
    wk = diagnose.build_feature_frame(ds, weekend=True)
    day = diagnose.build_feature_frame(ds, weekend=False)
    sun = pd.Timestamp("2015-01-04") + pd.Timedelta(days=7 * 35)
    rr = ds.race_result
    dates = pd.to_datetime(rr["レース日付"])
    graded = rr["リステッド・重賞競走"].isin(["G1", "G2", "G3", "G"])
    targets = rr.loc[(dates == sun) & graded]
    history = rr.loc[dates < sun - pd.Timedelta(days=1)]          # 土曜の結果は無い
    pred = predict.build_prediction_features(history, targets, ds.lap_df, keep_actual_going=True,
                                             substitute_weights=False)
    rows = wk.loc[(wk["レース日付"] == sun) & wk["リステッド・重賞競走"].isin(["G1", "G2", "G3", "G"])]
    cols = [c for c in pred.columns if c in set(predict.features.feature_columns(wk))]
    diff = predict.compare_paths(rows, pred, cols, exclude=())
    assert (diff["不一致"] == 0).all(), diff.loc[diff["不一致"] > 0].to_string()
    # 前日まで（day）の方は土曜の結果を使うので、騎手の成績が違うはず
    rows_day = day.loc[rows.index]
    assert (rows_day["騎手通算騎乗数"].to_numpy() > rows["騎手通算騎乗数"].to_numpy()).any()


def test_paired_proportion_test_properties():
    from src import trifecta
    rng = np.random.default_rng(0)
    x = rng.random(2000) < 0.05
    same = trifecta.paired_proportion_test(x, x)
    assert same["差"] == 0 and same["差下限"] <= 0 <= same["差上限"] and same["p値"] == 1.0
    y = x.copy()
    y[np.flatnonzero(~x)[:30]] = True                             # y だけ30レース多く当たる
    t = trifecta.paired_proportion_test(x, y)
    assert t["yだけ"] == 30 and t["xだけ"] == 0 and t["差上限"] < 0 and t["p値"] < 0.001
    # 対応のある区間は、別々の標本とみなした区間より狭い
    un = trifecta.two_proportion_test(int(x.sum()), len(x), int(y.sum()), len(y))
    assert (t["差上限"] - t["差下限"]) < (un["差上限"] - un["差下限"])
    # Newcombe (1998) の方法10を手計算した値（a=12, b=9, c=2, d=21）
    a, b, c, d = 12, 9, 2, 21
    xx = np.array([1] * a + [1] * b + [0] * c + [0] * d, bool)
    yy = np.array([1] * a + [0] * b + [1] * c + [0] * d, bool)
    r = trifecta.paired_proportion_test(xx, yy)
    assert r["差"] == pytest.approx(7 / 44)
    assert r["差下限"] == pytest.approx(0.0112, abs=5e-4) and r["差上限"] == pytest.approx(0.2954, abs=5e-4)


def test_weekend_gap_cv_runs():
    from src import diagnose, validate
    ds = jvmap.build_dataset(raw=make_weekend_raw())
    out = diagnose.weekend_gap_cv(ds, folds=[validate.Fold("t", (2015, 2015), (2015, 2015))],
                                  verbose=False)
    assert list(out["table"]["条件"]) == ["A", "B", "C"]
    assert out["影響を受ける割合"]["割合"] == pytest.approx(0.5, abs=0.1)   # 日曜の重賞＝2日目
    assert {"AUC_A", "AUC_B", "AUC_C"} <= set(out["auc"].columns)


def test_start_year_cv_rule_and_runs():
    from src import diagnose, validate
    assert "結果を見る前に固定" in diagnose.START_YEAR_RULE and "下限が 0 を超えたら" in diagnose.START_YEAR_RULE
    ds = jvmap.build_dataset(raw=make_raw(n_days=60))
    out = diagnose.start_year_cv(ds, folds=[validate.Fold("t", (2015, 2015), (2016, 2016))],
                                 years=(2014, 2015), verbose=False)
    # 2014年からでも2015年からでも学習データは同じ → 差は0 → 採用しない
    row = out["table"].set_index("条件").loc["学習開始2014"]
    assert row["差（−学習開始2015）"] == 0 and not out["adopt"] and out["verdict"].startswith("不採用")


def test_practice_prefers_model_that_did_not_see_the_race(tmp_path):
    raw = make_raw(n_days=12)
    ok, _ = _race_pred(raw, 9, until="2015-03-01")
    seen, _ = _race_pred(raw, 9, until="2015-12-31")
    _write_pred(tmp_path, ok, "replay", datetime(2026, 10, 1, 12, 0))
    _write_pred(tmp_path, seen, "replay", datetime(2026, 10, 5, 12, 0))   # 後から全データのモデルで再実行
    r = results.reconcile(results.load_predictions(tmp_path), raw)["races"].iloc[0]
    assert r["区分"] == results.PRACTICE and r["予測ファイル"].endswith("20261001_120000.csv")
