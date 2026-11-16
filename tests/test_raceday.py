"""v10 当日の仕組み（src/raceday.py）のテスト。JV-Link とシートは使わず、偽物の取り込み・保存したオッズで確かめる。

- 時刻のきまり：発走60分前に仮、20分前に最終。最終のあとは更新しない。失敗しても前の買い目が残る
- 当日の傾向：発走の40分以上前・同じ場・同じ芝ダの、届いていたレースだけ（後のレースの結果は使わない）
- 当日の結果が取れないときは「市場だけ」
- 本物のオッズで5万円に届かない買い目の警告と、届く金額
- 時計の確認（ネットの時刻と9時間ずれ・タイムゾーン・発表時刻が未来）
- タスクスケジューラの定義（XML）・答え合わせ（返還）・ドライラン（過去の日の再現）
"""

from __future__ import annotations

import itertools
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import bets, jvmap, market, odds, predict, raceday  # noqa: E402
from test_v8 import make_raw_v8  # noqa: E402


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "data_root", lambda: tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# 合成の1レース
# ---------------------------------------------------------------------------
DATE = "20261101"


def _race(n=12, rid="202605040511", hhmm="1540", jyo="05", surface="芝"):
    return pd.DataFrame({"レースID": rid, "レース日付": pd.Timestamp(DATE), "競馬場コード": jyo, "競馬場名": "東京",
                         "競走名": "テスト記念", "リステッド・重賞競走": "G1", "芝・ダート区分": surface, "距離(m)": 2000,
                         "発走時刻": hhmm, "公式出走頭数": n, "馬番": np.arange(1, n + 1), "馬名表示": [f"馬{i}" for i in range(n)],
                         "血統登録番号": [f"H{i:03d}" for i in range(n)], "脚質スコア": np.linspace(0.1, 0.9, n),
                         "着順": np.nan})


def _snap(n=12, seed=0, scale=1.0, announced="11011518"):
    """単勝オッズと、推定配当 × scale の本物のオッズ（4券種）。"""
    rng = np.random.default_rng(seed)
    q = rng.dirichlet(np.ones(n) * 2)
    win = {i + 1: round(0.8 / q[i], 1) for i in range(n)}
    inv = 1 / np.array(list(win.values()))
    qq = inv / inv.sum()
    Q = market.combo_probabilities(market.trifecta_tensor(qq, 0.85, 0.75))
    snap = {"単勝": win, "発表": announced, "取得時刻": "2026-11-01 15:19:30", "データ区分": "1"}
    posts = np.arange(1, n + 1)
    for kind, r in [("3連単", 3), ("3連複", 3), ("馬単", 2), ("馬連", 2)]:
        combos = list(itertools.permutations(range(n), r)) if kind in ("3連単", "馬単") else \
            list(itertools.combinations(range(n), r))
        pay = np.array([100 * market.PAYOUT_RATE[kind] / Q[kind][c] * scale for c in combos])
        snap[kind] = (np.array([[posts[i] for i in c] for c in combos]), np.round(pay / 10) * 10)
    return snap


COEFS = {"β": [1.0, 3.0, 5.0], "項": ["当日_内枠×枠位置", "当日_前残り×脚質"], "γ": 0.85, "δ": 0.75,
         "期間": ["2011-01-01", "2026-10-31"], "推定レース数": 1}


def _day_results(hhmms, jyo="05", surface="芝", inside_wins=True, n=10):
    """その日の先に終わったレース（内枠が1〜3着・4角で前にいた馬が残った形）。"""
    rows = []
    for r, hhmm in enumerate(hhmms, 1):
        for i in range(n):
            rank = i + 1 if inside_wins else n - i
            rows.append({"レースID": f"2026050405{r:02d}", "レース日付": pd.Timestamp(DATE), "競馬場コード": jyo,
                         "芝・ダート区分": surface, "距離(m)": 1600, "発走時刻": hhmm, "公式出走頭数": n,
                         "馬番": i + 1, "血統登録番号": f"D{r}{i:02d}", "着順": rank, "タイム": 95.0 + i * 0.1,
                         "_4コーナー順位": i + 1, "単勝オッズ": 5.0})
    return pd.DataFrame(rows)


def test_stage_and_key():
    post = datetime(2026, 11, 1, 15, 40)
    assert raceday.stage_of(post - timedelta(minutes=90), post) == "latest"
    assert raceday.stage_of(post - timedelta(minutes=60), post) == "provisional"
    assert raceday.stage_of(post - timedelta(minutes=21), post) == "provisional"
    assert raceday.stage_of(post - timedelta(minutes=20), post) == "final"
    assert raceday.stage_of(post, post) == "発走後"
    assert raceday.rt_key("20261101", "202605040511") == "2026110105040511"


def test_compute_plan_market_only_and_rules():
    plan = raceday.compute_plan(_race(), _snap(), COEFS, None, results_ok=False)
    assert plan["確率の出し方"].startswith("市場だけ") and not plan["当日の結果を使えた"]
    assert plan["点数"] <= raceday.MAX_POINTS and plan["合計"] <= raceday.BUDGET
    order = [raceday.SMAPPY_ORDER.index(b["券種"]) for b in plan["買い目"]]
    assert order == sorted(order)                                   # スマッピーに入れる順
    assert set(plan["熱さ"]) == {50_000, 40_000, 30_000}
    assert plan["熱さ"][50_000] <= plan["熱さ"][40_000] + 1e-9 <= plan["熱さ"][30_000] + 2e-9
    assert plan["本物のオッズで確かめた"]
    # 本物のオッズ＝推定どおりなら、余裕 1.4 があるので全部 5万円に届く
    assert plan["届かない買い目"] == 0


def test_compute_plan_warns_when_real_odds_short():
    plan = raceday.compute_plan(_race(), _snap(scale=0.5), COEFS, None, results_ok=False)   # 本物は推定の半分
    short = [b for b in plan["買い目"] if b["届く"] is False]
    assert short and plan["届かない買い目"] == len(short)
    for b in short:
        assert b["当たったときの払戻"] < 50_000
        assert b["届く金額"] * b["本物のオッズ"] >= 50_000 - 1e-6 and b["足りない額"] == b["届く金額"] - b["金額"]
    assert plan["足りない額の合計"] == sum(b["足りない額"] for b in short)


def test_scratched_horse_is_dropped():
    snap = _snap()
    del snap["単勝"][5]                                             # 5番は取消（単勝オッズが無い）
    plan = raceday.compute_plan(_race(), snap, COEFS, None, results_ok=False)
    assert all(5 not in b["馬番"] for b in plan["買い目"])


def test_day_terms_use_only_races_40_minutes_before():
    race = _race(hhmm="1540")
    early = _day_results(["1000", "1100", "1500"])                  # 15:00（15:40 − 40分）ちょうどまでは使える
    terms, used = raceday.day_terms(race, early)
    assert used == ["1R", "2R", "3R"]
    assert (terms["当日_使えたレース数"] == 3).all()
    # 後のレース（15:01・別の場・ダート）を足しても変わらない
    more = pd.concat([early, _day_results(["1501"]).assign(レースID="202605040599"),
                      _day_results(["1000"], jyo="08").assign(レースID="202608040501"),
                      _day_results(["1000"], surface="ダート").assign(レースID="202605040598")], ignore_index=True)
    t2, used2 = raceday.day_terms(race, more)
    assert used2 == used
    pd.testing.assert_frame_equal(terms[["馬番"] + COEFS["項"]].reset_index(drop=True),
                                  t2[["馬番"] + COEFS["項"]].reset_index(drop=True))
    # 内枠が来ている日は、内枠の馬の項がプラス
    assert terms.sort_values("馬番")["当日_内枠×枠位置"].iloc[0] > 0


def test_compute_plan_with_day_trend_changes_probabilities():
    a = raceday.compute_plan(_race(), _snap(), COEFS, _day_results(["1000", "1100"]), results_ok=True)
    b = raceday.compute_plan(_race(), _snap(), COEFS, None, results_ok=True)
    assert a["確率の出し方"] == "当日の傾向を使った" and a["使ったレース"] == ["1R", "2R"]
    assert b["確率の出し方"].startswith("当日の傾向：使えるレースがまだ無い")


class FakeSource:
    replay = False

    def __init__(self, snap, day=None, results_ok=True):
        self.snap, self.day, self.ok = snap, day, results_ok
        self.timing = {}
        self.calls = 0

    def fetch_results(self, now):
        return self.day, self.ok, "偽物"

    def fetch_odds(self, keys, now):
        self.calls += 1
        return "偽物"

    def odds(self, rid, now):
        return self.snap


def _prep():
    t = _race()
    return {"日付": DATE, "係数": COEFS, "対象": t.to_dict("records"),
            "レース": [{"レースID": "202605040511", "発走時刻": "1540", "競馬場名": "東京", "競走名": "テスト記念",
                       "グレード": "G1", "芝ダ": "芝", "距離": 2000, "頭数": 12}]}


def test_run_once_stages_and_final_is_frozen(root):
    prep = _prep()
    d = raceday.day_dir(DATE) / "202605040511"
    at = lambda hm: datetime(2026, 11, 1, int(hm[:2]), int(hm[2:]))  # noqa: E731
    src = FakeSource(_snap(seed=1))
    raceday.run_once(prep, src, at("1400"))
    assert (d / "latest.json").exists() and not (d / "provisional.json").exists()
    raceday.run_once(prep, src, at("1440"))
    prov = raceday.read_json(d / "provisional.json")
    assert prov["段階"] == "仮" and not (d / "final.json").exists()
    raceday.run_once(prep, FakeSource(_snap(seed=2)), at("1450"))
    assert raceday.read_json(d / "provisional.json")["作成時刻"] == prov["作成時刻"]   # 仮は最初の1回だけ
    raceday.run_once(prep, src, at("1520"))
    final = raceday.read_json(d / "final.json")
    assert final["段階"] == "最終"
    other = FakeSource(_snap(seed=3))
    logs = raceday.run_once(prep, other, at("1530"))                 # 最終のあとは更新しない（取り込みもしない）
    assert raceday.read_json(d / "final.json") == final and other.calls == 0
    assert "保存済み" in logs[0]["結果の要約"]
    log = pd.read_csv(raceday.day_dir(DATE) / "log.csv", encoding="utf-8-sig")
    assert len(log) == 5 and list(log["段階"]) == ["途中", "仮", "仮", "最終", "最終"]


def test_run_once_keeps_previous_on_failure_and_falls_back(root):
    prep = _prep()
    d = raceday.day_dir(DATE) / "202605040511"
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 14, 0))
    before = raceday.read_json(d / "latest.json")
    logs = raceday.run_once(prep, FakeSource(None), datetime(2026, 11, 1, 14, 10))   # オッズが取れない
    assert "単勝オッズが無い" in logs[0]["結果の要約"] and raceday.read_json(d / "latest.json") == before
    raceday.run_once(prep, FakeSource(_snap(), results_ok=False), datetime(2026, 11, 1, 14, 20))
    assert raceday.read_json(d / "latest.json")["確率の出し方"].startswith("市場だけ")
    bad = _snap()
    bad["単勝"] = {1: 2.0, 2: 3.0}                                   # 計算できない
    logs = raceday.run_once(prep, FakeSource(bad), datetime(2026, 11, 1, 14, 30))
    assert "失敗" in logs[0]["結果の要約"] and raceday.read_json(d / "latest.json")["確率の出し方"].startswith("市場だけ")


def test_clock_check():
    pc = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)
    c = raceday.check_clock(net=lambda: pc + timedelta(hours=9), pc_utc=pc,
                            pc_local=pc.astimezone(raceday.JST).replace(tzinfo=None))
    assert not c.ok and "540 分" in c.warnings[0]
    assert abs((c.now - datetime(2026, 11, 2, 0, 0)).total_seconds()) < 5     # ネットの時刻 15:00 UTC = 日本時間 0:00
    ok = raceday.check_clock(net=lambda: pc + timedelta(seconds=30), pc_utc=pc,
                             pc_local=pc.astimezone(raceday.JST).replace(tzinfo=None))
    assert ok.ok
    tz = raceday.check_clock(net=lambda: pc, pc_utc=pc, pc_local=pc.replace(tzinfo=None))   # PC が UTC のまま
    assert any("タイムゾーン" in w for w in tz.warnings)
    nonet = raceday.check_clock(net=lambda: None, pc_utc=pc, pc_local=pc.astimezone(raceday.JST).replace(tzinfo=None))
    assert nonet.offset_min is None and nonet.ok
    c2 = raceday.Clock(datetime(2026, 11, 1, 15, 0), datetime(2026, 11, 1, 15, 0), 0.0)
    assert raceday.check_announce(c2, {"発表": "11011510"}, DATE) is not None
    assert raceday.check_announce(c2, {"発表": "11011458"}, DATE) is None


def test_render_sheet_shows_everything(root):
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap(scale=0.5)), datetime(2026, 11, 1, 15, 20))
    clock = raceday.Clock(datetime(2026, 11, 1, 6, 20), datetime(2026, 11, 1, 15, 20), 540.0, ["PC の時計が 540 分ずれています"])
    text = "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(prep, clock=clock))
    for s in ("時計の警告", "最終（これ以降は更新しない）", "予測の達成確率", "当日の傾向", "オッズ 15:18 発表",
              "券種 組番 金額", "合計", "届かない", "熱さ（参考）"):
        assert s in text, s


def test_sheet_exception_mark(root):
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))
    p = raceday.day_dir(DATE) / "202605040511" / "final.json"
    f = raceday.read_json(p)
    f["例外を考えてもいい場面"] = True
    raceday.write_json(p, f)
    assert "例外を考えてもいい場面" in "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(prep))


def test_task_xml():
    xml = raceday.task_xml(DATE, _prep(), r"C:\Python\pythonw.exe", r"C:\keiba")
    root = ET.fromstring(xml.split("\n", 1)[1])
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    starts = [e.text for e in root.findall(".//t:TimeTrigger/t:StartBoundary", ns)]
    assert starts[0] == "2026-11-01T09:00:00" and "2026-11-01T14:40:00" in starts and "2026-11-01T15:20:00" in starts
    assert root.find(".//t:Repetition/t:Interval", ns).text == "PT10M"
    assert root.find(".//t:LogonType", ns).text == "InteractiveToken"              # 管理者権限は不要
    assert root.find(".//t:MultipleInstancesPolicy", ns).text == "Queue"
    assert "update --date 20261101" in root.find(".//t:Arguments", ns).text


def test_install_task_calls_schtasks(root):
    calls = []

    class R:
        returncode = 0
    name, path, r = raceday.install_task(DATE, _prep(), run=lambda cmd: calls.append(cmd) or R())
    assert calls[0][:3] == ["schtasks", "/create", "/tn"] and name in calls[0] and path.exists()
    assert path.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff")                     # UTF-16


def test_score_plan_refund_and_kinds():
    plan = {"合計": 3000, "買い目": [{"券種": "3連単", "馬番": [3, 1, 2], "金額": 1000},
                                   {"券種": "3連複", "馬番": [2, 3, 1], "金額": 1000},
                                   {"券種": "馬連", "馬番": [5, 1], "金額": 1000}]}
    wins = {"3連単": {(3, 1, 2): 6000.0}, "3連複": {(1, 2, 3): 900.0}, "馬単": {}, "馬連": {(1, 3): 300.0}}
    s = raceday.score_plan(plan, wins, scratched={5})
    assert s["払戻"] == 60_000 + 9_000 and s["返還"] == 1000 and s["熱い"] and s["収支"] == 69_000 + 1000 - 3000


def _write_odds(data_dir: Path, ds, race_ids, date: str):
    """確定オッズ（単勝・4券種）を合成して保存。当たりの組は実際の払戻と同じにする。"""
    rr = ds.race_result
    pay = ds.payout_df.set_index("レースID")
    w = odds.OddsWriter(odds.odds_dir(data_dir), label="test")
    for rid in race_ids:
        r = rr.loc[rr["レースID"] == rid].sort_values("馬番")
        posts = r["馬番"].astype(int).to_numpy()
        o = r["単勝オッズ"].astype(float).to_numpy()
        w.add(odds.OddsRecord("O1", rid, date, "5", 20200101, "", posts.astype(np.int32),
                              np.round(o * 10).astype(np.int32), np.ones(len(posts), np.int16)))
        q = (1 / o) / (1 / o).sum()
        Q = market.combo_probabilities(market.trifecta_tensor(q, 0.85, 0.75))
        wins = bets.winning_combos(pay.loc[rid]) if rid in pay.index else {}
        n = len(posts)
        for kind, rt, r_ in [("馬連", "O2", 2), ("馬単", "O4", 2), ("3連複", "O5", 3), ("3連単", "O6", 3)]:
            combos = list(itertools.permutations(range(n), r_)) if kind in ("3連単", "馬単") else \
                list(itertools.combinations(range(n), r_))
            horses = np.array([[posts[i] for i in c] for c in combos])
            v = np.array([100 * market.PAYOUT_RATE[kind] / max(Q[kind][c], 1e-9) for c in combos])
            for combo, p in wins.get(kind, {}).items():
                m = (np.sort(horses, axis=1) == np.sort(combo)).all(axis=1) if kind in ("3連複", "馬連") \
                    else (horses == np.array(combo)).all(axis=1)
                v[m] = p
            kumi = sum(horses[:, i] * 100 ** (r_ - 1 - i) for i in range(r_))
            w.add(odds.OddsRecord(rt, rid, date, "5", 20200101, "", kumi.astype(np.int32),
                                  np.clip(np.round(v / 10), 1, 9_999_999).astype(np.int32), np.ones(len(kumi), np.int16)))
    w.flush()


@pytest.fixture()
def env(root):
    raw = make_raw_v8(n_days=120, races=8)
    d = root / "jvlink"
    d.mkdir()
    for rt, df in raw.items():
        df.to_csv(d / f"{rt}.csv", index=False, encoding="utf-8-sig")
    return d


def test_replay_day_end_to_end(env, capsys):
    ds = jvmap.build_dataset(env)
    rr = ds.race_result
    graded = rr.loc[rr["リステッド・重賞競走"].isin(["G1", "G2", "G3", "G"])]
    day = pd.to_datetime(graded["レース日付"]).sort_values().iloc[-1]
    date = f"{day:%Y%m%d}"
    ids = graded.loc[pd.to_datetime(graded["レース日付"]) == day, "レースID"].unique()
    _write_odds(env, ds, ids, date)
    assert raceday.main(["--data", str(env), "replay", "--date", date]) == 0
    out = capsys.readouterr().out
    assert "最終の買い目" in out and "答え合わせ" in out
    ddir = raceday.day_dir(date, replay=True)
    for rid in ids:
        f = raceday.read_json(ddir / rid / "final.json")
        assert f is not None and f["段階"] == "最終" and f["点数"] <= raceday.MAX_POINTS
        p = raceday.read_json(ddir / rid / "provisional.json")
        assert p["作成時刻"] <= f["作成時刻"]
        post = raceday.post_time(date, f["発走"].replace(":", ""))
        assert (post - datetime.strptime(f["作成時刻"], "%Y-%m-%d %H:%M:%S")) >= timedelta(minutes=raceday.FINAL_BEFORE - 0.01)
        assert "確定オッズで代用" in f["オッズの出どころ"]
    # ドライランの係数はその日より前のデータだけで推定（期間の終わりが前日）
    prep = raceday.read_json(ddir / "prep.json")
    assert prep["係数"]["期間"][1] < f"{day:%Y-%m-%d}"
    # 答え合わせ：ドライランの最終の買い目を、本番の場所に置いて score
    for rid in ids:
        dst = raceday.day_dir(date) / rid
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "final.json").write_text((ddir / rid / "final.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert raceday.main(["--data", str(env), "score"]) == 0
    out = capsys.readouterr().out
    assert "通算" in out and "F（券種混合" in out
    sc = pd.read_csv(raceday.raceday_dir() / "score_races.csv", encoding="utf-8-sig", dtype={"レースID": str})
    assert set(sc["レースID"]) == set(ids) and (sc["照合"] == "済").all()


def test_fetch_rt_results_with_fake_client(tmp_path, monkeypatch):
    from src import jvlink

    class FakeCom:
        def __init__(self):
            self.recs = [b"RA" + b"x" * 10, b"SE" + b"y" * 10, b"SE" + b"z" * 10]
            self.opened = []

        def JVInit(self, sid):
            return 0

        def JVRTOpen(self, spec, key):
            self.opened.append((spec, key))
            self.i = 0
            return 0

        def JVGets(self, buff, size, fname):
            if self.i >= len(self.recs):
                return 0, None, ""
            r = self.recs[self.i]
            self.i += 1
            return len(r), r, ""

        def JVClose(self):
            pass

    monkeypatch.setattr(jvlink, "parse_record", lambda raw, m: (raw[:2].decode(), {"a": raw[2:3].decode()}))
    monkeypatch.setattr(jvlink, "columns_for", lambda rt, m: ["a"])
    com = FakeCom()
    c = jvlink.fetch_rt_results("20261101", tmp_path, client=jvlink.JVLinkClient(com=com), struct_module=object())
    assert com.opened == [("0B12", "20261101")]
    assert c["レコード"]["RA"] == 1 and c["レコード"]["SE"] == 2 and not c["エラー"]
    assert pd.read_csv(tmp_path / "SE.csv")["a"].tolist() == ["y", "z"]


def test_prep_live_from_entries_and_update(env, monkeypatch, capsys):
    """出馬表（データ区分 2）から prep → update（取り込みは偽物）→ 仮・最終・ログ・タスクの定義。"""
    raw = {rt: pd.read_csv(env / f"{rt}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
           for rt in ("RA", "SE", "HR")}
    g = raw["RA"].loc[raw["RA"]["GradeCD"].str.strip().isin(["A", "B", "C", "D"])]
    last, year = g["id.MonthDay"].iloc[-1], g["id.Year"].iloc[-1]
    day = datetime.strptime(year + last, "%Y%m%d") + timedelta(days=7)
    date = f"{day:%Y%m%d}"
    ent = {}
    for rt in ("RA", "SE"):
        x = raw[rt].loc[(raw[rt]["id.MonthDay"] == last) & (raw[rt]["id.Year"] == year)].copy()
        x["id.MonthDay"] = f"{day:%m%d}"
        x["head.DataKubun"] = "2"
        x["id.Kaiji"] = "09"                         # レースIDに月日は入らないので、確定したレースと重ならないように
        if rt == "SE":
            for c in ("KakuteiJyuni", "Time", "Jyuni4c", "Jyuni3c", "Jyuni2c", "Jyuni1c", "HaronTimeL3"):
                if c in x.columns:
                    x[c] = ""
        ent[rt] = x
    for rt in ("RA", "SE"):
        pd.concat([raw[rt], ent[rt]], ignore_index=True).to_csv(env / f"{rt}.csv", index=False, encoding="utf-8-sig")
    assert raceday.main(["--data", str(env), "prep", "--date", date]) == 0
    out = capsys.readouterr().out
    assert "準備を保存しました" in out and "係数: β" in out
    prep = raceday.read_json(raceday.day_dir(date) / "prep.json")
    assert prep["レース"] and all(pd.isna(t["着順"]) for t in prep["対象"])
    assert sum(t["脚質スコア"] is not None for t in prep["対象"]) > 0       # 過去の4角の位置から
    xml = (raceday.day_dir(date) / "task.xml")
    assert raceday.main(["--data", str(env), "schedule", "--date", date]) == 0 and xml.exists()

    snap = _snap(n=10)
    monkeypatch.setattr(raceday, "LiveSource", lambda d, data=None: FakeSource(snap))
    post = raceday.post_time(date, prep["レース"][0]["発走時刻"])
    hhmm = f"{post - timedelta(minutes=20):%H%M}"
    assert raceday.main(["--data", str(env), "update", "--date", date, "--now", hhmm, "--no-sheet"]) == 0
    out = capsys.readouterr().out
    rid = prep["レース"][0]["レースID"]
    assert (raceday.day_dir(date) / rid / "final.json").exists()
    assert (raceday.day_dir(date) / rid / "provisional.json").exists()     # 最初の回が20分前でも仮も残す
    assert "最終" in out


def test_write_sheet_with_fake_worksheet(root):
    class WS:
        def __init__(self):
            self.cleared, self.values = False, None

        def clear(self):
            self.cleared = True

        def update(self, values, range_name, value_input_option):
            self.values = values
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))
    ws = WS()
    assert raceday.write_sheet(prep, ws=ws) == raceday.SHEET_TITLE
    assert ws.cleared and ws.values and len({len(r) for r in ws.values}) == 1     # 長方形にそろえて書く


# ---------------------------------------------------------------------------
# v10b：本物の fetch_rt_results の CSV（"_seq" の列が無い・同じ日に何回も追記）から当日の傾向まで
# ---------------------------------------------------------------------------
def _rt_scenario():
    """合成データの1日から：対象の重賞（その場の最後のレース）と、それより前のレースの 0B12 のレコード。"""
    raw = make_raw_v8(n_days=120, races=8)
    ds = jvmap.build_dataset(raw=raw)
    rr = ds.race_result
    rr = rr.assign(_start=pd.to_numeric(rr["発走時刻"].astype(str).str[:2]) * 60 + pd.to_numeric(rr["発走時刻"].astype(str).str[2:]))
    graded = rr.loc[rr["リステッド・重賞競走"].isin(["G1", "G2", "G3", "G"])].drop_duplicates("レースID")
    for _, t in graded.sort_values("レース日付", ascending=False).iterrows():
        same = rr.loc[(rr["レース日付"] == t["レース日付"]) & (rr["競馬場コード"] == t["競馬場コード"])
                      & (rr["芝・ダート区分"] == t["芝・ダート区分"]) & (rr["_start"] <= t["_start"] - 40)]
        if same["レースID"].nunique() >= 2:
            return raw, ds, t, sorted(same["レースID"].unique())
    raise AssertionError("合成データに使える日が無い")


def test_live_results_from_real_csv_format(root, monkeypatch):
    from src import jvlink
    raw, ds, target, earlier = _rt_scenario()
    date = f"{target['レース日付']:%Y%m%d}"
    rid_of = {rt: jvmap.make_race_id(raw[rt]) for rt in ("RA", "SE")}
    day = {rt: raw[rt].loc[(raw[rt]["id.Year"] + raw[rt]["id.MonthDay"]) == date] for rt in ("RA", "SE")}
    # 対象の重賞はまだ走っていない。1つ前の早いレースは速報の途中（データ区分 4 まで）
    partial = earlier[0]
    recs = []
    for kubun in ("3", "4", "5"):
        for rt in ("RA", "SE"):
            d = day[rt]
            ids = rid_of[rt].loc[d.index]
            for _, row in d.loc[(ids != str(target["レースID"])).to_numpy()].iterrows():
                if kubun == "5" and rid_of[rt].loc[_] == partial:
                    continue
                recs.append((rt, {**row.drop(labels=["_seq"]).to_dict(), "head.DataKubun": kubun}))

    class FakeCom:
        def JVInit(self, sid):
            return 0

        def JVRTOpen(self, spec, key):
            assert (spec, key) == ("0B12", date)
            self.i = 0
            return 0

        def JVGets(self, buff, size, fname):
            if self.i >= len(recs):
                return 0, None, ""
            b = str(self.i).encode()
            self.i += 1
            return len(b), b, ""

        def JVClose(self):
            pass

    monkeypatch.setattr(jvlink, "parse_record", lambda b, m: recs[int(b.decode())])
    cols = {rt: [c for c in raw[rt].columns if c != "_seq"] for rt in ("RA", "SE", "HR")}
    monkeypatch.setattr(jvlink, "columns_for", lambda rt, m: cols[rt])
    src = raceday.LiveSource(date, client_factory=lambda: jvlink.JVLinkClient(com=FakeCom()), struct_module=object())
    for _ in range(3):                                            # 10分おきに同じ日の全レコードが追記される
        got, ok, msg = src.fetch_results(datetime.strptime(date + "1200", "%Y%m%d%H%M"))
    se = pd.read_csv(src.rt_dir / "SE.csv", dtype=str, encoding="utf-8-sig")
    assert "_seq" not in se.columns and len(se) == 3 * len([r for r in recs if r[0] == "SE"])   # 本物と同じ形
    assert ok and "失敗" not in msg, msg
    done = set(got["レースID"])
    assert partial not in done and str(target["レースID"]) not in done
    assert set(earlier[1:]) <= done
    assert not got.duplicated(["レースID", "血統登録番号"]).any()        # 着順が確定したレースが1回ずつ
    assert (got["_SEデータ区分"] == "5").all()
    times = pd.read_csv(raceday.day_dir(date) / "result_times.csv", dtype=str, encoding="utf-8-sig")
    assert times["レースID"].is_unique and set(times["レースID"]) == done

    # その結果で、対象の重賞の F（当日の傾向あり）
    rr = ds.race_result
    tg = predict.blank_results(rr.loc[rr["レースID"] == target["レースID"]]).copy()
    tg["脚質スコア"] = np.linspace(0.1, 0.9, len(tg))
    snap = _snap(n=len(tg))
    snap["単勝"] = {int(p): v for p, v in zip(sorted(tg["馬番"].astype(int)), snap["単勝"].values())}
    for k in ("3連単", "3連複", "馬単", "馬連"):
        snap.pop(k)
    plan = raceday.compute_plan(tg, snap, COEFS, got, results_ok=True)
    assert plan["確率の出し方"] == "当日の傾向を使った"
    expect = sorted((f"{int(x[-2:])}R" for x in earlier[1:]), key=lambda s: int(s[:-1]))
    assert plan["使ったレース"] == expect                          # 速報の途中のレースは使わない

    # 確認のコマンドの中身
    prep = {"日付": date, "レース": [{"レースID": str(target["レースID"]), "発走時刻": str(target["発走時刻"]),
                                    "競馬場名": "x", "競走名": "y", "芝ダ": target["芝・ダート区分"]}],
            "対象": tg.to_dict("records")}
    table, usable = raceday.check_results(date, prep, src.rt_dir)
    assert set(table.loc[table["着順の確定"] == "済", "レースID"]) == done
    assert usable[0]["使えるレース（今の時点）"] == "・".join(expect)


# ---------------------------------------------------------------------------
# v10c：シートの組番が日付に化けない（USER_ENTERED で書くので、6-8 や 2-8-10 は先頭に ' を付けて文字にする）
# ---------------------------------------------------------------------------
import re  # noqa: E402

_DATE_LIKE = re.compile(r"^\s*\d{1,4}\s*[-/年]\s*\d{1,2}(\s*[-/月]\s*\d{1,4})?\s*日?\s*$")   # 日本の設定のシート。64.4 は数値


def _parsed_as_date(v) -> bool:
    """Google スプレッドシートが USER_ENTERED で日付と解釈しうる文字か（先頭が ' なら文字のまま）。"""
    return isinstance(v, str) and not v.startswith("'") and bool(_DATE_LIKE.match(v))


def test_combo_text_arrows_and_order():
    assert raceday._combo_text("馬連", (8, 6)) == "6-8"
    assert raceday._combo_text("3連複", (10, 2, 8)) == "2-8-10"
    assert raceday._combo_text("馬単", (10, 11)) == "10→11"
    assert raceday._combo_text("3連単", (3, 8, 2)) == "3→8→2"
    # 前の版で保存した組番（"3-8-2"）も、馬番から作り直して表示する
    assert raceday.bet_text({"券種": "3連単", "組番": "3-8-2", "馬番": [3, 8, 2]}) == "3→8→2"


def test_sheet_cells_are_never_dates(root):
    class WS:
        def clear(self):
            pass

        def update(self, values, range_name, value_input_option):
            self.values, self.option = values, value_input_option
    prep = _prep()
    src = FakeSource(_snap())
    raceday.run_once(prep, src, datetime(2026, 11, 1, 14, 50))       # 仮
    raceday.run_once(prep, FakeSource(_snap(seed=5)), datetime(2026, 11, 1, 15, 20))   # 最終（仮と違う買い目）
    ws = WS()
    raceday.write_sheet(prep, ws=ws)
    cells = [v for r in ws.values for v in r]
    assert not [v for v in cells if _parsed_as_date(v)]
    final = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "final.json")
    shown = {r[1] for r in ws.values if len(r) > 2 and r[0] in raceday.SMAPPY_ORDER}
    assert shown == {"'" + raceday.bet_text(b) for b in final["買い目"]}
    for b in final["買い目"]:
        assert ("→" in raceday.bet_text(b)) == (b["券種"] in ("馬単", "3連単"))
    # 金額は数値のまま（表示は変えない）
    amounts = [r[2] for r in ws.values if len(r) > 2 and r[0] in raceday.SMAPPY_ORDER]
    assert all(isinstance(a, int) for a in amounts)
    prov = [r for r in ws.values if str(r[0]).startswith("仮の買い目")]
    assert prov and prov[0][1].startswith("'")
    assert ws.option == "USER_ENTERED"
    # 日付と誤解される形が実際に出ていたこと（' を外すと日付に見える）も確かめておく
    assert any(_parsed_as_date(v[1:]) for v in cells if isinstance(v, str) and v.startswith("'"))


def test_sheet_command_rewrites_from_saved(root, monkeypatch):
    prep = _prep()
    raceday.write_json(raceday.day_dir(DATE) / "prep.json", prep)
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))
    before = (raceday.day_dir(DATE) / "202605040511" / "final.json").read_text(encoding="utf-8")
    got = {}
    monkeypatch.setattr(raceday, "write_sheet", lambda p, replay=False, **k: got.setdefault("p", p) and raceday.SHEET_TITLE)
    monkeypatch.setattr(raceday, "write_mobile", lambda p, replay=False, **k: got.setdefault("m", k) and raceday.MOBILE_SHEET_TITLE)
    assert raceday.main(["sheet", "--date", DATE]) == 0
    assert got["p"]["日付"] == DATE
    assert got["m"]["force"] and not got["m"]["collapse"]                    # 過去の日：全部書き直す・たたまない
    assert (raceday.day_dir(DATE) / "202605040511" / "final.json").read_text(encoding="utf-8") == before   # 作り直さない


# ---------------------------------------------------------------------------
# v11 1：本物のオッズで届かない買い目の、予算内での付け替え
# ---------------------------------------------------------------------------
def _rows(spec):
    return [{"券種": k, "組番": c, "馬番": [int(x) for x in c.replace("→", "-").split("-")], "金額": a,
             "推定配当": 0.0, "本物のオッズ": o} for k, c, a, o in spec]


def test_rebalance_tenno_sho_example():
    rows = _rows([("3連複", "1-11-16", 400, 130.0), ("3連複", "1-7-13", 3500, 25.4), ("3連複", "1-7-17", 7800, 6.3),
                  ("3連単", "17→7→3", 300, 200.0)])
    moved = raceday.rebalance(rows)
    by = {r["組番"]: r for r in rows}
    assert by["1-7-17"]["金額"] == 8000 and by["1-7-17"]["当たったときの払戻"] == pytest.approx(50_400)
    assert by["1-7-13"]["金額"] == 3300 and by["1-7-13"]["当たったときの払戻"] == pytest.approx(83_820)
    assert by["1-7-17"]["元の金額"] == 7800 and by["1-7-17"]["付け替え"] == 200 and by["1-7-13"]["付け替え"] == -200
    assert by["1-11-16"]["金額"] == 400 and by["17→7→3"]["金額"] == 300          # 減らすと 5万円を割る買い目からは回さない
    assert sum(r["金額"] for r in rows) == 12_000 and all(r["届く"] for r in rows)
    assert moved == {"点数": 2, "回した金額": 200}


def test_rebalance_partial_and_limits():
    rows = _rows([("馬連", "1-2", 9000, 5.0), ("馬連", "3-4", 2600, 20.0), ("馬単", "5→6", 400, None)])
    raceday.rebalance(rows)
    by = {r["組番"]: r for r in rows}
    # 3-4 は 2,500円 まで（20倍 × 2,500 = 50,000）しか下げられない → 100円だけ回せる。1-2 は 9,100円 でまだ届かない
    assert by["3-4"]["金額"] == 2500 and by["1-2"]["金額"] == 9100
    assert by["1-2"]["届く"] is False and by["1-2"]["足りない額"] == 900
    assert by["5→6"]["金額"] == 400 and by["5→6"]["届く"] is None                 # 本物のオッズが無い買い目は対象外
    assert sum(r["金額"] for r in rows) == 12_000
    rows = _rows([("馬連", "1-2", 11_900, 4.0), ("馬連", "3-4", 100, 900.0)])
    raceday.rebalance(rows)
    assert min(r["金額"] for r in rows) >= 100                                   # 100円未満にはしない
    assert raceday.stake_to_reach(12.5) == 4000 and raceday.stake_to_reach(6.3) == 8000


def test_plan_sheet_and_score_use_rebalanced_stakes(root):
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap(scale=0.8)), datetime(2026, 11, 1, 15, 20))
    f = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "final.json")
    assert f["合計"] == sum(b["金額"] for b in f["買い目"]) == sum(b["元の金額"] for b in f["買い目"])
    if f["付け替え"]["点数"]:
        text = "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(prep))
        assert "本物のオッズで付け替えた" in text and "元 " in text
    # 答え合わせは付け替えたあと（実際に買う方）。元の金額は参考
    b0 = f["買い目"][0]
    b0["元の金額"], b0["金額"] = b0["金額"] + 100, b0["金額"]
    wins = {k: {} for k in ("3連単", "3連複", "馬単", "馬連")}
    key = tuple(sorted(b0["馬番"])) if b0["券種"] in ("3連複", "馬連") else tuple(b0["馬番"])
    wins[b0["券種"]][key] = 1000.0
    s = raceday.score_plan(f, wins, set())
    s0 = raceday.score_plan(f, wins, set(), stake="元の金額")
    assert s["払戻"] == pytest.approx(b0["金額"] * 10) and s0["払戻"] == pytest.approx((b0["金額"] + 100) * 10)


def test_odds_drift_rows():
    plan = {"レースID": "R", "買い目": [{"券種": "馬連", "組番": "1-2", "馬番": [1, 2], "金額": 8000, "元の金額": 7800, "付け替え": 200,
                                       "本物のオッズ": 6.3, "届く": True}]}
    snap = _snap(n=4)
    horses, pay = snap["馬連"]
    pay = pay.copy()
    pay[(horses == np.array([1, 2])).all(axis=1)] = 600.0                          # 確定では 6.0倍
    snap["馬連"] = (horses, pay)
    d = raceday.odds_drift_rows(plan, snap)
    assert d[0]["確定オッズ"] == pytest.approx(6.0) and d[0]["確定で届く"] is False and d[0]["付け替えた"]
    assert d[0]["確定÷20分前"] == pytest.approx(6.0 / 6.3)


# ---------------------------------------------------------------------------
# v11 3：予備の買い目・表示・0B31・ドライランの代用・タスクの掃除・6頭BOX の記録
# ---------------------------------------------------------------------------
def test_backup_from_night_odds_and_box(root, monkeypatch):
    prep = _prep()
    made = raceday.make_backups(prep, source=FakeSource(_snap()), now=datetime(2026, 10, 31, 21, 0))
    bk = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "backup.json")
    assert "前の晩のオッズ" in made["202605040511"] and bk["買い目"]["点数"] <= 15
    # オッズが無ければ 6頭BOX の予想（predict の CSV）
    (raceday.day_dir(DATE) / "202605040511" / "backup.json").unlink()
    pdir = predict.prediction_dir()
    pdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"レースID": "202605040511", "馬番": range(1, 13), "予測順位": range(1, 13)}).to_csv(
        pdir / f"predict_{DATE}_210000.csv", index=False, encoding="utf-8-sig")
    made = raceday.make_backups(prep, source=FakeSource(None), now=datetime(2026, 10, 31, 21, 0))
    bk = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "backup.json")
    assert bk["BOX"] == [1, 2, 3, 4, 5, 6] and "6頭BOX" in made["202605040511"]


def test_sheet_shows_backup_only_when_no_plan(root):
    prep = _prep()
    raceday.make_backups(prep, source=FakeSource(_snap()), now=datetime(2026, 10, 31, 21, 0))

    def text(hhmm):
        c = raceday.Clock(datetime(2026, 11, 1), datetime(2026, 11, 1, int(hhmm[:2]), int(hhmm[2:])), None)
        return "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(prep, clock=c))
    assert "予備あり（最終の時刻までに作れなければ表示）" in text("1400") and "予備の買い目" not in text("1400")
    assert "当日の買い目は一度も作れなかった → 予備を表示" in text("1521") and "予備（前の晩のオッズ" in text("1521")
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))     # 当日の買い目ができれば予備は出さない
    assert "予備の買い目" not in text("1530")


def test_post_time_messages_match_state(root):
    prep = _prep()
    logs = raceday.run_once(prep, FakeSource(None), datetime(2026, 11, 1, 15, 45))
    assert logs[0]["結果の要約"] == "発走後。買い目なし（一度も作れなかった）"
    raceday.make_backups(prep, source=FakeSource(_snap()), now=datetime(2026, 10, 31, 21, 0))
    logs = raceday.run_once(prep, FakeSource(None), datetime(2026, 11, 1, 15, 45))
    assert "予備の買い目を表示" in logs[0]["結果の要約"]
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 0))
    logs = raceday.run_once(prep, FakeSource(None), datetime(2026, 11, 1, 15, 45))
    assert "15:00 の買い目が残っています" in logs[0]["結果の要約"]
    c = raceday.Clock(datetime(2026, 11, 1), datetime(2026, 11, 1, 15, 45), None)
    t = "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(_prep(), clock=c))
    assert "前回の買い目" not in t


def test_log_csv_columns_stay_aligned(root):
    p = raceday.day_dir(DATE) / "log.csv"
    raceday.append_csv(p, [{"a": 1, "b": "x", "c": 0.5}])
    raceday.append_csv(p, [{"a": 2, "c": 0.7}])                                   # 列が少ない行
    raceday.append_csv(p, [{"a": 3, "d": "new"}])                                 # 新しい列
    t = pd.read_csv(p, encoding="utf-8-sig")
    assert list(t.columns) == ["a", "b", "c", "d"]
    assert t["c"].tolist()[:2] == [0.5, 0.7] and pd.isna(t.loc[1, "b"]) and t.loc[2, "d"] == "new"


def test_0b31_fallback_when_0b30_fails(root, monkeypatch):
    from src import jvlink
    calls = []

    def fake(keys, dataspec, out_dir, date, client, log, **kw):
        calls.append(dataspec)
        if dataspec == "0B30":
            raise RuntimeError("0B30 だめ")
        return {"レース": 1, "データなし": 0, "エラー": 0, "レコード": 1, "保存ファイル": [], "取得時刻": ""}
    monkeypatch.setattr(jvlink, "fetch_rt_odds", fake)
    src = raceday.LiveSource(DATE)
    msg = src.fetch_odds(["2026110105040511"], datetime(2026, 11, 1, 15, 0))
    assert calls == ["0B30", "0B31"] and "予備 0B31" in msg
    monkeypatch.setattr(raceday.realodds, "load_race_odds", lambda *a, **k: _snap())
    snap = src.odds("202605040511", datetime(2026, 11, 1, 15, 0))
    assert "0B31" in snap["出どころ"] and "3連単" not in snap and snap["単勝"]
    plan = raceday.compute_plan(_race(), snap, COEFS, None, results_ok=False)
    assert not plan["本物のオッズで確かめた"]


def test_replay_uses_se_win_odds_when_o1_missing(root):
    raw = make_raw_v8(n_days=60, races=8)
    ds = jvmap.build_dataset(raw=raw)
    rr = ds.race_result
    day = pd.to_datetime(rr["レース日付"]).max()
    src = raceday.ReplaySource(ds, f"{day:%Y%m%d}")
    rid = str(rr.loc[pd.to_datetime(rr["レース日付"]) == day, "レースID"].iloc[0])
    snap = src.odds(rid, datetime(day.year, day.month, day.day, 12, 0))
    assert snap and "SE" in snap["出どころ"] and len(snap["単勝"]) >= 4


def test_front_bias_ignores_races_without_corners():
    """速報の版 5（着順だけ・通過順なし）のレースは、前残り傾向を NaN として除く（0 として数えない）。"""
    race = _race(hhmm="1540")
    with_c = _day_results(["1000"])
    no_c = _day_results(["1100"]).assign(**{"_4コーナー順位": np.nan, "レースID": "202605040502"})
    a, _ = raceday.day_terms(race, with_c)
    b, used = raceday.day_terms(race, pd.concat([with_c, no_c], ignore_index=True))
    assert used == ["1R", "2R"]
    assert np.allclose(a["当日_前残り傾向"], b["当日_前残り傾向"])               # 通過順の無いレースは前残りに入らない
    assert not np.allclose(a["当日_内枠傾向"], b["当日_内枠傾向"])               # 内枠（着順と馬番だけ）には入る


def test_cleanup_old_tasks():
    out = '"\\keiba_raceday_20261031","2026/10/31 9:00:00","準備完了"\n"\\keiba_raceday_20261101","N/A","準備完了"\n' \
          '"\\keiba_raceday_20261108","2026/11/08 9:00:00","準備完了"\n"\\Other","N/A","準備完了"\n'
    calls = []

    class R:
        def __init__(self, stdout=""):
            self.stdout, self.returncode = stdout, 0

    def run(cmd):
        calls.append(cmd)
        return R(out) if cmd[1] == "/query" else R()
    removed = raceday.cleanup_old_tasks("20261107", run=run)
    assert removed == ["keiba_raceday_20261031", "keiba_raceday_20261101"]
    assert ["schtasks", "/delete", "/tn", "keiba_raceday_20261108", "/f"] not in calls


def test_record_box_prediction_calls_predict_with_sheet():
    got = []
    assert raceday.record_box_prediction("20261101", run=lambda argv: got.append(argv) or 0) == 0
    assert got[0] == ["predict", "--date", "2026-11-01", "--sheet"]

    def boom(argv):
        raise SystemExit("シートの設定がありません")
    assert raceday.record_box_prediction("20261101", run=boom) == 1


# ---------------------------------------------------------------------------
# v12：付け替えの目標 m・20分前ごろのオッズのデータ集め・ずれの表
# ---------------------------------------------------------------------------
def test_rebalance_target_multiplier():
    spec = [("3連複", "1-7-13", 3500, 25.4), ("3連複", "1-7-17", 7800, 6.3), ("3連複", "1-11-16", 700, 130.0)]
    a, b = _rows(spec), _rows(spec)
    raceday.rebalance(a)
    raceday.rebalance(b, target=55_000)
    by_a, by_b = {r["組番"]: r for r in a}, {r["組番"]: r for r in b}
    assert by_a["1-7-17"]["金額"] == 8000 and by_b["1-7-17"]["金額"] == 8800        # 55,000 ÷ 6.3 → 8,800円
    assert by_b["1-11-16"]["金額"] * 130 >= 55_000 and sum(r["金額"] for r in b) == 12_000


def _write_snapshot(directory: Path, rid: str, date: str, scale: float, label: str, fetched: str):
    snap = _snap(n=6, scale=scale)
    w = odds.OddsWriter(directory, label=label)
    posts = np.array(sorted(snap["単勝"]))
    w.add(odds.OddsRecord("O1", rid, date, "1", 20261101, "11011520", posts.astype(np.int32),
                          np.round(np.array([snap["単勝"][p] for p in posts]) * 10).astype(np.int32),
                          np.ones(len(posts), np.int16)), fetched_at=fetched)
    for kind, rt in [("馬連", "O2"), ("馬単", "O4"), ("3連複", "O5"), ("3連単", "O6")]:
        horses, pay = snap[kind]
        r_ = horses.shape[1]
        kumi = sum(horses[:, i] * 100 ** (r_ - 1 - i) for i in range(r_))
        w.add(odds.OddsRecord(rt, rid, date, "1" if label != "final" else "5", 20261101, "", kumi.astype(np.int32),
                              np.round(pay / 10).astype(np.int32), np.ones(len(kumi), np.int16)), fetched_at=fetched)
    w.flush()


def test_collect_drift_odds_once_per_race_in_window(root):
    sch = pd.DataFrame({"レースID": ["202605040501", "202605040502", "202605040503"],
                        "キー": ["2026110105040501", "2026110105040502", "2026110105040503"],
                        "発走": ["1000", "1020", "1100"]})
    calls = []

    def fetch(keys):
        calls.append(list(keys))
        d = odds.odds_dir() / "rt" / DATE
        for k in keys:
            _write_snapshot(d, k[:4] + k[8:], DATE, 1.0, "0B30", "2026-11-01 09:40:00")
        return {"レコード": 5 * len(keys), "保存ファイル": [str(p) for p in d.glob("*.npz")]}
    import src.config as cfg
    old = cfg.JV_DATA_DIR
    cfg.JV_DATA_DIR = str(root / "jvlink")
    try:
        c = raceday.collect_drift_odds(DATE, datetime(2026, 11, 1, 9, 40), fetch=fetch, schedule=sch)   # 1R は 20分前・2R は 40分前
        assert calls == [["2026110105040501"]] and c["レース"] == 1
        c = raceday.collect_drift_odds(DATE, datetime(2026, 11, 1, 9, 42), fetch=fetch, schedule=sch)   # 1R は取った後
        assert c["レース"] == 0 and len(calls) == 1
        raceday.collect_drift_odds(DATE, datetime(2026, 11, 1, 9, 58), fetch=fetch, schedule=sch)      # 2R が 22分前
        assert calls[-1] == ["2026110105040502"]
        assert not list((odds.odds_dir() / "rt" / DATE).glob("*.npz"))                               # 重賞用の rt には残さない
        assert list(raceday.drift_dir(None, DATE).glob("O6_drift_*.npz"))
    finally:
        cfg.JV_DATA_DIR = old


def test_update_continues_when_drift_collection_fails(root, monkeypatch, capsys):
    prep = _prep()
    raceday.write_json(raceday.day_dir(DATE) / "prep.json", prep)
    monkeypatch.setattr(raceday, "LiveSource", lambda d, data=None: FakeSource(_snap()))

    def boom(*a, **k):
        raise RuntimeError("JV-Link だめ")
    monkeypatch.setattr(raceday, "collect_drift_odds", boom)
    assert raceday.main(["update", "--date", DATE, "--now", "1520", "--no-sheet"]) == 0
    out = capsys.readouterr().out
    assert "集められませんでした（買い目には影響なし）" in out
    assert (raceday.day_dir(DATE) / "202605040511" / "final.json").exists()
    assert (raceday.day_dir(DATE) / "drift_errors.csv").exists()


def test_drift_pairs_and_table(root):
    import src.config as cfg
    old = cfg.JV_DATA_DIR
    cfg.JV_DATA_DIR = str(root / "jvlink")
    try:
        rid = "202605040511"
        _write_snapshot(raceday.drift_dir(None, DATE), rid, DATE, 1.0, "drift", "2026-11-01 15:20:00")
        _write_snapshot(odds.odds_dir(), rid, DATE, 0.9, "final", "")                 # 確定は 20分前の 0.9倍
        pairs = raceday.drift_pairs()
        assert set(pairs["券種"]) == {"馬連", "馬単", "3連複", "3連単"}
        assert np.allclose(pairs["r"], 0.9, atol=0.02)
        assert not pairs["確定でも届く"].all()
        t = raceday.drift_table(pairs)
        assert (t["組の数"] > 0).all() and t["r の中央値（確定÷20分前）"].between(0.85, 0.95).all()
        assert raceday.main(["drift"]) == 0 and raceday.drift_samples_path().exists()
    finally:
        cfg.JV_DATA_DIR = old


# ---------------------------------------------------------------------------
# v13：熱さの参考表示（今の買い目で数える）・k と m の設定
# ---------------------------------------------------------------------------
def test_heat_is_monotone_and_from_current_bets():
    for seed in range(6):
        for scale in (0.5, 0.8, 1.0):
            plan = raceday.compute_plan(_race(), _snap(seed=seed, scale=scale), COEFS, None, results_ok=False)
            h = plan["熱さ"]
            assert h[50_000] <= h[40_000] + 1e-12 <= h[30_000] + 2e-12                 # 5万 ≤ 4万 ≤ 3万
            r = plan["熱さ（目標を下げて選び直した場合）"]
            assert r[50_000] <= r[40_000] + 1e-12 <= r[30_000] + 2e-12
            assert r[50_000] == pytest.approx(plan["予測の達成確率"])
            assert plan["例外を考えてもいい場面"] == (h[50_000] < raceday.EXCEPTION_RATE)


def test_heat_counts_rebalanced_stakes_with_real_odds():
    """本物のオッズが推定どおりで付け替えも無ければ、5万円の熱さは F の予測の達成確率と同じ。"""
    plan = raceday.compute_plan(_race(), _snap(seed=1, scale=1.4), COEFS, None, results_ok=False)
    assert plan["付け替え"]["点数"] == 0
    # 本物のオッズが推定の 1.4倍 → 推定配当 × 金額 ≥ 7万円 の買い目は本物で ≥ 9.8万円。5万円の熱さは推定のときより下がらない
    assert plan["熱さ"][50_000] >= plan["予測の達成確率"] - 1e-9


def test_k_and_m_settings():
    default = raceday.compute_plan(_race(), _snap(seed=2), COEFS, None, results_ok=False)
    assert (default["k"], default["m"]) == (1.4, 1.0)
    low = raceday.compute_plan(_race(), _snap(seed=2), {**COEFS, "k": 1.0}, None, results_ok=False)
    assert low["k"] == 1.0 and low["予測の達成確率"] >= default["予測の達成確率"] - 1e-9   # 余裕が小さいほど天井が上がる
    hi_m = raceday.compute_plan(_race(), _snap(seed=2, scale=0.75), {**COEFS, "m": 1.2}, None, results_ok=False)
    for b in hi_m["買い目"]:
        if b["付け替え"] > 0:
            assert b["本物のオッズ"] * b["金額"] >= 60_000 - 1e-6                       # 目標は 5万円 × 1.2


def test_freeze_stores_k_m_and_sheet_header(root):
    raceday.write_json(raceday.raceday_dir() / "coefs" / "coefs_20261101.json", dict(COEFS))
    assert raceday.main(["freeze", "--k", "1.2", "--m", "1.05"]) == 0
    fr = raceday.read_json(raceday.frozen_path())
    assert (fr["k"], fr["m"]) == (1.2, 1.05)
    c = raceday.load_or_fit_coefs(None, "20261108")
    assert raceday.plan_settings(c) == (1.2, 1.05)
    prep = _prep()
    prep["係数"] = c
    text = "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(prep))
    assert "k = 1.2・m = 1.05" in text
    assert raceday.main(["unfreeze"]) == 0
    assert "k = 1.4・m = 1.00" in "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(_prep()))


# ---------------------------------------------------------------------------
# v14：止まらない・眠らない・止まったら分かる
# ---------------------------------------------------------------------------
import time as _time  # noqa: E402


class StuckCom:
    """JVRTOpen が -413 を返し続ける／1回ごとに長く待たされる偽の JV-Link。"""

    def __init__(self, code=-413, delay=0.0):
        self.code, self.delay, self.opened = code, delay, []

    def JVInit(self, sid):
        return 0

    def JVRTOpen(self, spec, key):
        self.opened.append((spec, key))
        _time.sleep(self.delay)
        return self.code

    def JVGets(self, buff, size, fname):
        return 0, None, ""

    def JVClose(self):
        pass


@pytest.fixture()
def fast_limits(monkeypatch):
    monkeypatch.setattr(raceday, "RESULTS_TIMEOUT", 1.0)
    monkeypatch.setattr(raceday, "ODDS_TIMEOUT", 1.0)
    monkeypatch.setattr(raceday, "ODDS31_TIMEOUT", 1.0)
    monkeypatch.setattr(raceday, "JOIN_MARGIN", 0.3)


def _live(com, monkeypatch, fail_keys=True):
    from src import jvlink
    called = {"keys": 0}

    def keys_for_date(*a, **k):
        called["keys"] += 1
        return [(f"2026110105040{i:03d}", f"202605040{i:03d}") for i in range(36)]
    monkeypatch.setattr(jvlink, "race_keys_for_date", keys_for_date)
    monkeypatch.setattr(jvlink, "columns_for", lambda rt, m: ["a"])
    src = raceday.LiveSource(DATE, client_factory=lambda: jvlink.JVLinkClient(com=com), struct_module=object())
    return src, called


def test_0b12_413_stops_fast_and_falls_back_to_market(root, monkeypatch, fast_limits):
    com = StuckCom(-413)
    src, called = _live(com, monkeypatch)
    t0 = _time.time()
    day, ok, msg = src.fetch_results(datetime(2026, 11, 1, 14, 28))
    assert _time.time() - t0 < 3 and not ok and "-413" in msg
    assert called["keys"] == 0 and len(com.opened) == 1                       # 通信のエラーではレース単位の予備を試さない
    keys = [f"2026110105040{i:03d}" for i in range(36)]
    t0 = _time.time()
    msg = src.fetch_odds(keys, datetime(2026, 11, 1, 14, 28))
    assert _time.time() - t0 < 4 and "打ち切り" in msg
    assert len([o for o in com.opened if o[0] == "0B30"]) == raceday.MAX_SAME_ERRORS   # 同じエラーが続いたら残りは試さない
    assert not src.got_odds


def test_slow_jvlink_is_cut_at_the_limit(root, monkeypatch, fast_limits):
    com = StuckCom(0, delay=5.0)                                             # 1回の JVRTOpen が 5 秒戻らない
    src, _ = _live(com, monkeypatch)
    t0 = _time.time()
    day, ok, msg = src.fetch_results(datetime(2026, 11, 1, 14, 28))
    assert _time.time() - t0 < 2.5 and not ok and "打ち切り" in msg
    t0 = _time.time()
    src.fetch_odds(["2026110105040511"], datetime(2026, 11, 1, 14, 28))
    assert _time.time() - t0 < 4.5 and not src.got_odds


def test_final_is_made_even_when_jvlink_hangs(root, monkeypatch, fast_limits):
    """取り込みが戻らなくても、その回は先に進み、20分前の回で最終の買い目ができる（保存してある新しいオッズで・市場だけ）。"""
    prep = _prep()
    import src.config as cfg
    old = cfg.JV_DATA_DIR
    cfg.JV_DATA_DIR = str(root / "jvlink")
    try:
        _write_snapshot(odds.odds_dir() / "rt" / DATE, "202605040511", DATE, 1.0, "0B30", "2026-11-01 15:12:00")
        com = StuckCom(0, delay=5.0)
        src, _ = _live(com, monkeypatch)
        t0 = _time.time()
        logs = raceday.run_once(prep, src, datetime(2026, 11, 1, 15, 20))
        assert _time.time() - t0 < 6
        f = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "final.json")
        assert f and f["確率の出し方"].startswith("市場だけ") and "保存済み" not in logs[0]["結果の要約"]
    finally:
        cfg.JV_DATA_DIR = old


def test_stale_odds_are_not_used_when_fetch_failed(root, monkeypatch, fast_limits):
    prep = _prep()
    import src.config as cfg
    old = cfg.JV_DATA_DIR
    cfg.JV_DATA_DIR = str(root / "jvlink")
    try:
        _write_snapshot(odds.odds_dir() / "rt" / DATE, "202605040511", DATE, 1.0, "0B30", "2026-11-01 13:09:00")
        src, _ = _live(StuckCom(-413), monkeypatch)
        logs = raceday.run_once(prep, src, datetime(2026, 11, 1, 15, 20))
        assert "古い" in logs[0]["結果の要約"]
        assert not (raceday.day_dir(DATE) / "202605040511" / "final.json").exists()
    finally:
        cfg.JV_DATA_DIR = old


def test_network_down_skips_jvlink(root, monkeypatch):
    src, _ = _live(StuckCom(0), monkeypatch)
    src.network_ok = False
    day, ok, msg = src.fetch_results(datetime(2026, 11, 1, 14, 0))
    assert not ok and "ネットにつながっていない" in msg
    assert "ネットにつながっていない" in src.fetch_odds(["k"], datetime(2026, 11, 1, 14, 0))
    assert raceday.wait_for_network(probe=lambda: False, max_wait=20, step=10, sleep=lambda s: None) is False
    seq = iter([False, False, True])
    assert raceday.wait_for_network(probe=lambda: next(seq), max_wait=60, step=10, sleep=lambda s: None) is True


def test_jvlink_fetch_deadline_and_same_errors(tmp_path, monkeypatch):
    from src import jvlink
    monkeypatch.setattr(jvlink, "columns_for", lambda rt, m: ["a"])
    com = StuckCom(-413)
    c = jvlink.fetch_rt_results("20261101", tmp_path, client=jvlink.JVLinkClient(com=com), struct_module=object(),
                                keys=[f"k{i}" for i in range(36)], max_same_errors=3)
    assert len(com.opened) == 3 and "3 回続いた" in c["打ち切り"]
    com = StuckCom(-413)
    c = jvlink.fetch_rt_results("20261101", tmp_path, client=jvlink.JVLinkClient(com=com), struct_module=object(),
                                keys=[f"k{i}" for i in range(36)], deadline=_time.time() - 1)
    assert com.opened == [] and c["打ち切り"] == "時間の上限"
    com = StuckCom(-413)                                                     # 既定（rt-odds のコマンド）は今までどおり全部試す
    jvlink.fetch_rt_results("20261101", tmp_path, client=jvlink.JVLinkClient(com=com), struct_module=object(),
                            keys=[f"k{i}" for i in range(5)])
    assert len(com.opened) == 5


def test_watchdog_exits_on_wall_clock_jump():
    exited = []
    t = [1000.0]
    stop = raceday.start_watchdog(limit=300, clock=lambda: t[0], exit_fn=exited.append, interval=0.05, log=lambda *a: None)
    _time.sleep(0.15)
    assert not exited
    t[0] += 7240                                                             # スリープ明け：壁の時計が2時間進んだ
    _time.sleep(0.3)
    assert exited == [3]
    stop.set()
    exited2 = []
    stop2 = raceday.start_watchdog(limit=300, clock=lambda: 0.0, exit_fn=exited2.append, interval=0.05)
    stop2.set()                                                              # 正常に終わった回は止める
    _time.sleep(0.15)
    assert not exited2


def test_gap_is_noted_and_shown(root):
    assert raceday.note_gap(DATE, datetime(2026, 11, 1, 13, 10)) is None
    assert raceday.note_gap(DATE, datetime(2026, 11, 1, 13, 20)) is None
    msg = raceday.note_gap(DATE, datetime(2026, 11, 1, 14, 28))
    assert "13:20〜14:28" in msg and "止まっていた可能性" in msg
    text = "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(_prep()))
    assert "PC が止まっていた可能性 13:20〜14:28" in text


def test_sheet_has_minutes_since_update_formula_and_red_rule(root):
    rows = raceday.render_sheet(_prep(), clock=raceday.Clock(datetime(2026, 11, 1), datetime(2026, 11, 1, 14, 28, 32), None))
    assert rows[1][0].startswith('=IF(B2>15,"!!!!! 最終更新から "&B2&" 分') and "PC が止まっている可能性" in rows[1][0]
    assert rows[1][1] == "=ROUND(TO_PURE_NUMBER(NOW()-DATE(2026,11,1)-TIME(14,28,32))*1440)"   # 時刻の表示にならない数
    from src import sheets

    class Book:
        def __init__(self):
            self.rules, self.calls = [], 0

        def fetch_sheet_metadata(self):
            return {"sheets": [{"properties": {"sheetId": 7}, "conditionalFormats": self.rules}]}

        def batch_update(self, body):
            self.calls += 1
            self.rules.append(body["requests"][0]["addConditionalFormatRule"]["rule"])

    class WS:
        id = 7
        spreadsheet = Book()
    ws = WS()
    assert sheets.ensure_red_rule(ws, "A2:E2", "=$B$2>15") is True
    assert sheets.ensure_red_rule(ws, "A2:E2", "=$B$2>15") is False and ws.spreadsheet.calls == 1   # 毎回足さない
    rng = ws.spreadsheet.rules[0]["ranges"][0]
    assert (rng["startRowIndex"], rng["endRowIndex"], rng["startColumnIndex"], rng["endColumnIndex"]) == (1, 2, 0, 5)


def test_state_says_final_missing_after_t_minus_20(root):
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 14, 0))
    c = raceday.Clock(datetime(2026, 11, 1), datetime(2026, 11, 1, 15, 30), None)
    text = "\n".join(" ".join(map(str, r)) for r in raceday.render_sheet(prep, clock=c))
    assert "最終の買い目を作れていない。14:00 の買い目のまま" in text


def test_keepawake_spawned_once_until_last_post(root):
    prep = _prep()
    spawned = []
    assert raceday.ensure_keepawake(DATE, prep, datetime(2026, 11, 1, 9, 0), spawn=spawned.append) is True
    assert spawned[0][:3] == ["-m", "src.raceday", "keepawake"] and spawned[0][-1] == "1550"   # 最後の発走 15:40 の10分あと
    assert raceday.ensure_keepawake(DATE, prep, datetime(2026, 11, 1, 9, 10), spawn=spawned.append) is False   # 動いている
    raceday.keepawake_path(DATE).unlink()
    assert raceday.ensure_keepawake(DATE, prep, datetime(2026, 11, 1, 16, 0), spawn=spawned.append) is False   # 最後の発走のあと
    assert len(spawned) == 1


def test_score_counts_replay_separately(env, capsys):
    raw = {rt: pd.read_csv(env / f"{rt}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig") for rt in ("RA", "SE", "HR")}
    ds = jvmap.build_dataset(env)
    rr = ds.race_result
    graded = rr.loc[rr["リステッド・重賞競走"].isin(["G1", "G2", "G3", "G"])]
    day = pd.to_datetime(graded["レース日付"]).sort_values().iloc[-1]
    date = f"{day:%Y%m%d}"
    ids = graded.loc[pd.to_datetime(graded["レース日付"]) == day, "レースID"].unique()
    _write_odds(env, ds, ids, date)
    assert raceday.main(["--data", str(env), "replay", "--date", date]) == 0
    capsys.readouterr()
    assert raceday.main(["--data", str(env), "score"]) == 1                    # 本番の分が無く、ドライランも指定していない
    assert raceday.main(["--data", str(env), "score", "--replay-dates", date]) == 0
    out = capsys.readouterr().out
    assert raceday.REPLAY_LABEL in out and "本番の通算には入れない" in out
    sc = pd.read_csv(raceday.raceday_dir() / "score_races.csv", encoding="utf-8-sig", dtype={"レースID": str})
    assert set(sc["区分"]) == {raceday.REPLAY_LABEL}
    summ = pd.read_csv(raceday.raceday_dir() / "score_summary.csv", encoding="utf-8-sig")
    assert summ["買い方"].str.contains(raceday.REPLAY_LABEL).all()



# ---------------------------------------------------------------------------
# v15：シートの直し・ずれの表の区分・全レースの練習の記録
# ---------------------------------------------------------------------------
def test_gap_ignores_runs_before_the_day_starts(root):
    assert raceday.note_gap(DATE, datetime(2026, 10, 31, 17, 19)) is None     # 前の晩の試し
    assert raceday.note_gap(DATE, datetime(2026, 11, 1, 9, 0)) is None        # 当日の朝：前の晩からの間隔は数えない
    assert raceday.note_gap(DATE, datetime(2026, 11, 1, 9, 10)) is None
    assert raceday.note_gap(DATE, datetime(2026, 11, 1, 9, 40)) is not None   # 当日 9:00 以降の 30 分は数える
    assert not (raceday.read_json(raceday.day_dir(DATE) / "gaps.json")[0]["から"] == "17:19")


def test_ensure_timezone_sets_tokyo_once():
    from src import sheets

    class Book:
        def __init__(self, tz):
            self.tz, self.calls = tz, []

        def fetch_sheet_metadata(self):
            return {"properties": {"timeZone": self.tz}}

        def batch_update(self, body):
            self.calls.append(body)
            self.tz = body["requests"][0]["updateSpreadsheetProperties"]["properties"]["timeZone"]

    class WS:
        spreadsheet = Book("Etc/GMT")
    assert sheets.ensure_timezone(WS(), "Asia/Tokyo") is True and WS.spreadsheet.tz == "Asia/Tokyo"
    assert sheets.ensure_timezone(WS(), "Asia/Tokyo") is False and len(WS.spreadsheet.calls) == 1


def test_drift_table_split_by_grade():
    rng = np.random.default_rng(0)
    n = 400
    pairs = pd.DataFrame({"レースID": rng.choice(["A", "B", "C"], n), "券種": "馬連", "帯": "10〜50倍",
                          "r": np.exp(rng.normal(0, 0.1, n)), "確定でも届く": rng.random(n) < 0.6})
    pairs["グレード"] = pairs["レースID"].map({"A": "G1", "B": "G3", "C": ""})
    pairs["区分"] = np.where(pairs["グレード"] != "", "重賞", "平場")
    t = raceday.drift_table(pairs)
    assert list(dict.fromkeys(t["区分"])) == ["全部", "重賞", "平場", "G1"]
    by = t.set_index("区分")["組の数"]
    assert by["全部"] == n and by["重賞"] + by["平場"] == n and by["G1"] == (pairs["グレード"] == "G1").sum()


def _practice_env(env):
    import shutil
    ds = jvmap.build_dataset(env)
    rr = ds.race_result
    day = pd.to_datetime(rr["レース日付"]).sort_values().iloc[-1]
    date = f"{day:%Y%m%d}"
    ids = rr.loc[pd.to_datetime(rr["レース日付"]) == day, "レースID"].astype(str).unique()
    have = ids[:-2]                                                          # 2 レースは 20分前のオッズが無い
    tmp = env.parent / "tmp_odds"
    _write_odds(tmp, ds, have, date)
    dst = raceday.drift_dir(env, date)
    dst.mkdir(parents=True, exist_ok=True)
    for f in odds.odds_dir(tmp).glob("*.npz"):
        shutil.move(str(f), dst / f.name.replace("_test_", "_drift_"))
    return ds, date, ids, have


def test_practice_records_all_races_without_double_counting(env, capsys):
    ds, date, ids, have = _practice_env(env)
    assert raceday.main(["--data", str(env), "practice", "--date", date]) == 0
    out = capsys.readouterr().out
    assert out.startswith("※ 記述だけ。F は変えない") and "目安の線" in out and "9%" in out
    assert "オッズが無くて飛ばしたレース: 2" in out
    rec = pd.read_csv(raceday.practice_dir() / "records.csv", dtype={"レースID": str}, encoding="utf-8-sig")
    assert set(rec["レースID"]) <= set(have) and len(rec) > 0 and rec["レースID"].is_unique
    assert (rec["点数"] <= raceday.MAX_POINTS).all() and (rec["投資"] <= raceday.BUDGET).all()
    assert set(rec["区分"]) <= {"平場", "重賞（参考）"}
    assert (rec["当たったのに届かず"] == (rec["的中"] & ~rec["熱い"])).all()
    n = len(rec)
    assert raceday.main(["--data", str(env), "practice", "--date", f"{date},{date}"]) == 0       # 2回回しても二重に数えない
    rec2 = pd.read_csv(raceday.practice_dir() / "records.csv", dtype={"レースID": str}, encoding="utf-8-sig")
    assert len(rec2) == n
    summ = pd.read_csv(raceday.practice_dir() / "summary.csv", encoding="utf-8-sig")
    flat = summ.loc[summ["区分"] == "平場"].iloc[0]
    assert flat["レース数"] == (rec2["区分"] == "平場").sum() and flat["検証（F）"] == pytest.approx(0.1191)


def test_practice_uses_prep_coefficients_when_present(env, capsys):
    ds, date, ids, have = _practice_env(env)
    coefs = dict(COEFS, 期間=["2011-01-01", "x"], 推定レース数=1, k=1.2)
    raceday.write_json(raceday.day_dir(date) / "prep.json", {"日付": date, "係数": coefs, "対象": [], "レース": []})
    assert raceday.main(["--data", str(env), "practice", "--date", date]) == 0
    assert "係数: その日の prep.json・k = 1.2・m = 1.00" in capsys.readouterr().out
    rec = pd.read_csv(raceday.practice_dir() / "records.csv", dtype={"レースID": str}, encoding="utf-8-sig")
    assert (rec["k"] == 1.2).all()


def test_parse_dates():
    assert raceday.parse_dates("20261107-20261109,20261115") == ["20261107", "20261108", "20261109", "20261115"]


# ---------------------------------------------------------------------------
# v16：スマホ用タブ「スマホ」と「成績」
# ---------------------------------------------------------------------------
class _FakeBook:
    def __init__(self):
        self.requests, self.meta_reads = [], 0

    def batch_update(self, body):
        self.requests.append(body["requests"])

    def fetch_sheet_metadata(self):
        self.meta_reads += 1
        return {"sheets": [{"properties": {"sheetId": 7}, "conditionalFormats": []}], "properties": {"timeZone": "Asia/Tokyo"}}


class _FakeWS:
    """値の書き込み（batch_update）と書式（spreadsheet.batch_update）を記録する。"""
    def __init__(self):
        self.id, self.title, self.spreadsheet = 7, "スマホ", _FakeBook()
        self.value_calls, self.cells = [], {}

    def batch_update(self, data, value_input_option):
        assert value_input_option == "USER_ENTERED"
        self.value_calls.append(data)
        for d in data:
            for (r, c), v in _cells_of(d["range"], d["values"]).items():
                self.cells[(r, c)] = v

    def clear(self):
        self.cells = {}

    def update(self, values, range_name, value_input_option):
        self.batch_update([{"range": range_name, "values": values}], value_input_option)


def _cells_of(rng: str, values) -> dict:
    import re as _re
    m = _re.match(r"([A-Z]+)(\d+)", rng)
    c0, r0 = ord(m[1]) - 65, int(m[2])
    return {(r0 + i, c0 + j): v for i, row in enumerate(values) for j, v in enumerate(row)}


def _touched_rows(ws, call) -> set:
    return {r for d in call for (r, c) in _cells_of(d["range"], d["values"])}


def _final_then(prep, now):
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))        # 最終
    return raceday.mobile_views(prep, False, now)


def test_mobile_tab_has_three_columns_and_text_combos(root):
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 14, 50))       # 仮
    ws = _FakeWS()
    assert raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 14, 55)) == raceday.MOBILE_SHEET_TITLE
    assert max(c for (r, c) in ws.cells) < 3                                          # A〜C の3列だけ
    for reqs in ws.spreadsheet.requests:
        for q in reqs:
            for v in q.values():
                rng = v.get("range") if isinstance(v, dict) else None
                if rng and "endColumnIndex" in rng:
                    assert rng["endColumnIndex"] <= 3
    props = [q["updateSheetProperties"] for q in ws.spreadsheet.requests[0] if "updateSheetProperties" in q]
    assert props and props[0]["properties"]["gridProperties"]["columnCount"] == 3
    plan = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "latest.json")
    bet_cells = [v for (r, c), v in ws.cells.items() if c == 1 and isinstance(v, str) and v[1:].split(" ")[0] in raceday.SMAPPY_ORDER]
    assert bet_cells == ["'" + f"{b['券種']} {raceday.bet_text(b)}" for b in plan["買い目"]]  # 入れる順のまま・文字として
    assert not [v for v in ws.cells.values() if _parsed_as_date(v)]
    amounts = [ws.cells[(r, 2)] for (r, c), v in ws.cells.items() if c == 1 and v in bet_cells]
    assert amounts == [b["金額"] for b in plan["買い目"]]                               # 付け替えたあとの金額
    texts = " ".join(str(v) for v in ws.cells.values())
    assert "仮。最終は 15:20" in texts and "最終更新から" in texts
    assert ws.cells[(2, 2)].startswith("=ROUND(")                   # C2：最終更新から○分（シートで計算）
    assert len(ws.value_calls) == 1 and len(ws.spreadsheet.requests) == 1             # 値1回・書式1回にまとめる


def test_mobile_states(root):
    prep = _prep()
    rid = "202605040511"
    at = lambda hm: datetime(2026, 11, 1, int(hm[:2]), int(hm[2:]))  # noqa: E731
    r = prep["レース"][0]
    v = raceday.mobile_view(r, DATE, {}, None, at("1300"))
    assert v["種類"] == "待ち" and not v["買い目"]
    v = raceday.mobile_view(r, DATE, {}, None, at("1450"))
    assert v["種類"] == "なし"
    bk = {"作り方": "6頭BOX", "BOX": [1, 2, 3, 4, 5, 6], "作成": "2026-10-31 21:00:00"}
    v = raceday.mobile_view(r, DATE, {}, bk, at("1525"))
    assert v["種類"] == "予備" and v["買い目"][0] == ("3連単 BOX 1・2・3・4・5・6", 12_000) and "予備" in v["状態"]
    raceday.run_once(prep, FakeSource(_snap()), at("1400"))
    st = {s: raceday.read_json(raceday.day_dir(DATE) / rid / f"{s}.json") for s in raceday.STAGES}
    assert raceday.mobile_view(r, DATE, st, None, at("1405"))["種類"] == "途中"
    v = raceday.mobile_view(r, DATE, st, None, at("1525"), ["PC の時計が 3 分ずれている"])
    assert v["種類"] == "なし" and "最終を作れていない" in v["状態"] and v["警告"][0].startswith("時計・ネット")
    raceday.run_once(prep, FakeSource(_snap(scale=0.5)), at("1450"))
    st = {s: raceday.read_json(raceday.day_dir(DATE) / rid / f"{s}.json") for s in raceday.STAGES}
    v = raceday.mobile_view(r, DATE, st, None, at("1455"))
    assert v["種類"] == "仮" and any("5万円に届かない" for w in v["警告"])
    raceday.run_once(prep, FakeSource(_snap()), at("1520"))
    st = {s: raceday.read_json(raceday.day_dir(DATE) / rid / f"{s}.json") for s in raceday.STAGES}
    v = raceday.mobile_view(r, DATE, st, None, at("1521"))
    assert v["種類"] == "最終" and v["状態"] == "最終 15:20 作成。この通りに買う" and not v["済"]
    assert v["合計"] == (f"合計 {st['final']['点数']}点", st["final"]["合計"])
    assert raceday.mobile_view(r, DATE, st, None, at("1541"))["済"]


def test_mobile_final_block_is_not_rewritten(root):
    """最終を一度書いた枠は、その後の回では値も書式も書かない（チェックが消えない）。発走後は行を隠すだけ。"""
    prep = _prep()
    prep["レース"].append({"レースID": "202605040512", "発走時刻": "1625", "競馬場名": "東京", "競走名": "あとのレース",
                          "グレード": "G3", "芝ダ": "芝", "距離": 1600, "頭数": 12})
    prep["対象"] += _race(rid="202605040512", hhmm="1625").to_dict("records")
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))        # 1つめは最終・2つめは途中
    ws = _FakeWS()
    raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 15, 21))
    top, blk = raceday.MOBILE_TOP, raceday.MOBILE_BLOCK
    first = range(top + 1, top + blk + 1)                                             # 1つめの枠（シートの行番号）
    final = raceday.read_json(raceday.day_dir(DATE) / "202605040511" / "final.json")
    checks = [r for r in first if ws.cells.get((r, 0)) is False]
    assert len(checks) == final["点数"]                                                # 買い目の行ごとにチェック欄
    state = raceday.read_json(raceday._mobile_state_path())["スマホ"]
    assert state["最終を書いた"] == ["202605040511"]
    for r in checks:                                                                  # 人がチェックを付けた
        ws.cells[(r, 0)] = True
    raceday.run_once(prep, FakeSource(_snap(seed=3)), datetime(2026, 11, 1, 15, 30))
    raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 15, 30))
    assert not (_touched_rows(ws, ws.value_calls[-1]) & set(first))                   # 1つめの枠には書かない
    assert all(ws.cells[(r, 0)] is True for r in checks)
    second = range(top + blk + 1, top + 2 * blk + 1)
    assert _touched_rows(ws, ws.value_calls[-1]) & set(second)                         # 2つめ（途中）は書き直す

    def block_reqs(reqs, lo, hi):
        out = []
        for q in reqs:
            for v in q.values():
                g = v.get("range") if isinstance(v, dict) else None
                if g and "startRowIndex" in g and g["startRowIndex"] < hi and g["endRowIndex"] > lo:
                    out.append(q)
                if g and g.get("dimension") == "ROWS" and g["startIndex"] < hi and g["endIndex"] > lo:
                    out.append(q)
        return out
    assert not block_reqs(ws.spreadsheet.requests[-1], top, top + blk)                 # 書式も触らない
    # 発走後：題の行の文字と「行を隠す」だけ。チェックの行の値は書かない
    raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 15, 45))
    assert _touched_rows(ws, ws.value_calls[-1]) & set(first) == {top + 1}
    assert ws.cells[(top + 1, 0)].startswith("済 ")
    hidden = [q for q in block_reqs(ws.spreadsheet.requests[-1], top, top + blk)]
    assert hidden and all(q["updateDimensionProperties"]["properties"] == {"hiddenByUser": True} for q in hidden)
    assert all(ws.cells[(r, 0)] is True for r in checks)
    # sheet コマンド（force）は書き直す（チェックも消える）
    raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 15, 45), force=True, collapse=False)
    assert all(ws.cells[(r, 0)] is False for r in checks)


def test_mobile_resets_on_new_date(root):
    prep = _prep()
    raceday.run_once(prep, FakeSource(_snap()), datetime(2026, 11, 1, 15, 20))
    ws = _FakeWS()
    raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 15, 21))
    assert ws.spreadsheet.meta_reads == 1                                              # 最初の回だけ赤の書式の確認
    raceday.write_mobile(prep, ws=ws, now=datetime(2026, 11, 1, 15, 31))
    assert ws.spreadsheet.meta_reads == 1
    nxt = dict(prep, 日付="20261102")
    raceday.write_mobile(nxt, ws=ws, now=datetime(2026, 11, 2, 9, 0))
    assert any("updateCells" in q for q in ws.spreadsheet.requests[-1])                # 日付が変わったら全体を消して書き直す
    assert raceday.read_json(raceday._mobile_state_path())["スマホ"]["最終を書いた"] == []


def test_update_saves_plans_even_if_sheet_fails(root, monkeypatch, capsys):
    prep = _prep()
    raceday.write_json(raceday.day_dir(DATE) / "prep.json", prep)
    monkeypatch.setattr(raceday, "LiveSource", lambda d, data=None: FakeSource(_snap()))

    def boom(*a, **k):
        raise RuntimeError("シートの API が落ちた")
    monkeypatch.setattr(raceday, "write_sheet", boom)
    monkeypatch.setattr(raceday, "write_mobile", boom)
    assert raceday.main(["update", "--date", DATE, "--now", "1520", "--no-drift"]) == 0
    out = capsys.readouterr().out
    assert (raceday.day_dir(DATE) / "202605040511" / "final.json").exists()
    assert "スマホ用タブに書けませんでした（保存した買い目は残っています）" in out


def test_mobile_sample_writes_every_state(root):
    ws = _FakeWS()
    assert raceday.write_mobile_sample(ws=ws) == raceday.MOBILE_SAMPLE_SHEET_TITLE
    texts = " ".join(str(v) for v in ws.cells.values())
    for s in ("この通りに買う", "仮。最終は", "途中（", "予備。当日のオッズが取れていない", "最終を作れていない", "まだ買い目なし", "済 "):
        assert s in texts, s
    assert max(c for (r, c) in ws.cells) < 3 and not [v for v in ws.cells.values() if _parsed_as_date(v)]
    colors = [q["repeatCell"]["cell"]["userEnteredFormat"].get("backgroundColor") for q in ws.spreadsheet.requests[0]
              if "repeatCell" in q and q["repeatCell"]["cell"]["userEnteredFormat"].get("textFormat", {}).get("fontSize") == 18]
    assert len({tuple(c.values()) for c in colors}) == 5                               # 緑・黄・灰・橙・赤


def test_results_tab(root):
    summ = pd.DataFrame([{"買い方": "F（券種混合・市場＋3b・余裕×1.4・15点まで）", "レース数": 12, "熱い当たり": 2, "達成率": 2 / 12,
                          "予測の達成確率（平均）": 0.18},
                         {"買い方": "（参考）F の付け替える前の金額", "レース数": 12, "熱い当たり": 2, "達成率": 2 / 12},
                         {"買い方": "今の6頭BOX（predict --sheet の最終の予測。同じレース）", "レース数": 12, "熱い当たり": 1,
                          "達成率": 1 / 12}])
    races = pd.DataFrame([{"区分": "本番（自動実行）", "レースID": f"20260504051{i}", "日付": f"2026-11-0{i}", "レース": f"東京1{i}R 記念{i}",
                           "照合": "済", "熱い": i == 2, "的中": i >= 2, "払戻": 64_000.0 if i == 2 else 3_000.0 * i}
                          for i in range(1, 4)])
    pr = pd.DataFrame([{"区分": "平場", "レース数": 200, "熱い当たり": 30, "熱い当たりの割合": 0.15, "予測の達成確率（平均）": 0.17},
                       {"区分": "重賞（参考）", "レース数": 10, "熱い当たり": 1, "熱い当たりの割合": 0.1, "予測の達成確率（平均）": 0.2}])
    rows = raceday.render_results(summ, races, pr, datetime(2026, 11, 16, 9, 0))
    assert all(len(r) == 3 for r in rows)
    labels = [r[0] for r in rows]
    assert labels[2:6] == ["F 本番（重賞）", "6頭BOX（同じレース）", "練習 平場", "練習 重賞（参考）"]
    assert rows[2][1] == "'2 / 12" and rows[2][2] == "'16.7%（18.0%）" and rows[3][2] == "'8.3%（—）"
    recent = rows[rows.index(["最近の重賞（本番）", "熱い？", "払戻"]) + 1:]
    assert [r[1] for r in recent] == ["外れ", "熱い", "当たり（5万円に届かず）"][::-1] and recent[1][2] == 64_000
    assert not [v for r in rows for v in r if _parsed_as_date(v)]
    # 保存した CSV から書く
    summ.to_csv(raceday.raceday_dir() / "score_summary.csv", index=False, encoding="utf-8-sig")
    races.to_csv(raceday.raceday_dir() / "score_races.csv", index=False, encoding="utf-8-sig")
    ws = _FakeWS()
    assert raceday.write_results(ws=ws) == raceday.RESULTS_SHEET_TITLE
    assert ws.cells[(3, 0)] == "F 本番（重賞）" and max(c for (r, c) in ws.cells) < 3
