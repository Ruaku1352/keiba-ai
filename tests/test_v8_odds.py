"""v8 4：オッズ（O1〜O6）の読み取り・保存・取得のテスト（偽物の JV-Link。SDK があれば構造体とも突き合わせる）。"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, jvlink, odds  # noqa: E402
from test_v5_jvlink import FakeJVLink  # noqa: E402


# ===========================================================================
# 仕様書の位置どおりのダミーのレコード
# ===========================================================================
def make_odds_record(rt: str, year="2026", md="1101", jyo="05", kai="04", nichi="03", rno="11",
                     kubun="4", make="20261102", announced="00000000", n_horses=8, seed=0,
                     special: dict | None = None) -> bytes:
    """O1〜O6 のレコードを、仕様書の位置で作る。組番は昇順、オッズは決まった式で入れる。"""
    kind, length, start, count, size, kw, ow, (np_pos, nw) = odds.LAYOUT[rt]
    buf = bytearray(b" " * length)
    head = f"{rt}{kubun}{make}{year}{md}{jyo}{kai}{nichi}{rno}{announced}{n_horses:02d}{n_horses:02d}7"
    buf[:len(head)] = head.encode("ascii")
    width = kw // 2
    if rt == "O1":
        combos = [(h,) for h in range(1, n_horses + 1)]
    elif rt in ("O2", "O3"):
        combos = list(itertools.combinations(range(1, n_horses + 1), 2))
    elif rt == "O4":
        combos = list(itertools.permutations(range(1, n_horses + 1), 2))
    elif rt == "O5":
        combos = list(itertools.combinations(range(1, n_horses + 1), 3))
    else:
        combos = list(itertools.permutations(range(1, n_horses + 1), 3))
    for i, c in enumerate(combos):
        pos = start - 1 + i * size
        kumi = "".join(f"{x:02d}" for x in c)
        o = special.get(c) if special and c in special else f"{(i + 1) * 13 + 11:0{ow}d}"[-ow:]
        entry = kumi + o
        buf[pos:pos + len(entry)] = entry.encode("ascii")
        buf[pos + np_pos - 1:pos + np_pos - 1 + nw] = f"{i + 1:0{nw}d}".encode("ascii")
    buf[-2:] = b"\r\n"
    return bytes(buf)


def test_parse_each_type():
    for rt in odds.ODDS_TYPES:
        raw = make_odds_record(rt, n_horses=6)
        rec = odds.parse_odds(raw)
        assert rec.record_type == rt and rec.race_id == "202605040311" and rec.date == "20261101"
        assert rec.kubun == "4" and rec.make_date == 20261102
        n = {"O1": 6, "O2": 15, "O3": 15, "O4": 30, "O5": 20, "O6": 120}[rt]
        assert len(rec.kumi) == n
        assert rec.odds10[0] == 24 and rec.ninki[0] == 1         # (0+1)*13+11 = 24 → 2.4倍
        first = odds.kumi_tuple(int(rec.kumi[0]), rt)
        assert first == {"O1": (1,), "O2": (1, 2), "O3": (1, 2), "O4": (1, 2), "O5": (1, 2, 3),
                         "O6": (1, 2, 3)}[rt]


def test_special_odds_values_are_missing():
    raw = make_odds_record("O6", n_horses=4, special={(1, 2, 3): "-------", (1, 2, 4): "*******",
                                                      (1, 3, 2): "0000000"})
    rec = odds.parse_odds(raw)
    lookup = {odds.kumi_tuple(int(k), "O6"): o for k, o in zip(rec.kumi, rec.odds10)}
    assert lookup[(1, 2, 3)] == -1 and lookup[(1, 2, 4)] == -1 and lookup[(1, 3, 2)] == -1
    assert lookup[(1, 3, 4)] > 0


def test_parse_rejects_short_and_other_records():
    assert odds.parse_odds(b"RA" + b" " * 100) is None
    assert odds.parse_odds(make_odds_record("O6")[:5000]) is None


def test_writer_and_loader_pick_final_and_latest(tmp_path):
    d = tmp_path / "odds"
    w = odds.OddsWriter(d, kubun_filter=None)
    w.add(odds.parse_odds(make_odds_record("O6", kubun="1", make="20261101", announced="11011430")))
    w.add(odds.parse_odds(make_odds_record("O6", kubun="4", make="20261101")))
    w.add(odds.parse_odds(make_odds_record("O6", kubun="5", make="20261102", rno="10")))
    w.add(odds.parse_odds(make_odds_record("O6", kubun="4", make="20261102", rno="10", seed=1)))
    w.flush()
    final = odds.load_odds("O6", directory=d)
    assert set(final["レースID"]) == {"202605040311", "202605040310"}
    assert final.groupby("レースID")["データ区分"].first().to_dict() == {"202605040310": "5", "202605040311": "4"}
    assert len(final) == 2 * 336 and final["オッズ"].iloc[0] == pytest.approx(2.4)
    allk = odds.load_odds("O6", directory=d, final_only=False, latest=False)
    assert len(allk) == 4 * 336
    only = odds.load_odds("O6", directory=d, race_ids=["202605040310"])
    assert set(only["レースID"]) == {"202605040310"}


def test_writer_filters_races_and_kubun(tmp_path):
    w = odds.OddsWriter(tmp_path / "odds", race_filter={"202605040311"}, kubun_filter={"4", "5"})
    assert w.add(odds.parse_odds(make_odds_record("O1")))
    assert not w.add(odds.parse_odds(make_odds_record("O1", rno="10")))
    assert not w.add(odds.parse_odds(make_odds_record("O1", kubun="1")))
    assert w.kept == 1 and w.dropped == 2


# ===========================================================================
# 取得（偽物の JV-Link）
# ===========================================================================
class TruncatingFake(FakeJVLink):
    """本物と同じく、バッファより長いレコードは切れて返る偽物。"""

    def JVGets(self, buff, size, name):
        code, mv, fname = super().JVGets(buff, size, name)
        if code > 0 and code > size - 1:
            data = bytes(mv)[:size - 1]
            return len(data), memoryview(data), fname
        return code, mv, fname

    def JVRTOpen(self, spec, key):
        self.calls.append(("JVRTOpen", spec, key))
        self.file_index = 0
        self.script = list(self.rt_scripts.get(key, [])) if hasattr(self, "rt_scripts") else []
        return 0 if self.script else -1


def _ra_record_bytes():
    return b"RA7" + b"0" * 1269          # 中身は使わない（先頭2バイトで読み飛ばす）


def test_fetch_odds_reads_only_odds_with_adaptive_buffer(tmp_path):
    script = [_ra_record_bytes(), _ra_record_bytes(), -1,
              make_odds_record("O6", n_horses=10), make_odds_record("O6", n_horses=10, rno="10"), -1,
              make_odds_record("O1"), make_odds_record("O1", rno="10"), -1,
              make_odds_record("O3"), -1,
              make_odds_record("O6", n_horses=10, rno="09"), -1, 0]
    fake = TruncatingFake(script, open_ret=(0, 5, 0, "20261102140000"))
    res = jvlink.fetch_odds(fromtime="20261028000000", out_dir=tmp_path, client=jvlink.JVLinkClient(com=fake),
                            log=lambda *a: None, sleep=lambda s: None)
    assert res.truncated == 0
    assert res.records == {"O6": 3, "O1": 2}
    assert res.skipped_files == 2           # RA と O3（ワイドは既定で保存しない）
    o6 = odds.load_odds("O6", base=tmp_path)
    assert o6["レースID"].nunique() == 3 and len(o6) == 3 * 720
    assert odds.load_odds("O1", base=tmp_path)["レースID"].nunique() == 2
    big = jvlink._odds_buffer()
    # ファイルの先頭は大きいバッファ、2件目以降はその種別の大きさ（O1 は約1KB）
    assert fake.sizes[0] == big
    assert jvlink._odds_buffer_for("O1") in fake.sizes and jvlink._odds_buffer_for("O1") < 2000
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["ODDS"]["last_file_timestamp"] == "20261102140000"
    assert "RACE" not in state                # いつもの fetch の記録には触らない


def test_fetch_odds_default_fromtime_and_targets(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"RACE": {"fromtime": "20261027000000",
                                                              "last_file_timestamp": "20261102"}}),
                                         encoding="utf-8")
    script = [make_odds_record("O6", rno="11"), make_odds_record("O6", rno="10"), -1, 0]
    fake = TruncatingFake(script)
    res = jvlink.fetch_odds(out_dir=tmp_path, client=jvlink.JVLinkClient(com=fake), targets={"202605040310"},
                            log=lambda *a: None)
    assert ("JVOpen", "RACE", "20261027000000", 1) in fake.calls      # 初回は前回の fetch の開始時刻から
    assert res.saved == 1 and res.dropped == 1
    fake2 = TruncatingFake([0])
    jvlink.fetch_odds(out_dir=tmp_path, client=jvlink.JVLinkClient(com=fake2), log=lambda *a: None)
    assert fake2.calls[1][2] == "20261008120000"                       # 2回目は前回の fetch-odds の続き


def test_fetch_odds_resumes_after_interrupt(tmp_path):
    script = [make_odds_record("O6", rno="11"), -1, make_odds_record("O6", rno="10"), -1, 0]
    fake = TruncatingFake(script)
    calls = {"n": 0}

    def boom(f):
        calls["n"] += 1
        if calls["n"] == 3:          # 2つ目のファイルの途中で止める
            raise KeyboardInterrupt
    fake.delay_fn = boom
    with pytest.raises(KeyboardInterrupt):
        jvlink.fetch_odds(fromtime="20261028000000", out_dir=tmp_path, client=jvlink.JVLinkClient(com=fake),
                          log=lambda *a: None, flush_every=1)
    assert odds.load_odds("O6", base=tmp_path)["レースID"].nunique() == 1
    fake2 = TruncatingFake(script)
    jvlink.fetch_odds(fromtime="20261028000000", out_dir=tmp_path, client=jvlink.JVLinkClient(com=fake2),
                      log=lambda *a: None)
    assert ("JVSkip",) in fake2.calls                                  # 読み終えたファイルは飛ばす
    assert odds.load_odds("O6", base=tmp_path)["レースID"].nunique() == 2


def test_rt_odds_saves_snapshots(tmp_path):
    fake = TruncatingFake([])
    fake.rt_scripts = {"2026110105040311": [make_odds_record("O1", kubun="1", announced="11011420"),
                                            make_odds_record("O6", kubun="1", announced="11011420"), 0]}
    counts = jvlink.fetch_rt_odds(["2026110105040311", "2026110105040310"], out_dir=tmp_path, date="20261101",
                                  client=jvlink.JVLinkClient(com=fake), log=lambda *a: None)
    assert counts["レース"] == 1 and counts["データなし"] == 1 and counts["レコード"] == 2
    d = odds.odds_dir(tmp_path) / "rt" / "20261101"
    snap = odds.load_odds("O6", directory=d, final_only=False, latest=False)
    assert set(snap["データ区分"]) == {"1"} and set(snap["発表"]) == {"11011420"}
    assert snap["取得時刻"].iloc[0] == counts["取得時刻"]


def test_odds_files_are_gitignored():
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in gi.splitlines()       # data/jvlink/odds/ は data/ の下


# ===========================================================================
# SDK がある環境：構造体の位置と一致すること
# ===========================================================================
def test_parse_matches_sdk_structs():
    if not jvlink.struct_path().is_file():
        pytest.skip("SDK（JVSDK_DIR）が無い")
    mod = jvlink.load_struct_module()
    cases = [("O1", "JV_O1_ODDS_TANFUKUWAKU", "OddsTansyoInfo", "Umaban"),
             ("O2", "JV_O2_ODDS_UMAREN", "OddsUmarenInfo", "Kumi"),
             ("O4", "JV_O4_ODDS_UMATAN", "OddsUmatanInfo", "Kumi"),
             ("O5", "JV_O5_ODDS_SANREN", "OddsSanrenInfo", "Kumi"),
             ("O6", "JV_O6_ODDS_SANRENTAN", "OddsSanrentanInfo", "Kumi")]
    for rt, cls, attr, key in cases:
        raw = make_odds_record(rt, n_horses=9)
        obj = getattr(mod, cls).SetDataB(raw)
        sdk = [(int(getattr(x, key)), int(x.Odds), int(x.Ninki)) for x in getattr(obj, attr)
               if getattr(x, key).strip() and getattr(x, key).strip("0")]
        rec = odds.parse_odds(raw)
        ours = list(zip(rec.kumi.tolist(), rec.odds10.tolist(), rec.ninki.tolist()))
        assert ours == sdk, rt
        assert obj.head.DataKubun == rec.kubun and obj.id.RaceNum == "11"
