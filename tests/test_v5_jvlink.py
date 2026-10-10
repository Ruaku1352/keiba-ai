"""v5（JRA-VAN 取得・変換・再検証）のテスト。

JV-Link は Windows 専用の COM なので、ここでは偽物の JV-Link で置き換える。
テストは2種類に分かれる:

  SDK 不要（公開リポジトリの CI でも必ず走る）
    - 取得ループの戻り値の扱い（-3 で待つ、-1 で続ける、不要種別は JVSkip、必ず JVClose）
    - jvmap の変換（データ区分の重複解消、中央への絞り込み、単位変換、コード表）
    - JRA-VAN 形式の入力でのリーク検証（未来シャッフル不変性・同一レース内リーク）

  SDK 必要（環境変数 JVSDK_DIR に SDK が無ければ skip）
    - 仕様書どおりの長さのダミーバイト列を SDK の構造体に通して、値が取れること
    - cp932 への差し替えで「髙」「﨑」が消えないこと
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, features, jvlink, jvmap, pace, preprocess, trifecta, validate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# 合成データは小さいので統計量の最低母数を下げる（test_v2 と同じ理由）
pace.PACE_Z_MIN_COUNT = 5
pace.EXPECTED_PACE_MIN_COUNT = 5


# ===========================================================================
# 偽物の JV-Link
# ===========================================================================
class FakeJVLink:
    """JV-Link の COM オブジェクトの偽物。

    script は JVGets が順に返すもののリスト:
      bytes            → 正のレコード（戻り値 = 長さ）
      int（-1,0,-3...）→ その戻り値
    ファイルの区切りは -1。ファイル名は区切りごとに F000.jvd, F001.jvd … と変わる
    （本物の JVGets も読み込み中のファイル名を返す）。
    JVSkip が呼ばれたら、次の -1 まで（＝そのファイルの残り）を読み飛ばす。
    reopen_scripts を渡すと、2回目以降の JVOpen でその script に差し替わる
    （エラーで開き直したときに、本物と同じく先頭から読み直す動きを再現する）。
    """

    def __init__(self, script, open_ret=(0, 3, 0, "20261008120000"), status_seq=None,
                 init_ret=0, reopen_scripts=None):
        self.initial = list(script)
        self.script = list(script)
        self.reopen_scripts = [list(x) for x in (reopen_scripts or [])]
        self.open_ret = open_ret
        self.status_seq = list(status_seq or [])
        self.init_ret = init_ret
        self.calls = []
        self.closed = 0
        self.opened = 0
        self.deleted = []
        self.file_index = 0
        self.sizes = []          # JVGets に渡された size
        self.delay_fn = None     # JVGets の中で遅延を起こしたいときに使う
        self.gets_count = 0

    @property
    def fname(self):
        return f"F{self.file_index:03d}.jvd"

    def JVInit(self, sid):
        self.calls.append(("JVInit", sid))
        return self.init_ret

    def JVOpen(self, dataspec, fromtime, option, a, b, c):
        self.calls.append(("JVOpen", dataspec, fromtime, option))
        if self.opened > 0:
            self.script = self.reopen_scripts.pop(0) if self.reopen_scripts else list(self.initial)
        self.opened += 1
        self.file_index = 0
        return self.open_ret

    def JVStatus(self):
        return self.status_seq.pop(0) if self.status_seq else 0

    def JVGets(self, buff, size, name):
        self.sizes.append(size)
        if self.delay_fn is not None:
            self.delay_fn(self)
        if not self.script:
            return 0, None, ""
        item = self.script.pop(0)
        if isinstance(item, (bytes, bytearray)):
            return len(item), memoryview(bytes(item)), self.fname
        if item == -1:
            name = self.fname
            self.file_index += 1
            return -1, None, name
        return item, None, self.fname

    def JVRead(self, buff, size, name):
        """JVRead の偽物。JV-Link と同じく SJIS を Unicode 文字列にして返す。"""
        code, mv, fname = self.JVGets(None, size, None)
        text = bytes(mv).decode("cp932", errors="replace") if mv is not None else ""
        return code, text, code, fname

    def JVSkip(self):
        self.calls.append(("JVSkip",))
        while self.script and self.script[0] != -1:
            self.script.pop(0)

    def JVFiledelete(self, fname):
        self.deleted.append(fname)

    def JVClose(self):
        import signal
        self.closed += 1
        self.sigint_at_close = signal.getsignal(signal.SIGINT)
        return 0


def _client(fake):
    return jvlink.JVLinkClient(com=fake)


def _fetch_no_parse(fake, tmp_path, **kw):
    """構造体なしで取得ループだけを試す（record_types を空にすれば全種別が JVSkip 対象）。"""
    return jvlink.fetch(dataspec="RACE", fromtime="20260901000000", option=1,
                        out_dir=tmp_path, record_types=(), client=_client(fake),
                        struct_module=object(), sleep=lambda s: None, log=lambda *a: None)


# ===========================================================================
# 取得ループ（SDK 不要）
# ===========================================================================
def test_minus3_waits_and_retries(tmp_path):
    """-3（ダウンロード中）で打ち切らず、待ってから読み続けること。"""
    fake = FakeJVLink([-3, -3, b"O1" + b" " * 50, -1, 0])
    res = _fetch_no_parse(fake, tmp_path)
    assert res.skipped_files == {"O1": 1}   # -3 の後のレコードまで到達している
    assert fake.closed == 1


def test_minus3_gives_up_after_limit(tmp_path):
    """-3 が延々続くなら、上限で止めて例外にする（無限ループしない）。"""
    fake = FakeJVLink([-3] * 10_000)
    with pytest.raises(jvlink.JVLinkError):
        jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=(),
                     client=_client(fake), struct_module=object(),
                     sleep=lambda s: None, log=lambda *a: None,
                     retry_sleep_sec=1.0, max_retry_sec=5)
    assert fake.closed == 1  # 例外でも JVClose している


def test_unwanted_record_type_skips_whole_file(tmp_path):
    """不要な種別（O6 など）は1件読んだら JVSkip で残りを飛ばすこと。"""
    o6_file = [b"O6" + b" " * 100] * 5
    fake = FakeJVLink(o6_file + [-1] + [b"H1" + b" " * 10] * 3 + [-1, 0])
    res = _fetch_no_parse(fake, tmp_path)
    assert res.skipped_files == {"O6": 1, "H1": 1}
    assert sum(1 for c in fake.calls if c[0] == "JVSkip") == 2


def test_open_no_data_is_not_error(tmp_path):
    """JVOpen の -1（該当データなし）はエラーではなく、そのまま終わること。"""
    fake = FakeJVLink([], open_ret=(-1, 0, 0, ""))
    res = _fetch_no_parse(fake, tmp_path)
    assert res.open_code == -1
    assert fake.closed == 1


def test_open_error_raises_and_closes(tmp_path):
    """JVOpen の負の戻り値（-1以外）は例外にし、それでも JVClose すること。"""
    fake = FakeJVLink([], open_ret=(-301, 0, 0, ""))
    with pytest.raises(jvlink.JVLinkError, match="-301"):
        _fetch_no_parse(fake, tmp_path)
    assert fake.closed == 1


def test_open_accepts_non_tuple_return(tmp_path):
    """JVOpen が単値を返す環境にも対応すること（公式サンプルと同じ配慮）。"""
    fake = FakeJVLink([0], open_ret=0)
    res = _fetch_no_parse(fake, tmp_path)
    assert res.open_code == 0


def test_corrupt_file_is_deleted_and_reopened(tmp_path):
    """-402（ダウンロードしたファイルが異常）なら JVFiledelete して、自動で開き直す。"""
    fake = FakeJVLink([-402], reopen_scripts=[[b"O1" + b" " * 10, -1, 0]])
    res = _fetch_no_parse(fake, tmp_path)
    assert fake.deleted == ["F000.jvd"]
    assert res.reopened == 1
    assert fake.opened == 2
    assert res.skipped_files == {"O1": 1}   # 開き直した後は最後まで読めている


def test_corrupt_file_raises_when_reopen_disabled(tmp_path):
    fake = FakeJVLink([-402])
    with pytest.raises(jvlink.JVLinkError, match="-402"):
        jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=(),
                     client=_client(fake), struct_module=object(), max_reopen=0,
                     sleep=lambda s: None, log=lambda *a: None)
    assert fake.closed >= 1


def test_reopen_gives_up_after_limit(tmp_path):
    """開き直しても同じエラーが続くなら、上限回数で止める。"""
    fake = FakeJVLink([-502], reopen_scripts=[[-502]] * 10)
    with pytest.raises(jvlink.JVLinkError, match="-502"):
        jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=(),
                     client=_client(fake), struct_module=object(), max_reopen=2,
                     sleep=lambda s: None, log=lambda *a: None)
    assert fake.opened == 3   # 最初の1回 + 開き直し2回


def test_progress_file_records_finished_files_and_is_cleared(tmp_path):
    """読み終えたファイルが記録され、完走したら記録が消えること。"""
    fake = FakeJVLink([b"O1" + b" " * 10, -1, b"O2" + b" " * 10, -1, 0])
    _fetch_no_parse(fake, tmp_path)
    assert not list(tmp_path.glob("progress_*.txt"))   # 完走したので消えている


def test_resume_skips_finished_files(tmp_path):
    """途中で止まったら、次は読み終えたファイルを飛ばして続きから読むこと。"""
    files = [b"O1" + b" " * 10, -1, b"O2" + b" " * 10, -1, b"O3" + b" " * 10, -1, 0]
    # 1回目: 1つ目を読み終えて、2つ目に入ったところで通信エラー（開き直しなし）
    first = FakeJVLink(files[:2] + [-502])
    with pytest.raises(jvlink.JVLinkError):
        jvlink.fetch(fromtime="19860101000000", option=4, out_dir=tmp_path, record_types=(),
                     client=_client(first), struct_module=object(), max_reopen=0,
                     sleep=lambda s: None, log=lambda *a: None)
    progress = list(tmp_path.glob("progress_*.txt"))
    assert len(progress) == 1
    assert progress[0].read_text(encoding="utf-8").split() == ["F000.jvd"]

    # 2回目: 同じパラメータで再実行 → 1つ目は飛ばし、2つ目以降を読む
    second = FakeJVLink(files)
    res = jvlink.fetch(fromtime="19860101000000", option=4, out_dir=tmp_path, record_types=(),
                       client=_client(second), struct_module=object(),
                       sleep=lambda s: None, log=lambda *a: None)
    assert res.resumed_files == 1
    assert res.skipped_files == {"O2": 1, "O3": 1}   # O1 は種別を見る前に飛ばされた
    assert not list(tmp_path.glob("progress_*.txt"))


def test_resume_is_per_parameters(tmp_path):
    """fromtime や option が違う取得の記録は、別の取得の再開に使われないこと。"""
    p1 = jvlink.Progress(tmp_path, "RACE", 4, "19860101000000")
    p1.mark("F000.jvd")
    p2 = jvlink.Progress(tmp_path, "RACE", 1, "20260901000000")
    assert "F000.jvd" not in p2
    assert "F000.jvd" in jvlink.Progress(tmp_path, "RACE", 4, "19860101000000")


def test_waits_for_download_before_reading(tmp_path):
    """JVStatus がダウンロード数に達するまで JVGets を呼ばないこと（仕様書の注意）。"""
    fake = FakeJVLink([0], open_ret=(0, 3, 3, "x"), status_seq=[0, 1, 2, 3])
    _fetch_no_parse(fake, tmp_path)
    assert fake.status_seq == []  # 3 に達するまで全部消費した


def test_init_error_raises(tmp_path):
    fake = FakeJVLink([], init_ret=-303)
    with pytest.raises(jvlink.JVLinkError, match="-303"):
        _fetch_no_parse(fake, tmp_path)


def test_minus3_wait_grows_from_short_interval(tmp_path):
    """-3 の待ちは 0.05 秒から倍々に伸び、上限で頭打ちになること。

    1秒固定だと、すぐ終わる待ちでも毎回1秒を捨てることになる。
    """
    slept = []
    fake = FakeJVLink([-3] * 7 + [b"O1" + b" " * 10, -1, 0])
    jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=(),
                 client=_client(fake), struct_module=object(), retry_sleep_sec=1.0,
                 sleep=slept.append, log=lambda *a: None)
    assert slept == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0]


def test_buffer_size_is_passed_to_jvgets(tmp_path):
    fake = FakeJVLink([b"O1" + b" " * 10, -1, 0])
    jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=(),
                 client=_client(fake), struct_module=object(), buffer_size=2048,
                 sleep=lambda s: None, log=lambda *a: None)
    assert set(fake.sizes) == {2048}


def test_timing_per_file_and_minus3_attribution(tmp_path):
    """保存対象ファイルごとに件数・時間・-3 の回数が記録されること（parse なしで JVGets だけ）。"""
    rec = b"SE" + b" " * 553
    fake = FakeJVLink([-3, -3, rec, rec, rec, -1, b"O6" + b" " * 9, -1, rec, -1, 0])
    res = jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=("SE",),
                       client=_client(fake), parse=False,
                       sleep=lambda s: None, log=lambda *a: None)
    rows = res.timings
    assert [r["records"] for r in rows] == [3, 1]
    assert rows[0]["minus3_count"] == 2      # ファイルを読み始める前の -3 もこのファイルの待ち
    assert rows[1]["minus3_count"] == 0
    assert res.skipped_files == {"O6": 1}
    assert (tmp_path / "fetch_timing.csv").exists()
    logged = pd.read_csv(tmp_path / "fetch_timing.csv", encoding="utf-8-sig")
    assert list(logged["records"]) == [3, 1]


def test_timing_detects_slowdown_within_file(tmp_path):
    """読む位置に比例して JVGets が遅くなる場合、後半の1回あたりが前半より大きく出ること。"""
    import time as _time
    rec = b"SE" + b" " * 553
    fake = FakeJVLink([rec] * 300 + [-1, 0])

    def slower_and_slower(f):
        f.gets_count += 1
        _time.sleep(0.00002 * f.gets_count)   # 呼ぶたびに少しずつ遅くなる

    fake.delay_fn = slower_and_slower
    res = jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=("SE",),
                       client=_client(fake), parse=False, timing_path=None,
                       sleep=lambda s: None, log=lambda *a: None)
    row = res.timings[0]
    assert row["gets_last100_ms"] > 2 * row["gets_first100_ms"]


def test_bench_stop_keeps_progress_and_state(tmp_path):
    """max_data_files で途中終了したときは、完走扱いにしない（進捗を消さず state も保存しない）。"""
    rec = b"SE" + b" " * 553
    fake = FakeJVLink([rec, -1, rec, -1, rec, -1, 0])
    res = jvlink.fetch(fromtime="19860101000000", option=4, out_dir=tmp_path,
                       record_types=("SE",), client=_client(fake), parse=False,
                       max_data_files=1, sleep=lambda s: None, log=lambda *a: None)
    assert res.stopped_early and res.data_files == 1
    assert list(tmp_path.glob("progress_*.txt"))
    assert jvlink.load_state(tmp_path) == {}


def test_default_buffer_is_2048(tmp_path):
    """既定のバッファは 2048（bench で 110,000 の約2.7倍速かった）。"""
    assert jvlink.BUFFER_SIZE == 2048
    fake = FakeJVLink([b"O1" + b" " * 10, -1, 0])
    _fetch_no_parse(fake, tmp_path)
    assert set(fake.sizes) == {2048}


def test_buffer_must_fit_saved_record_types():
    """保存する種別（最長 RA 1,272 バイト）が切り捨てられるバッファは拒否する。"""
    assert max(jvlink.RECORD_LENGTHS.values()) < jvlink.BUFFER_SIZE
    with pytest.raises(ValueError, match="RA"):
        jvlink.check_buffer_size(1272)
    jvlink.check_buffer_size(1273)


def test_jvread_method_reads_records(tmp_path):
    """JVRead（文字列で返る）でも、種別の判定と読み飛ばしが同じように動くこと。"""
    rec = b"SE" + b" " * 553
    fake = FakeJVLink([rec, rec, -1, b"O6" + b" " * 50, -1, 0])
    res = jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=("SE",),
                       client=_client(fake), parse=False, method="read",
                       sleep=lambda s: None, log=lambda *a: None)
    assert res.records == {"SE": 2}
    assert res.skipped_files == {"O6": 1}
    assert fake.sizes  # JVRead も同じ size で呼ばれている


def test_ctrl_c_closes_and_ignores_second_ctrl_c(tmp_path):
    """Ctrl+C で止めても JVClose し、後始末の間は2回目の Ctrl+C を無視していること。"""
    import signal
    rec = b"SE" + b" " * 553
    fake = FakeJVLink([rec, -1, rec, rec])

    def interrupt_on_third_call(f):
        f.gets_count += 1
        if f.gets_count == 3:
            raise KeyboardInterrupt

    fake.delay_fn = interrupt_on_third_call
    messages = []
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        jvlink.fetch(fromtime="19860101000000", option=4, out_dir=tmp_path,
                     record_types=("SE",), client=_client(fake), parse=False,
                     sleep=lambda s: None, log=messages.append)
    assert fake.closed == 1
    assert fake.sigint_at_close == signal.SIG_IGN      # 後始末中は Ctrl+C を無視
    assert signal.getsignal(signal.SIGINT) == before   # 終わったら元に戻っている
    assert any("JVClose しました" in m for m in messages)
    # 1つ目のファイルは読み終えているので、再開記録に残っている
    assert "F000.jvd" in jvlink.Progress(tmp_path, "RACE", 4, "19860101000000")


def test_only_slow_files_are_printed(tmp_path):
    """画面には1秒以上かかったファイルだけ出す（CSV には全部記録する）。"""
    rec = b"SE" + b" " * 553
    fake = FakeJVLink([rec, -1, rec, -1, 0])
    messages = []
    res = jvlink.fetch(fromtime="20260901000000", out_dir=tmp_path, record_types=("SE",),
                       client=_client(fake), parse=False, sleep=lambda s: None,
                       log=messages.append)
    assert len(res.timings) == 2
    assert not any(".jvd SE" in m for m in messages)   # 速いファイルは表示しない
    assert len(pd.read_csv(tmp_path / "fetch_timing.csv", encoding="utf-8-sig")) == 2


def test_missing_sdk_gives_clear_error(tmp_path):
    """SDK が見つからないとき、JVSDK_DIR の設定方法を含むエラーになること。"""
    with pytest.raises(FileNotFoundError, match="JVSDK_DIR"):
        jvlink.load_struct_module(tmp_path / "nowhere")


def test_sdk_files_are_not_in_repository():
    """SDK の著作物（構造体・仕様書）がリポジトリに紛れ込んでいないこと。"""
    tracked = [p for p in REPO.rglob("*") if ".git" not in p.parts]
    names = {p.name for p in tracked}
    assert "JVData_Struct.py" not in names
    assert not any(n.startswith("JV-Data仕様書") for n in names)


def test_gitignore_excludes_jv_data():
    """取得データと SDK が .gitignore されていること（再配布禁止）。"""
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in text
    assert "JVData_Struct.py" in text


# ===========================================================================
# 合成の生データ（jvlink が保存する CSV と同じ列名・すべて文字列）
# ===========================================================================
def _race_key(year, jyo, kai, nichi, race_no, monthday):
    """レースのキーと、データ作成年月日（ここではレース当日にしておく）。"""
    return {"id.Year": f"{year:04d}", "id.MonthDay": monthday, "id.JyoCD": jyo,
            "id.Kaiji": f"{kai:02d}", "id.Nichiji": f"{nichi:02d}", "id.RaceNum": f"{race_no:02d}",
            "head.MakeDate.Year": f"{year:04d}", "head.MakeDate.Month": monthday[:2],
            "head.MakeDate.Day": monthday[2:]}


def _time_str(seconds: float) -> str:
    """94.5 秒 -> "1345"（分1桁 + 秒2桁 + 1/10秒1桁）。"""
    m = int(seconds // 60)
    t = round((seconds - 60 * m) * 10)
    return f"{m}{t:03d}"


def make_raw(n_races: int = 120, n_horses: int = 10, start_year: int = 2011,
             years: int = 4, seed: int = 0, graded_every: int = 5) -> dict[str, pd.DataFrame]:
    """RA / SE / HR の生データを作る。データ区分は全部 7（月曜確定）。"""
    rng = np.random.default_rng(seed)
    horses = [f"20{rng.integers(10, 20)}1{i:05d}" for i in range(40)]
    # 騎手は1レースに1頭しか乗れないので重複なしで選ぶ。
    # 調教師は1レースに複数頭を出せるので、少人数から重複ありで選ぶ（現実と同じ）。
    jockeys = [f"0{i:04d}" for i in range(n_horses + 6)]
    trainers = [f"0{i:04d}" for i in range(6)]
    days = pd.date_range(f"{start_year}-01-05", f"{start_year + years - 1}-12-25",
                         periods=n_races)
    ra, se, hr = [], [], []
    seq = 0
    for r, day in enumerate(days):
        jyo = ["05", "06", "08", "09"][r % 4]
        key = _race_key(day.year, jyo, 1 + r % 5, 1 + r % 8, 1 + r % 12, day.strftime("%m%d"))
        grade = "A" if r % graded_every == 0 else ("C" if r % graded_every == 1 else " ")
        track = ["11", "17", "23", "24"][r % 4]
        ra.append({"_seq": str(seq), "head.RecordSpec": "RA", "head.DataKubun": "7", **key,
                   "RaceInfo.Hondai": f"テストレース{r}", "RaceInfo.Ryakusyo10": f"テスト{r}",
                   "GradeCD": grade, "Kyori": str([1200, 1600, 2000, 2400][r % 4]),
                   "TrackCD": track, "TenkoBaba.TenkoCD": "1",
                   "TenkoBaba.SibaBabaCD": "1", "TenkoBaba.DirtBabaCD": "2",
                   "SyussoTosu": f"{n_horses:02d}", "HassoTime": "1530",
                   "HaronTimeS3": str(340 + rng.integers(-15, 15)),
                   "HaronTimeL3": str(350 + rng.integers(-15, 15))})
        seq += 1
        runners = rng.choice(horses, size=n_horses, replace=False)
        race_jockeys = rng.choice(jockeys, size=n_horses, replace=False)
        order = rng.permutation(n_horses) + 1
        corner = rng.permutation(n_horses) + 1
        for i, (h, rank) in enumerate(zip(runners, order)):
            se.append({"_seq": str(seq), "head.RecordSpec": "SE", "head.DataKubun": "7", **key,
                       "Wakuban": str(1 + i // 2), "Umaban": f"{i + 1:02d}", "KettoNum": h,
                       "Bamei": f"ウマ{h[-3:]}", "SexCD": "1", "Barei": "04",
                       "ChokyosiCode": rng.choice(trainers), "ChokyosiRyakusyo": "調教師",
                       "KisyuCode": race_jockeys[i], "KisyuRyakusyo": "騎手",
                       "Futan": "550", "BaTaijyu": str(470 + rng.integers(-20, 20)),
                       "ZogenFugo": "+", "ZogenSa": "004", "IJyoCD": "0",
                       "KakuteiJyuni": f"{rank:02d}", "Time": _time_str(94.0 + rank * 0.2),
                       "Jyuni1c": "00", "Jyuni2c": "00",
                       "Jyuni3c": f"{corner[i]:02d}", "Jyuni4c": f"{corner[i]:02d}",
                       "Odds": f"{int(rank * 25 + rng.integers(0, 30)):04d}",
                       "Ninki": f"{rank:02d}", "HaronTimeL4": "000",
                       "HaronTimeL3": str(340 + rank), "TimeDiff": "+005",
                       "DMTime": "13450", "DMJyuni": "01",
                       "KyakusituKubun": str(1 + (corner[i] - 1) * 4 // n_horses)})
            seq += 1
        win = [i + 1 for i in np.argsort(order)[:3]]
        hr_row = {"_seq": str(seq), "head.RecordSpec": "HR", "head.DataKubun": "2", **key}
        for k in range(1, 10):
            hr_row[f"FuseirituFlag[{k}]"] = "0"
            hr_row[f"TokubaraiFlag[{k}]"] = "0"
            hr_row[f"HenkanFlag[{k}]"] = "0"
        for prefix in ("PaySanrentan", "PaySanrenpuku"):
            for k in range(1, 7 if prefix == "PaySanrentan" else 4):
                hr_row[f"{prefix}[{k}].Kumi"] = "000000"
                hr_row[f"{prefix}[{k}].Pay"] = "000000000"
        hr_row["PaySanrentan[1].Kumi"] = "".join(f"{w:02d}" for w in win)
        hr_row["PaySanrentan[1].Pay"] = f"{int(np.exp(rng.normal(9.3, 1.4))):09d}"
        hr_row["PaySanrenpuku[1].Kumi"] = "".join(f"{w:02d}" for w in sorted(win))
        hr_row["PaySanrenpuku[1].Pay"] = f"{int(np.exp(rng.normal(7.2, 1.1))):09d}"
        hr.append(hr_row)
        seq += 1
    return {"RA": pd.DataFrame(ra), "SE": pd.DataFrame(se), "HR": pd.DataFrame(hr)}


# ===========================================================================
# 単位変換・コード表（SDK 不要）
# ===========================================================================
def test_race_time_conversion():
    s = pd.Series(["1345", "2005", "0000", "    "])
    out = jvmap.to_race_time(s)
    assert out[0] == pytest.approx(94.5)    # 1分34秒5
    assert out[1] == pytest.approx(120.5)   # 2分00秒5
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_tenths_conversion():
    assert jvmap.to_tenths(pd.Series(["345"]))[0] == pytest.approx(34.5)   # 上がり3F
    assert jvmap.to_tenths(pd.Series(["0123"]))[0] == pytest.approx(12.3)  # 単勝オッズ
    assert jvmap.to_tenths(pd.Series(["550"]))[0] == pytest.approx(55.0)   # 斤量
    assert np.isnan(jvmap.to_tenths(pd.Series(["000"]), invalid=("000",))[0])


def test_weight_diff_sign():
    """馬体重の増減は符号と差が別フィールド。結合して数値にする。"""
    sign = pd.Series(["+", "-", " ", " ", "+"])
    diff = pd.Series(["004", "012", "000", "   ", "999"])
    out = jvmap.to_signed_diff(sign, diff)
    assert out[0] == 4 and out[1] == -12
    assert out[2] == 0               # 前差なし
    assert np.isnan(out[3])          # 初出走
    assert np.isnan(out[4])          # 計量不能


def test_grade_mapping_matches_kaggle_labels():
    """A/B/C/D が Kaggle の G1/G2/G3/G に、障害重賞とリステッドは重賞外になること。"""
    assert jvmap.GRADE_LABELS["A"] == "G1"
    assert jvmap.GRADE_LABELS["D"] == "G"
    for code in ("F", "G", "H", "L"):
        assert jvmap.GRADE_LABELS[code] not in config.GRADED_VALUES
    assert "E" not in jvmap.GRADE_LABELS  # 重賞以外の特別は平場扱い
    # オプションで障害重賞を平地と同じ表記にもできる
    assert jvmap.GRADE_LABELS_JUMP_AS_FLAT["F"] == "G1"


def test_track_code_mapping():
    """トラックコード1つから 芝ダ・回り・内外 を復元できること。"""
    cases = {"11": ("芝", "左", "内"), "18": ("芝", "右", "外"), "10": ("芝", "直線", None),
             "24": ("ダート", "右", "内"), "29": ("ダート", "直線", None),
             "54": ("障害", None, "内"), "13": ("芝", "左", "内→外")}
    for code, (surface, turn, inout) in cases.items():
        assert jvmap.track_surface(code) == surface, code
        assert jvmap.track_turn(code) == turn, code
        assert jvmap.track_inout(code) == inout, code


def test_race_id_is_kaggle_12_digits():
    df = pd.DataFrame([_race_key(2026, "05", 4, 2, 11, "1004")])
    assert jvmap.make_race_id(df)[0] == "202605040211"


# ===========================================================================
# データ区分（SDK 不要）
# ===========================================================================
def _kubun_rows(*pairs):
    """(データ区分, 値) を届いた順に並べた RA 風の表。"""
    base = _race_key(2026, "05", 4, 2, 11, "1004")
    return pd.DataFrame([{"_seq": str(i), "head.DataKubun": k, **base, "v": v}
                         for i, (k, v) in enumerate(pairs)])


def test_kubun_keeps_most_confirmed():
    """速報(3) → 確定(7) と届いたら確定が残る。"""
    out = jvmap.apply_data_kubun(_kubun_rows(("3", "速報"), ("7", "確定")),
                                 jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)
    assert list(out["v"]) == ["確定"]


def test_kubun_late_preliminary_does_not_overwrite():
    """確定(7)の後に速報(3)が遅れて届いても、確定を上書きしない。"""
    out = jvmap.apply_data_kubun(_kubun_rows(("7", "確定"), ("3", "遅れた速報")),
                                 jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)
    assert list(out["v"]) == ["確定"]


def test_kubun_same_level_later_wins():
    """同じ確定度なら後から届いた方（訂正版）を採る。"""
    out = jvmap.apply_data_kubun(_kubun_rows(("7", "旧"), ("7", "訂正")),
                                 jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)
    assert list(out["v"]) == ["訂正"]


def test_kubun_zero_deletes():
    """区分0（該当レコード削除）が来たら、そのキーは消える。"""
    out = jvmap.apply_data_kubun(_kubun_rows(("7", "確定"), ("0", "")),
                                 jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)
    assert out.empty


def test_kubun_newer_make_date_wins_regardless_of_fetch_order():
    """訂正版（作成日が新しい）を先に取得し、元の版（作成日が古い）を後から取得しても、
    訂正版が残ること。

    直近1ヶ月を先に取ってから全期間のセットアップをすると、この順番で届く。
    取得順（_seq）で並べていた旧実装では、古い元の版が訂正版を上書きしていた。
    """
    base = _race_key(1987, "05", 4, 2, 11, "1004")
    df = pd.DataFrame([
        {**base, "_seq": "0", "head.DataKubun": "7", "head.MakeDate.Year": "2026",
         "head.MakeDate.Month": "09", "head.MakeDate.Day": "15", "v": "訂正版"},
        {**base, "_seq": "1", "head.DataKubun": "7", "head.MakeDate.Year": "1987",
         "head.MakeDate.Month": "10", "head.MakeDate.Day": "05", "v": "元の版"},
    ])
    out = jvmap.apply_data_kubun(df, jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)
    assert list(out["v"]) == ["訂正版"]


def test_kubun_same_make_date_falls_back_to_fetch_order():
    """作成日が同じなら、取得した順（後の方）を採ること。"""
    base = _race_key(2026, "05", 4, 2, 11, "1004")
    md = {"head.MakeDate.Year": "2026", "head.MakeDate.Month": "10", "head.MakeDate.Day": "05"}
    df = pd.DataFrame([
        {**base, **md, "_seq": "5", "head.DataKubun": "7", "v": "後"},
        {**base, **md, "_seq": "2", "head.DataKubun": "7", "v": "先"},
    ])
    out = jvmap.apply_data_kubun(df, jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)
    assert list(out["v"]) == ["後"]


def test_build_reports_races_without_horses():
    """RA だけ届いた過去レース（訂正など）は、どこで何件落ちたかを報告すること。"""
    raw = make_raw(n_races=10, years=1)
    extra = raw["RA"].iloc[[0, 1, 2]].copy()
    extra["id.Year"] = "1987"            # SE が届いていない過去レースの RA
    extra["_seq"] = ["900", "901", "902"]
    raw["RA"] = pd.concat([raw["RA"], extra], ignore_index=True)

    ds = jvmap.build_dataset(raw=raw)
    steps = dict(ds.report.steps)
    assert steps["RA のうち馬のデータが無く除外したレース"] == 3
    assert steps["最終レース数"] == 10
    assert any("1987: 3" in what for what, n in ds.report.steps if n is None)
    assert steps["最終レースのうち 3連単払戻あり"] == 10


def _with_old_races(raw, n_old=4, year=1960):
    """1986年より前のレース（RA と SE）を足す。HR は無い（実データと同じ）。"""
    ra_old = raw["RA"].iloc[:n_old].copy()
    se_old = raw["SE"].loc[raw["SE"]["id.RaceNum"].isin(ra_old["id.RaceNum"])
                           & raw["SE"]["id.Year"].isin(ra_old["id.Year"])
                           & raw["SE"]["id.JyoCD"].isin(ra_old["id.JyoCD"])].copy()
    for df in (ra_old, se_old):
        df["id.Year"] = str(year)
        df["head.MakeDate.Year"] = str(year)
    ra_old["_seq"] = [str(10_000 + i) for i in range(len(ra_old))]
    se_old["_seq"] = [str(20_000 + i) for i in range(len(se_old))]
    raw = dict(raw)
    raw["RA"] = pd.concat([raw["RA"], ra_old], ignore_index=True)
    raw["SE"] = pd.concat([raw["SE"], se_old], ignore_index=True)
    return raw, len(ra_old)


def test_build_starts_from_1986_by_default():
    """既定では1986年より前のレースとその馬を外し、件数と開催年の内訳を出すこと。"""
    raw, n_old = _with_old_races(make_raw(n_races=20, years=1))
    ds = jvmap.build_dataset(raw=raw)
    assert ds.race_result["レース日付"].min() >= pd.Timestamp("1986-01-01")
    steps = dict(ds.report.steps)
    assert steps["RA 開始日 1986-01-01 の範囲外を除外（生データには残す）"] == n_old
    assert any(what == f"開催年の内訳: {{1960: {n_old}}}" for what, n in ds.report.steps)
    assert steps["最終レース数"] == 20
    assert steps["RA のうち馬のデータが無く除外したレース"] == 0  # 古いレースはここでは数えない


def test_build_can_include_pre_1986():
    """start=None なら1986年より前も含めて作れる（生データは消していない）。"""
    raw, n_old = _with_old_races(make_raw(n_races=20, years=1))
    ds = jvmap.build_dataset(raw=raw, start=None)
    assert ds.race_result["レースID"].nunique() == 20 + n_old
    # 古いレースは HR が無いので「HR が未着」に数えられる
    assert any(f"HR が未着 {n_old}" in what for what, n in ds.report.steps if n is None)


def test_payout_reasons_split_not_sold_and_irregular():
    """払戻なしを「発売なし」と「不成立・特払」に分けて数えること。"""
    raw = make_raw(n_races=6, years=1)
    hr = raw["HR"].copy()
    hr.loc[0, "PaySanrentan[1].Kumi"] = "000000"                    # 発売なし
    hr.loc[1, "PaySanrentan[1].Kumi"] = "000000"                    # 発売なし
    hr.loc[2, "PaySanrentan[1].Kumi"] = "000000"                    # 不成立
    hr.loc[2, "FuseirituFlag[9]"] = "1"
    hr.loc[3, "HenkanFlag[9]"] = "1"                                # 返還（既定は除外しない）
    raw["HR"] = hr
    ds = jvmap.build_dataset(raw=raw)
    notes = [what for what, n in ds.report.steps if n is None]
    assert any("3連単の発売なし 2 / 不成立・特払 1 / 返還など（除外の設定） 0" in w for w in notes)
    assert dict(ds.report.steps)["最終レースのうち 3連単払戻あり"] == 3
    assert any(w.startswith("3連単の発売なしの開催年") for w in notes)

    ds2 = jvmap.build_dataset(raw=raw, exclude_irregular_payout=True)
    notes2 = [what for what, n in ds2.report.steps if n is None]
    assert any("返還など（除外の設定） 1" in w for w in notes2)


def test_three_furlongs_filled_from_laps():
    """公式の前3F・後3Fが空なら、ラップタイムから計算して補うこと。"""
    raw = make_raw(n_races=3, years=1)
    ra = raw["RA"].copy()
    laps = ["120", "110", "115", "120", "118", "116", "117", "119"]   # 1600m（8本）
    for i in range(1, 26):
        ra[f"LapTime[{i}]"] = laps[i - 1] if i <= len(laps) else "000"
    ra.loc[0, ["HaronTimeS3", "HaronTimeL3"]] = "000"   # 公式値なし → ラップから
    ra.loc[1, ["HaronTimeS3", "HaronTimeL3"]] = ["345", "352"]   # 公式値あり → そのまま
    for i in range(1, 26):
        ra.loc[2, f"LapTime[{i}]"] = "000"
    ra.loc[2, ["HaronTimeS3", "HaronTimeL3"]] = "000"   # どちらも無い
    races = jvmap.prepare_races(ra)
    assert races.loc[0, "_前3F"] == pytest.approx(12.0 + 11.0 + 11.5)
    assert races.loc[0, "_後3F"] == pytest.approx(11.6 + 11.7 + 11.9)
    assert races.loc[0, "_3F出所"] == "ラップ"
    assert races.loc[1, "_前3F"] == pytest.approx(34.5) and races.loc[1, "_3F出所"] == "公式"
    assert np.isnan(races.loc[2, "_前3F"]) and races.loc[2, "_3F出所"] == "なし"


def test_coverage_by_year_table():
    """年ごとの欠損率の表が作れること（3F は平地だけで数える）。"""
    raw = make_raw(n_races=40, years=2)
    se = raw["SE"].copy()
    se.loc[se["id.Year"] == "2011", "KyakusituKubun"] = "0"     # 1年目は脚質判定なし
    raw["SE"] = se
    table = jvmap.coverage_by_year(jvmap.build_dataset(raw=raw))
    assert {"頭数", "_前3F", "_後3F", "3F_ラップ補完", "_4コーナー順位", "_脚質判定",
            "馬体重", "後３Ｆタイム"} <= set(table.columns)
    assert table.loc[2011, "_脚質判定"] == 100.0
    assert table.loc[2012, "_脚質判定"] == 0.0
    assert table.loc[2011, "_1コーナー順位"] == 100.0   # 合成データは1角を通らない


def test_jump_grade_labels_match_kaggle():
    """障害重賞は Kaggle と同じ「J.G1」表記。"""
    assert [jvmap.GRADE_LABELS[c] for c in "FGH"] == ["J.G1", "J.G2", "J.G3"]


def test_no_runtime_warning_when_run_as_module():
    """`py -m src.jvlink` で RuntimeWarning（二重読み込みの警告）が出ないこと。"""
    import subprocess
    r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-m", "src.jvlink", "--help"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "RuntimeWarning" not in r.stderr


def test_cancelled_and_partial_races_are_excluded():
    """中止(9)と、3着/5着までしか無い速報(3/4)のレースは学習に使わない。"""
    raw = make_raw(n_races=12, years=1)
    ra = raw["RA"].copy()
    ra.loc[0, "head.DataKubun"] = "9"
    ra.loc[1, "head.DataKubun"] = "3"
    ra.loc[2, "head.DataKubun"] = "4"
    races = jvmap.prepare_races(ra)
    assert len(races) == 12 - 3


def test_non_central_is_excluded():
    """地方（競馬場コード30以上）・海外（区分A/B）を落とす。"""
    raw = make_raw(n_races=8, years=1)
    ra = raw["RA"].copy()
    ra.loc[0, "id.JyoCD"] = "30"           # 門別
    ra.loc[1, "head.DataKubun"] = "A"      # 地方
    ra.loc[2, "head.DataKubun"] = "B"      # 海外
    ra.loc[3, "id.JyoCD"] = "00"           # 未設定
    assert len(jvmap.prepare_races(ra)) == 8 - 4


def test_abnormal_horses_are_excluded():
    """取消・除外・中止・失格は落とし、降着(7)は残す。"""
    raw = make_raw(n_races=3, years=1)
    se = raw["SE"].copy()
    for i, code in enumerate(["1", "2", "3", "4", "5", "7"]):
        se.loc[i, "IJyoCD"] = code
    races = jvmap.prepare_races(raw["RA"])
    horses = jvmap.prepare_horses(se, races)
    assert len(horses) == len(se) - 5
    assert (horses["異常区分"] == "7").sum() == 1


def test_payout_irregular_flags_exclude():
    """3連単の不成立・特払・返還があるレースは払戻を NaN にする（検証から外す）。"""
    raw = make_raw(n_races=6, years=1)
    hr = raw["HR"].copy()
    hr.loc[0, "FuseirituFlag[9]"] = "1"
    hr.loc[1, "TokubaraiFlag[9]"] = "1"
    hr.loc[2, "HenkanFlag[9]"] = "1"
    hr.loc[3, "HenkanFlag[1]"] = "1"   # 単勝の返還は3連単に関係ない
    pay = jvmap.prepare_payout(hr, exclude_irregular=True)
    assert pay["3連単払戻"].isna().sum() == 3
    assert pay.loc[3, "3連単払戻"] > 0


def test_payout_keeps_irregular_by_default():
    """既定では不成立・特払・返還があっても除外しない（v4 と条件を揃える）。"""
    raw = make_raw(n_races=6, years=1)
    hr = raw["HR"].copy()
    hr.loc[2, "HenkanFlag[9]"] = "1"
    report = jvmap.BuildReport()
    pay = jvmap.prepare_payout(hr, report)
    assert pay["3連単払戻"].notna().all()
    assert ("HR 3連単 不成立・特払・返還あり（除外しない）", 1) in report.steps


def test_payout_uses_hundred_yen_units():
    raw = make_raw(n_races=1, years=1)
    hr = raw["HR"].copy()
    hr.loc[0, "PaySanrentan[1].Pay"] = "000123450"
    pay = jvmap.prepare_payout(hr)
    assert pay.loc[0, "3連単払戻"] == 123450


# ===========================================================================
# 既存パイプラインとの接続（SDK 不要）
# ===========================================================================
@pytest.fixture(scope="module")
def dataset():
    return jvmap.build_dataset(raw=make_raw())


def test_columns_resolve_to_codes(dataset):
    """馬・騎手・調教師が名前ではなくコードの列に解決されること。"""
    cols = config.resolve_columns(dataset.race_result)
    assert cols["horse"] == "血統登録番号"
    assert cols["jockey"] == "騎手コード"
    assert cols["trainer"] == "調教師コード"
    assert cols["grade"] == "リステッド・重賞競走"


def test_kaggle_column_resolution_unchanged():
    """Kaggle 形式では従来どおり 馬名・騎手・調教師 に解決されること（後方互換）。"""
    df = pd.DataFrame(columns=["レースID", "レース日付", "馬名", "騎手", "調教師", "着順"])
    cols = config.resolve_columns(df)
    assert (cols["horse"], cols["jockey"], cols["trainer"]) == ("馬名", "騎手", "調教師")


def test_pipeline_runs_on_jv_data(dataset):
    """変換結果が preprocess → features にそのまま通ること（下流は無改修）。"""
    df = preprocess.basic_clean(dataset.race_result)
    out = features.add_all_features(df, lap_df=dataset.lap_df)
    for col in ["脚質スコア", "公式脚質_過去平均", "公式脚質_逃げ率", "想定ペース", "通算勝率"]:
        assert col in out.columns, col
    assert out["脚質スコア"].notna().any()
    assert out["公式脚質_過去平均"].notna().any()
    # その日の結果である列（_ 始まり）は特徴量に残らない
    assert not [c for c in out.columns if c.startswith("_")]
    feats = features.feature_columns(out)
    assert "公式脚質_過去平均" in feats
    assert not any(c.startswith("DM_") for c in feats)       # DM予想は初回は入れない
    assert "単勝オッズ" not in feats and "人気" not in feats  # オッズも入れない


def test_relative_position_uses_official_field_size(dataset):
    """4角相対位置の分母に公式の出走頭数を使い、0〜1に収まること。"""
    df = preprocess.basic_clean(dataset.race_result)
    from src import corner
    out = corner.attach_corner_from_columns(df)
    pos = out["_4角相対位置"].dropna()
    assert pos.between(0, 1).all()
    # 1番手は 0、最後方は 1
    first = out.loc[out["_4コーナー順位"] == 1, "_4角相対位置"]
    assert (first == 0).all()


def test_official_style_first_race_is_nan(dataset):
    df = preprocess.basic_clean(dataset.race_result)
    out = features.add_all_features(df, lap_df=dataset.lap_df)
    first = out.groupby("血統登録番号", sort=False).head(1)
    assert first["公式脚質_過去平均"].isna().all()
    assert first["公式脚質_前走"].isna().all()


def test_official_style_equals_past_mean(dataset):
    """公式脚質_過去平均 が「自分より前の脚質判定の平均」と一致すること。"""
    df = preprocess.basic_clean(dataset.race_result)
    out = features.add_all_features(df, lap_df=dataset.lap_df, drop_helper_cols=False)
    for _, g in out.groupby("血統登録番号", sort=False):
        k = pd.to_numeric(g["_脚質判定"]).to_numpy(dtype=float)
        got = g["公式脚質_過去平均"].to_numpy()
        for i in range(1, len(g)):
            past = k[:i][~np.isnan(k[:i])]
            if len(past):
                assert got[i] == pytest.approx(past.mean(), abs=1e-4)


# ===========================================================================
# リーク検証（JRA-VAN 形式の入力で再確認）
# ===========================================================================
def _feature_frame(raw):
    ds = jvmap.build_dataset(raw=raw)
    df = preprocess.basic_clean(ds.race_result)
    return features.add_all_features(df, lap_df=ds.lap_df)


CHECK_COLS = ["通算勝率", "騎手通算勝率", "調教師通算勝率", "調教師通算出走数", "脚質スコア", "前走4角相対位置",
              "公式脚質_過去平均", "公式脚質_逃げ率", "公式脚質_前走", "公式脚質_レース内平均差",
              "前走上がり3F", "前走ペース指標", "想定ペース"]


def test_future_shuffle_invariance_jv():
    """未来のレースの着順・コーナー順位・脚質判定・3Fを書き換えても、過去行の特徴量は不変。"""
    raw = make_raw(n_races=160, seed=3)
    f1 = _feature_frame(raw)

    tampered = {k: v.copy() for k, v in raw.items()}
    se, ra = tampered["SE"], tampered["RA"]
    cutoff = "2013"
    future_se = se["id.Year"] >= cutoff
    rng = np.random.default_rng(42)
    se.loc[future_se, "KakuteiJyuni"] = rng.permutation(se.loc[future_se, "KakuteiJyuni"].to_numpy())
    se.loc[future_se, "Jyuni4c"] = rng.permutation(se.loc[future_se, "Jyuni4c"].to_numpy())
    se.loc[future_se, "KyakusituKubun"] = rng.permutation(se.loc[future_se, "KyakusituKubun"].to_numpy())
    se.loc[future_se, "HaronTimeL3"] = "399"
    future_ra = ra["id.Year"] >= cutoff
    ra.loc[future_ra, "HaronTimeS3"] = "300"
    f2 = _feature_frame(tampered)

    past = (f1["レース日付"].dt.year < int(cutoff)).to_numpy()
    for col in CHECK_COLS:
        a = f1.loc[past, col].astype("float64").fillna(-999).to_numpy()
        b = f2.loc[past, col].astype("float64").fillna(-999).to_numpy()
        assert np.allclose(a, b), f"{col} が未来の改変で変化した"


def test_no_same_race_leak_jv():
    """自分のレースのコーナー順位・脚質判定・着順を書き換えても、自分の行の特徴量は不変。"""
    raw = make_raw(n_races=160, seed=4)
    f1 = _feature_frame(raw)

    tampered = {k: v.copy() for k, v in raw.items()}
    se = tampered["SE"]
    target = jvmap.make_race_id(se).unique()[80]
    rows = jvmap.make_race_id(se) == target
    n = int(rows.sum())
    se.loc[rows, "Jyuni4c"] = [f"{n - i:02d}" for i in range(n)]       # 通過順を逆転
    se.loc[rows, "KyakusituKubun"] = "4"
    se.loc[rows, "KakuteiJyuni"] = [f"{n - i:02d}" for i in range(n)]  # 着順も逆転
    f2 = _feature_frame(tampered)

    mask = (f1["レースID"] == target).to_numpy()
    for col in CHECK_COLS:
        a = f1.loc[mask, col].astype("float64").fillna(-999).to_numpy()
        b = f2.loc[mask, col].astype("float64").fillna(-999).to_numpy()
        assert np.allclose(a, b), f"{col} に同一レースの結果が漏れている"


def test_trainer_stats_exclude_same_race_stablemates():
    """同じレースに出た同厩馬の結果が、調教師の過去成績に混ざらないこと。

    v1〜v4 は cumsum - 自分 だったので、馬番の若い同厩馬が勝つと、
    後ろに並ぶ同厩馬の「調教師通算勝率」にその勝ちが入っていた（v5 で修正）。
    """
    df = pd.DataFrame({
        "レースID": ["R1", "R1", "R2", "R2", "R3"],
        "レース日付": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-08"] * 2 + ["2020-01-15"]),
        "馬名": ["a", "b", "c", "d", "e"],
        "馬番": [1, 2, 1, 2, 1],
        "着順": [1, 2, 2, 1, 1],
        "タイム": 100.0,
        "調教師": ["T"] * 5,
        "騎手": ["J1", "J2", "J1", "J2", "J1"],
    })
    df = preprocess.basic_clean(df)
    out = features.add_trainer_features(df)
    # R1 の2頭目: 1頭目（同じレース）の勝ちを見てはいけない → 過去なしで NaN
    assert np.isnan(out.loc[1, "調教師通算勝率"])
    assert out.loc[1, "調教師通算出走数"] == 0
    # R2 の2頭は、どちらも R1 の2戦1勝だけを見る
    assert out.loc[2, "調教師通算勝率"] == pytest.approx(0.5)
    assert out.loc[3, "調教師通算勝率"] == pytest.approx(0.5)
    assert out.loc[3, "調教師通算出走数"] == 2
    # R3 は R1・R2 の4戦2勝
    assert out.loc[4, "調教師通算勝率"] == pytest.approx(0.5)
    assert out.loc[4, "調教師通算出走数"] == 4


# ===========================================================================
# v5 の判定（SDK 不要）
# ===========================================================================
def test_v4_reference_interval_is_consistent():
    """v4 の熱い当たり数（逆算値）が、依頼書の区間と一致すること。"""
    lo, hi = trifecta.wilson_interval(validate.V4_REFERENCE["pooled_hits"],
                                      validate.V4_REFERENCE["pooled_n"])
    assert lo == pytest.approx(0.0497, abs=5e-4)
    assert hi == pytest.approx(0.0755, abs=5e-4)


def _results_with_rate(rate_graded, rate_flat=0.03, n=600, since_year=2022):
    """2021年8月以降の重賞の達成率を狙って作った fold 結果。"""
    from test_v4_validate import make_fold_race

    race = make_fold_race(n=n, hot_rate_graded=rate_graded, hot_rate_flat=rate_flat, seed=7)
    race["日付"] = pd.date_range(f"{since_year}-01-01", periods=n, freq="D")
    return [{"fold": validate.Fold("5", (2003, 2021), (2022, 2026)), "race": race, "auc": 0.78}]


def test_recent_check_detects_collapse():
    """直近の達成率が v4 より明確に低ければ「崩れた」と判定すること。"""
    out = validate.recent_period_check(_results_with_rate(0.0, n=2000))
    assert out["判定（達成率）"].startswith("崩れた")


def test_recent_check_keeps_when_similar():
    """v4 と同程度なら「維持」と判定すること（有意に低くはない）。"""
    out = validate.recent_period_check(_results_with_rate(0.065, n=1500))
    assert out["判定（達成率）"].startswith("維持")


def test_recent_check_uses_only_recent_rows():
    """since より前のレースは集計に入らないこと。"""
    results = _results_with_rate(0.06, n=400, since_year=2019)  # 2019〜2020 のデータ
    out = validate.recent_period_check(results, since="2021-08-01")
    assert out["重賞レース数"] == 0


def test_run_v5_end_to_end():
    """JRA-VAN 形式のデータで v5 の検証が一通り動くこと（学習器は偽物）。"""
    raw = make_raw(n_races=300, start_year=2019, years=6, seed=5, graded_every=3)
    ds = jvmap.build_dataset(raw=raw)

    def fake_model(train_df, test_df, feature_cols):
        imp = pd.DataFrame({"feature": feature_cols,
                            "gain": np.arange(len(feature_cols), dtype=float),
                            "split": 1})
        return np.random.default_rng(len(train_df)).random(len(test_df)), 0.75, imp

    folds = [validate.Fold("1", (2019, 2020), (2021, 2022)),
             validate.Fold("2", (2019, 2022), (2023, 2024))]
    out = validate.run_v5(ds, folds=folds, model_fn=fake_model, verbose=False, ablation=True)

    assert {"recent", "comparison", "style_importance", "ablation"} <= set(out)
    assert set(out["style_importance"]["系統"]) == {"公式脚質判定", "自前の脚質"}
    assert len(out["comparison"]) >= 6
    assert out["recent"]["重賞レース数"] > 0
    assert len(out["ablation"]) == 2


def test_v5_folds_are_walk_forward():
    for f in validate.V5_FOLDS:
        assert f.train_years[1] < f.test_years[0]
    assert validate.V5_FOLDS[-1].test_years[1] >= 2026


# ===========================================================================
# SDK を使うテスト（JVSDK_DIR が無ければ skip）
# ===========================================================================
def _sdk_module():
    try:
        return jvlink.load_struct_module()
    except FileNotFoundError:
        pytest.skip("JVSDK_DIR に JRA-VAN SDK が無いので skip（CI では正常）")


def _record(rt: str, fields: dict[int, bytes]) -> bytes:
    """仕様書どおりの長さの空白バイト列に、指定位置（1始まり）へ値を書き込む。"""
    buf = bytearray(b" " * jvlink.RECORD_LENGTHS[rt])
    buf[0:2] = rt.encode()
    for pos, value in fields.items():
        buf[pos - 1:pos - 1 + len(value)] = value
    buf[-2:] = b"\r\n"
    return bytes(buf)


# 位置は JV-Data仕様書 4.9.0.1「フォーマット」シートより（テストのダミー作成用）
def _ra_bytes():
    return _record("RA", {
        3: b"7", 12: b"2026", 16: b"1004", 20: b"05", 22: b"04", 24: b"02", 26: b"11",
        33: "テスト記念".encode("cp932"), 615: b"A", 698: b"2400", 706: b"11",
        884: b"18", 888: b"1", 889: b"2", 890: b"1", 970: b"352", 976: b"341",
    })


def _se_bytes():
    return _record("SE", {
        3: b"7", 12: b"2026", 16: b"1004", 20: b"05", 22: b"04", 24: b"02", 26: b"11",
        28: b"3", 29: b"05", 31: b"2021101234",
        41: "髙﨑テスト".encode("cp932"),           # shift_jis では消える文字
        79: b"1", 83: b"05", 86: b"01234",
        289: b"570", 297: b"05678", 325: b"482", 328: b"-", 329: b"006",
        332: b"0", 335: b"03", 339: b"2245", 358: b"04",
        360: b"0123", 364: b"02", 391: b"345", 553: b"2",
    })


def _hr_bytes():
    fields = {3: b"2", 12: b"2026", 16: b"1004", 20: b"05", 22: b"04", 24: b"02", 26: b"11"}
    for i in range(27):          # フラグ 9 × 3 種類を "0" に
        fields[32 + i] = b"0"
    fields[49] = b"1"            # 特払フラグ 3連単（41 + 8）
    fields[604] = b"050302"      # 3連単 組番
    fields[610] = b"000123450"   # 3連単 払戻
    return _record("HR", fields)


def test_sdk_parses_ra_fields():
    m = _sdk_module()
    rt, row = jvlink.parse_record(_ra_bytes(), m)
    assert rt == "RA"
    assert row["GradeCD"] == "A" and row["Kyori"] == "2400" and row["TrackCD"] == "11"
    assert row["SyussoTosu"] == "18"
    assert row["HaronTimeS3"] == "352" and row["HaronTimeL3"] == "341"
    assert row["RaceInfo.Hondai"] == "テスト記念"
    assert len(row["LapTime[25]"]) >= 0  # リストの全要素が列になっている


def test_sdk_parses_se_fields_and_keeps_cp932_chars():
    """cp932 への差し替えで「髙」「﨑」が消えないこと（shift_jis だと消える）。"""
    m = _sdk_module()
    rt, row = jvlink.parse_record(_se_bytes(), m)
    assert rt == "SE"
    assert row["Bamei"] == "髙﨑テスト"
    assert row["KettoNum"] == "2021101234"
    assert row["KakuteiJyuni"] == "03" and row["Jyuni4c"] == "04"
    assert row["Odds"] == "0123" and row["KyakusituKubun"] == "2"
    assert row["ZogenFugo"] == "-" and row["ZogenSa"] == "006"

    # 差し替えない場合は機種依存文字が消えることも確認しておく（差し替えの理由の裏付け）
    raw_mod = jvlink.load_struct_module(patch_cp932=False)
    _, raw_row = jvlink.parse_record(_se_bytes(), raw_mod)
    assert raw_row["Bamei"] != "髙﨑テスト"


def test_sdk_parses_hr_flags_with_one_based_columns():
    """HR のフラグ配列が1始まりの列名で、3連単が [9] になっていること。"""
    m = _sdk_module()
    rt, row = jvlink.parse_record(_hr_bytes(), m)
    assert rt == "HR"
    assert row["TokubaraiFlag[9]"] == "1"
    assert row["TokubaraiFlag[8]"] == "0"
    assert row["PaySanrentan[1].Kumi"] == "050302"
    assert row["PaySanrentan[1].Pay"] == "000123450"


def test_sdk_fetch_writes_csv_and_maps(tmp_path):
    """偽 JV-Link + 本物の構造体で、取得 → CSV → 変換 まで通ること。"""
    m = _sdk_module()
    script = ([_ra_bytes(), -1, _se_bytes(), -1]
              + [b"O6" + b" " * 200] * 3 + [-1, _hr_bytes(), -1, 0])
    fake = FakeJVLink(script)
    res = jvlink.fetch(fromtime="20261001000000", out_dir=tmp_path,
                       client=_client(fake), struct_module=m,
                       sleep=lambda s: None, log=lambda *a: None)
    assert res.records == {"RA": 1, "SE": 1, "HR": 1}
    assert res.skipped_files == {"O6": 1}
    assert fake.closed == 1
    assert jvlink.load_state(tmp_path)["RACE"]["last_file_timestamp"] == "20261008120000"

    raw = jvmap.load_raw(tmp_path)
    assert raw["SE"].loc[0, "Bamei"] == "髙﨑テスト"   # CSV 往復でも文字が残る
    races = jvmap.prepare_races(raw["RA"])
    assert races.loc[0, "レースID"] == "202605040211"
    assert races.loc[0, "リステッド・重賞競走"] == "G1"
    assert races.loc[0, "芝・ダート区分"] == "芝"
    horses = jvmap.prepare_horses(raw["SE"], races)
    h = horses.iloc[0]
    assert h["着順"] == 3 and h["タイム"] == pytest.approx(144.5)
    assert h["場体重増減"] == -6 and h["単勝オッズ"] == pytest.approx(12.3)
    pay = jvmap.prepare_payout(raw["HR"], exclude_irregular=True)
    assert np.isnan(pay.loc[0, "3連単払戻"])  # 特払フラグで除外
    assert jvmap.prepare_payout(raw["HR"]).loc[0, "3連単払戻"] == 123450  # 既定は除外しない


def test_sdk_jvread_gives_same_values_as_jvgets(tmp_path):
    """JVRead（Unicode で返る）を cp932 に戻して構造体に通しても、JVGets と同じ値になること。"""
    m = _sdk_module()
    rows = {}
    for method in ("gets", "read"):
        out = tmp_path / method
        fake = FakeJVLink([_se_bytes(), -1, 0])
        jvlink.fetch(fromtime="20261001000000", out_dir=out, record_types=("SE",),
                     client=_client(fake), struct_module=m, method=method,
                     sleep=lambda s: None, log=lambda *a: None)
        rows[method] = jvmap.load_raw(out, record_types=("SE",))["SE"].drop(columns="_seq")
    pd.testing.assert_frame_equal(rows["gets"], rows["read"])
    assert rows["read"].loc[0, "Bamei"] == "髙﨑テスト"


def test_sdk_fetch_appends_and_keeps_order(tmp_path):
    """2回目の取得は既存 CSV に追記され、通し番号が続くこと。"""
    m = _sdk_module()
    for _ in range(2):
        fake = FakeJVLink([_ra_bytes(), -1, 0])
        jvlink.fetch(fromtime="20261001000000", out_dir=tmp_path, record_types=("RA",),
                     client=_client(fake), struct_module=m,
                     sleep=lambda s: None, log=lambda *a: None)
    ra = jvmap.load_raw(tmp_path, record_types=("RA",))["RA"]
    assert list(ra["_seq"]) == ["0", "1"]
    # 同じレースが2回届いても、重複解消で1行になる
    assert len(jvmap.apply_data_kubun(ra, jvmap.RACE_KEY, jvmap.RACE_KUBUN_PRIORITY)) == 1
