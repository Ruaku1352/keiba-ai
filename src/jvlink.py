"""v5 依頼A：JV-Link から RA / SE / HR を取得して CSV に保存する。

■ このモジュールの役割は「取得して生のまま保存する」ことだけ
  - 構造体の解釈は SDK 同梱の JVData_Struct.py に任せる（自前でオフセットを書かない）
  - データ区分による重複の解消・中央競馬への絞り込み・単位変換は jvmap.py の仕事
  ここで変換まで済ませないのは、取得をやり直さずに変換ルールだけ直せるようにするため。
  JV-Link の取得は時間がかかるので、生データは一度取ったら使い回したい。

■ SDK はリポジトリに入れない
  JVData_Struct.py は JRA システムサービスの著作物で、リポジトリは公開されている。
  コミットすると再配布になるので、SDK の展開先（環境変数 JVSDK_DIR）から
  importlib で実行時に読み込む。

■ 文字コード
  SDK の MidB2S は shift_jis でデコードするため、「髙」「﨑」などの機種依存文字が
  errors="ignore" で黙って消える。読み込んだモジュールの MidB2S を cp932 版に
  差し替えてから使う（ファイルは書き換えず、実行時に関数を入れ替えるだけ）。
  SetDataB は呼び出し時にモジュールの MidB2S を探すので、差し替えが全フィールドに効く。
  ただし馬・騎手・調教師の識別にはコード（KettoNum / KisyuCode / ChokyosiCode）を使い、
  名前は表示用に留める。

■ JV-Link の読み込みループ（公式サンプル Form2.py との違い）
  1. JVGets の戻り値 -3（ダウンロード中）は**待って再試行**する。
     サンプルはエラー扱いで打ち切るが、仕様書の指示は「少し待ってから読み込みを再開」。
  2. 不要なレコード種別（オッズ O1〜O6 や票数 H1/H6）を読んだら JVSkip でファイルごと飛ばす。
     仕様書に「蓄積系データは1つのファイルにレコード種別は1種類しか収容されていない」
     とあるので、先頭1件を見れば残りは読まなくてよい。RACE には巨大な3連単オッズ(O6)
     も含まれるので、これをやるかどうかで取得時間が大きく変わる。
  3. 何があっても最後に JVClose を呼ぶ（try/finally）。呼ばないと次の JVOpen が -202 になる。

Windows での使い方は README の「v5：JRA-VAN」節を参照。

    py -m src.jvlink check
    py -m src.jvlink fetch --from 20260901000000 --option 1
    py -m src.jvlink summary
"""

from __future__ import annotations

import __future__
import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

from . import config

# レコード種別ID -> SDK の構造体クラス名
RECORD_STRUCTS = {
    "RA": "JV_RA_RACE",
    "SE": "JV_SE_RACE_UMA",
    "HR": "JV_HR_PAY",
}

# 仕様書のレコード長（CR/LF を含む）。短すぎるレコードを検出するのに使う
RECORD_LENGTHS = {"RA": 1272, "SE": 555, "HR": 719}

# 保存するフィールド（構造体の属性パス）。
#   "X[*]"   はリストの全要素を X[1], X[2], ... として保存（**1始まり**で列名を付ける）
#   "X[*].Y" はリスト要素の属性 Y を X[1].Y, X[2].Y, ... として保存
# 全フィールドを保存すると SE だけで数GBになるので、使うものに絞っている。
# 後で必要になったら、ここに足して取り直せばよい。
_RACE_KEY = ["head.RecordSpec", "head.DataKubun", "head.MakeDate.Year",
             "head.MakeDate.Month", "head.MakeDate.Day",
             "id.Year", "id.MonthDay", "id.JyoCD", "id.Kaiji", "id.Nichiji", "id.RaceNum"]

FIELDS: dict[str, list[str]] = {
    "RA": _RACE_KEY + [
        "RaceInfo.YoubiCD", "RaceInfo.Hondai", "RaceInfo.Ryakusyo10",
        "GradeCD", "JyokenInfo.SyubetuCD", "JyokenInfo.JyokenCD[*]",
        "Kyori", "TrackCD", "CourseKubunCD", "HassoTime",
        "TorokuTosu", "SyussoTosu", "NyusenTosu",
        "TenkoBaba.TenkoCD", "TenkoBaba.SibaBabaCD", "TenkoBaba.DirtBabaCD",
        "LapTime[*]", "HaronTimeS3", "HaronTimeS4", "HaronTimeL3", "HaronTimeL4",
    ],
    "SE": _RACE_KEY + [
        "Wakuban", "Umaban", "KettoNum", "Bamei", "SexCD", "Barei", "TozaiCD",
        "ChokyosiCode", "ChokyosiRyakusyo", "Futan", "Blinker",
        "KisyuCode", "KisyuRyakusyo", "MinaraiCD",
        "BaTaijyu", "ZogenFugo", "ZogenSa", "IJyoCD", "NyusenJyuni", "KakuteiJyuni",
        "DochakuKubun", "Time", "Jyuni1c", "Jyuni2c", "Jyuni3c", "Jyuni4c",
        "Odds", "Ninki", "HaronTimeL4", "HaronTimeL3", "TimeDiff",
        "DMKubun", "DMTime", "DMGosaP", "DMGosaM", "DMJyuni", "KyakusituKubun",
    ],
    "HR": _RACE_KEY + [
        "TorokuTosu", "SyussoTosu",
        "FuseirituFlag[*]", "TokubaraiFlag[*]", "HenkanFlag[*]",
        "PayUmaren[*].Kumi", "PayUmaren[*].Pay",
        "PayUmatan[*].Kumi", "PayUmatan[*].Pay",
        "PaySanrenpuku[*].Kumi", "PaySanrenpuku[*].Pay", "PaySanrenpuku[*].Ninki",
        "PaySanrentan[*].Kumi", "PaySanrentan[*].Pay", "PaySanrentan[*].Ninki",
    ],
}

# JVGets / JVRead に渡すバッファの大きさ。
#
# 公式サンプルは 110,000（最大の H6 も入る大きさ）だが、2048 にした。
# 開発者の Windows で、セットアップ用 SE ファイル（3,226件）を bench した結果:
#     110,000 → JVGets 1回 91.65ms / 2,048 → 34.09ms
# 1回あたりの時間がバッファの大きさにほぼ比例して増えるため（約0.53ms/1万バイト）。
# 残り約33msは大きさに関係しない JV-Link 側の処理で、ここでは減らせない。
#
# 2048 で足りる理由（JV-Data仕様書のレコード長、CR/LF を含む）:
#     保存する種別   RA 1,272 / SE 555 / HR 719 → 全部収まる
#     読み飛ばす種別 JG 80 / O1 962 / O2 2,042 は収まる。
#                   O3〜O6・H1・H6・WF（2,654〜102,890）は切り捨てられるが、
#                   種別を見る先頭2バイトさえ読めれば JVSkip するので問題ない
# 仕様書: size がレコード長より小さいと残りは切り捨て、最後の1バイトが NULL になる。
# 保存する種別が切り捨てられないよう、レコード長より大きいことを起動時に確かめる。
BUFFER_SIZE = 2048


def check_buffer_size(size: int) -> None:
    """保存する種別のレコードが切り捨てられない大きさか確かめる。

    仕様書: size がレコード長より小さいと切り捨て、最後の1バイトが NULL になる。
    「同じ長さ」でも最後の1バイトが NULL になりうるので、1バイト以上の余裕を求める。
    """
    need = max(RECORD_LENGTHS.values()) + 1
    if size < need:
        raise ValueError(f"バッファ {size} バイトでは RA（{RECORD_LENGTHS['RA']}バイト）が切り捨てられます。"
                         f"{need} 以上を指定してください")


class JVLinkError(RuntimeError):
    """JV-Link が負の戻り値を返したとき。"""

    def __init__(self, where: str, code: int, advice: str = ""):
        self.where, self.code = where, code
        super().__init__(f"{where} が {code} を返しました。{advice}".strip())


# JV-Link の主な戻り値と対処（インターフェース仕様書「３．コード表」より）
ERROR_ADVICE = {
    -1: "該当データなし（エラーではない）",
    -2: "セットアップダイアログでキャンセルされた",
    -111: "dataspec が不正",
    -112: "fromtime（開始時刻）が不正。YYYYMMDDhhmmss の14桁で指定",
    -113: "fromtime（終了時刻）が不正",
    -115: "option が不正",
    -116: "dataspec と option の組み合わせが不正",
    -201: "JVInit が呼ばれていない",
    -202: "前回の JVOpen に対して JVClose が呼ばれていない。JV-Link を使う他のソフトを閉じて再実行",
    -203: "JVOpen が呼ばれていない",
    -301: "認証エラー（利用キーが不正、または複数マシンで同じキーを使用）",
    -302: "利用キーの有効期限切れ",
    -303: "利用キーが未設定",
    -305: "利用規約に同意していない",
    -402: "ダウンロードしたファイルが異常（サイズ0）。該当ファイルを削除した",
    -403: "ダウンロードしたファイルが異常（データ内容）。該当ファイルを削除した",
    -502: "ダウンロード失敗（通信・ディスクエラー、サーバー混雑時のタイムアウト）。時間をおいて再実行",
    -503: "読み込むべきファイルが見つからない。JVOpen からやり直す",
}


# ---------------------------------------------------------------------------
# SDK の読み込み
# ---------------------------------------------------------------------------
def _midb2s_cp932(b: bytes, start: int, length: int) -> str:
    """SDK の MidB2S を cp932 でデコードする版に置き換えるための関数。

    引数と切り出し方は SDK と完全に同じ（1始まりの開始位置とバイト長）。
    違いはデコードだけ。cp932 は shift_jis の上位互換で、NEC/IBM 拡張文字
    （髙・﨑など）も読める。errors="replace" にしてあるので、
    それでも読めない文字があれば黙って消えずに「�」として残る。
    """
    return b[start - 1:start - 1 + length].decode("cp932", errors="replace")


def struct_path(sdk_dir: str | os.PathLike | None = None) -> Path:
    """JVData_Struct.py の場所を返す。"""
    base = Path(sdk_dir or config.JVSDK_DIR)
    return base / config.JVSDK_STRUCT_RELPATH


def load_struct_module(sdk_dir: str | os.PathLike | None = None,
                       patch_cp932: bool = True) -> ModuleType:
    """SDK の JVData_Struct.py を実行時に読み込む（リポジトリにはコピーしない）。"""
    path = struct_path(sdk_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"JV-Data 構造体が見つかりません: {path}\n"
            "SDK を展開したフォルダを環境変数 JVSDK_DIR に設定してください。\n"
            '  例) PowerShell:  $env:JVSDK_DIR = "C:\\Users\\tensu\\keiba\\JRA-VAN Data Lab. SDK Ver5.0.0_64bit"\n'
            "  中に「JV-Data構造体\\Python版\\JVData_Struct.py」があるフォルダを指定します。"
        )

    # 構造体ファイルは「後で定義されるクラス」を型注釈で先に参照している
    # （例: HON_ZEN_RUIKEISEI_INFO が後方の CHAKUKAISU6_INFO を参照）。
    # Python 3.14 は注釈を遅延評価するので通るが、3.13 以前では NameError になる。
    # ファイルは書き換えずに、`from __future__ import annotations` と同じフラグで
    # コンパイルして注釈を文字列のまま保持させる。dataclass はこれで問題なく動く。
    source = path.read_text(encoding="utf-8-sig")
    code = compile(source, str(path), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    module = ModuleType("JVData_Struct")
    module.__file__ = str(path)
    sys.modules["JVData_Struct"] = module  # dataclass が型の解決で参照するため
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop("JVData_Struct", None)
        raise

    missing = [c for c in RECORD_STRUCTS.values() if not hasattr(module, c)]
    if missing:
        raise ImportError(f"JVData_Struct.py に {missing} がありません（SDK のバージョン違い？）")

    if patch_cp932:
        module.MidB2S = _midb2s_cp932
    return module


# ---------------------------------------------------------------------------
# 構造体 -> 平たい dict
# ---------------------------------------------------------------------------
_LIST_PATTERN = re.compile(r"^(\w+)\[\*\]$")


def _expand(obj, path: str, prefix: str = "") -> list[tuple[str, object]]:
    """属性パスをたどって (列名, 値) のリストを返す。"[*]" はリストを展開する。"""
    head, _, rest = path.partition(".")
    m = _LIST_PATTERN.match(head)
    if m:
        name = m.group(1)
        items = getattr(obj, name)
        out = []
        for i, item in enumerate(items, start=1):  # 列名は1始まり
            col = f"{prefix}{name}[{i}]"
            out += _expand(item, rest, col + ".") if rest else [(col, item)]
        return out

    value = getattr(obj, head)
    col = f"{prefix}{head}"
    return _expand(value, rest, col + ".") if rest else [(col, value)]


def columns_for(record_type: str, struct_module: ModuleType) -> list[str]:
    """保存する列名の一覧（空のレコードを1件パースして列名を確定させる）。"""
    dummy = b" " * RECORD_LENGTHS[record_type]
    obj = getattr(struct_module, RECORD_STRUCTS[record_type]).SetDataB(dummy)
    return [col for path in FIELDS[record_type] for col, _ in _expand(obj, path)]


def parse_record(raw: bytes, struct_module: ModuleType) -> tuple[str, dict] | None:
    """1レコードのバイト列を (レコード種別, {列名: 値}) にする。対象外なら None。"""
    record_type = raw[:2].decode("ascii", errors="replace")
    cls_name = RECORD_STRUCTS.get(record_type)
    if cls_name is None:
        return None
    obj = getattr(struct_module, cls_name).SetDataB(raw)
    row = {}
    for path in FIELDS[record_type]:
        for col, value in _expand(obj, path):
            # 文字列の前後の空白は落とす（固定長フィールドの詰め物）
            row[col] = value.strip() if isinstance(value, str) else value
    return record_type, row


# ---------------------------------------------------------------------------
# COM のラッパー
# ---------------------------------------------------------------------------
@dataclass
class OpenResult:
    code: int
    read_count: int = 0
    download_count: int = 0
    last_file_timestamp: str = ""


class JVLinkClient:
    """JV-Link（COM）を薄く包む。テストでは偽物の com を渡して差し替える。"""

    def __init__(self, com=None, buffer_size: int = None, method: str = "gets"):
        """
        buffer_size : JVGets/JVRead に渡すバッファの大きさ（既定 BUFFER_SIZE の説明を参照）
        method      : "gets"（JVGets・既定）か "read"（JVRead）。
                      JVRead は JV-Link の中で SJIS → Unicode 変換をして文字列で返す。
                      こちらでは cp932 でバイト列に戻してから構造体に通すので、
                      読めない文字があると位置がずれうる。**速さの比較（bench）用**。
        """
        self.buffer_size = buffer_size or BUFFER_SIZE
        check_buffer_size(self.buffer_size)
        if method not in ("gets", "read"):
            raise ValueError(f"method は 'gets' か 'read': {method!r}")
        self.method = method
        if com is None:
            try:
                import win32com.client  # Windows + pywin32 でのみ動く
            except ImportError as e:
                raise RuntimeError(
                    "pywin32 が見つかりません。JV-Link は Windows 専用です。"
                    "Windows で `py -m pip install pywin32` を実行してください。"
                ) from e
            com = win32com.client.Dispatch("JVDTLab.JVLink")
        self.jv = com

    def init(self, sid: str = "UNKNOWN") -> None:
        code = int(self.jv.JVInit(sid))
        if code != 0:
            raise JVLinkError("JVInit", code, ERROR_ADVICE.get(code, ""))

    def open(self, dataspec: str, fromtime: str, option: int) -> OpenResult:
        """JVOpen。公式サンプルと同じく6引数で呼び、タプル／単値の両方に対応する。"""
        ret = self.jv.JVOpen(dataspec, fromtime, option, 0, 0, "")
        if isinstance(ret, (list, tuple)):
            return OpenResult(int(ret[0]), int(ret[1] or 0), int(ret[2] or 0), str(ret[3] or ""))
        return OpenResult(int(ret))

    def status(self) -> int:
        return int(self.jv.JVStatus())

    def gets(self) -> tuple[int, bytes, str]:
        """1レコード読む。戻り値は (コード, データ本体のバイト列, ファイル名)。"""
        if self.method == "read":
            return self._read()
        buff = bytearray(self.buffer_size)
        ret, memview, fname = self.jv.JVGets(buff, self.buffer_size, bytearray())
        code = int(ret)
        raw = b""
        if code > 0 and memview is not None:
            data = memview.tobytes() if hasattr(memview, "tobytes") else bytes(memview)
            raw = data[:code]  # 戻り値 = バッファにセットしたバイト数
        return code, raw, str(fname or "")

    def _read(self) -> tuple[int, bytes, str]:
        """JVRead 版。公式サンプルのコメントと同じく (戻り値, 文字列, サイズ, ファイル名) を受け取る。"""
        ret, text, _size, fname = self.jv.JVRead("", self.buffer_size, "")
        code = int(ret)
        raw = b""
        if code > 0 and text:
            # JV-Link が Unicode にしたものを、構造体が期待する SJIS のバイト列に戻す
            raw = str(text).encode("cp932", errors="replace")[:code]
        return code, raw, str(fname or "")

    def skip(self) -> None:
        self.jv.JVSkip()

    def delete_file(self, fname: str) -> None:
        try:
            self.jv.JVFiledelete(fname)
        except Exception:  # 削除に失敗しても元のエラーの方が重要なので握りつぶす
            pass

    def close(self) -> None:
        try:
            self.jv.JVClose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CSV への追記
# ---------------------------------------------------------------------------
class CsvAppender:
    """レコード種別ごとの CSV に、一定件数ずつまとめて追記する。

    40年分の SE は数百万行になるので、全部メモリに溜めずに流し込む。
    文字コードは utf-8-sig（Windows の Excel でも文字化けしない）。
    """

    def __init__(self, path: Path, columns: list[str], flush_every: int = 20_000):
        self.path, self.columns, self.flush_every = path, columns, flush_every
        self.buffer: list[list] = []
        self.written = 0

    def append(self, row: dict) -> None:
        self.buffer.append([row.get(c, "") for c in self.columns])
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        new_file = not self.path.exists()
        if not new_file:
            self._check_header()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.columns)
            writer.writerows(self.buffer)
        self.written += len(self.buffer)
        self.buffer = []

    def _check_header(self) -> None:
        """既存 CSV の列と今回の列が違えば止める（FIELDS を変えたまま追記すると壊れるため）。"""
        with open(self.path, encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if header != self.columns:
            raise ValueError(
                f"{self.path} の列構成が今回の設定と違います。"
                "FIELDS を変更した場合は既存の CSV を別名に退避してから取り直してください。"
            )


# ---------------------------------------------------------------------------
# 取得の本体
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    dataspec: str
    fromtime: str
    option: int
    open_code: int
    read_count: int = 0
    download_count: int = 0
    last_file_timestamp: str = ""
    records: dict = field(default_factory=dict)       # 種別ごとの保存件数
    skipped_files: dict = field(default_factory=dict)  # JVSkip した種別ごとのファイル数
    files_switched: int = 0
    short_records: int = 0
    files_done: int = 0       # 今回読み終えたファイル数
    data_files: int = 0       # そのうち保存対象（RA/SE/HR）のファイル数
    resumed_files: int = 0    # 前回までに読み終えていたので飛ばしたファイル数
    reopened: int = 0         # エラーで JVOpen し直した回数
    minus3: int = 0           # JVGets が -3 を返した回数
    stopped_early: bool = False  # max_data_files で途中終了した（bench）
    interrupted: bool = False    # Ctrl+C で止めた
    timings: list = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"JVOpen: code={self.open_code} 読込ファイル={self.read_count} "
            f"ダウンロード={self.download_count} 最新ファイル時刻={self.last_file_timestamp}",
            f"保存レコード: {self.records}",
            f"JVSkipしたファイル（種別ごと）: {self.skipped_files}",
            f"再開で飛ばしたファイル: {self.resumed_files}  開き直し: {self.reopened}回  "
            f"-3（ダウンロード中）: {self.minus3}回",
            f"短すぎるレコード: {self.short_records}  所要: {self.seconds:.0f}秒",
        ]
        return "\n".join(lines)


def _state_path(out_dir: Path) -> Path:
    return out_dir / "state.json"


def load_state(out_dir: str | os.PathLike | None = None) -> dict:
    path = _state_path(Path(out_dir or config.JV_DATA_DIR))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(out_dir: Path, dataspec: str, result: FetchResult) -> None:
    state = load_state(out_dir)
    state[dataspec] = {
        "last_file_timestamp": result.last_file_timestamp,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "fromtime": result.fromtime,
        "option": result.option,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _state_path(out_dir).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


class _ignore_ctrl_c:
    """with の中だけ Ctrl+C（SIGINT）を無視する。後始末を最後まで通すため。

    signal はメインスレッドでしか差し替えられないので、それ以外では何もしない。
    """

    def __enter__(self):
        import signal
        self.previous = None
        try:
            self.previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            pass
        return self

    def __exit__(self, *exc):
        import signal
        if self.previous is not None:
            signal.signal(signal.SIGINT, self.previous)
        return False


def wait_for_download(client: JVLinkClient, download_count: int, poll_sec: float = 0.5,
                      timeout_sec: float = 3600, sleep=time.sleep, log=print) -> None:
    """JVStatus がダウンロード対象数に達するまで待つ。

    仕様書: 「ダウンロード処理の完了を待たず JVRead/JVGets を呼び出すと
    予期しないエラーが発生する場合があります」。サンプルもここで待っている。
    """
    if download_count <= 0:
        return
    waited, last_shown = 0.0, -1
    while True:
        done = client.status()
        if done < 0:
            raise JVLinkError("JVStatus", done, ERROR_ADVICE.get(done, ""))
        if done != last_shown:
            log(f"  ダウンロード {done}/{download_count}")
            last_shown = done
        if done >= download_count:
            return
        if waited >= timeout_sec:
            raise TimeoutError(f"ダウンロードが {timeout_sec:.0f} 秒で終わりませんでした（{done}/{download_count}）")
        sleep(poll_sec)
        waited += poll_sec


class Progress:
    """読み終えたファイル名を1行ずつ記録する（中断からの再開用）。

    セットアップは数時間かかるので、途中で止まったときに最初から読み直したくない。
    仕様書の「セットアップデータの中断・再開」は「最後に読み込んだファイル名を保持し、
    同じパラメータで JVOpen して、そのファイルまで JVSkip する」方式。
    ここではそれを少し一般化して、読み終えたファイル名の集合を持ち、
    再開時に集合に含まれるファイルを JVSkip で飛ばす。

    記録は「CSV への書き出し（flush）が終わってから」行う。
    先にファイル名を記録すると、停電などで書き出し前のレコードが消えたのに
    「読み終えた」扱いになり、そのファイルのデータが欠けたままになるため。

    進捗ファイルは (dataspec, option, fromtime) ごとに別。完走したら消す。
    """

    def __init__(self, out_dir: Path, dataspec: str, option: int, fromtime: str):
        safe = re.sub(r"[^0-9A-Za-z_-]", "_", f"{dataspec}_{option}_{fromtime}")
        self.path = out_dir / f"progress_{safe}.txt"
        self.done: set[str] = set()
        if self.path.exists():
            self.done = {line.strip() for line in self.path.read_text(encoding="utf-8").splitlines()
                         if line.strip()}

    def __contains__(self, fname: str) -> bool:
        return bool(fname) and fname in self.done

    def mark(self, fname: str) -> None:
        if not fname or fname in self.done:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(fname + "\n")
        self.done.add(fname)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


# JVOpen からやり直せば解消しうるエラー（仕様書の対処に「JVOpen からやりなおす」とあるもの）
REOPEN_CODES = {-402, -403, -502, -503}


@dataclass
class FileTiming:
    """1ファイルを読むのにかかった時間の内訳（遅い原因を切り分けるため）。

    wall      : そのファイルの最初のレコードから読み終わりまでの実時間
    gets      : JVGets の呼び出しにかかった時間の合計（JV-Link の中の処理を含む）
    wait_m3   : -3（ダウンロード中）で待った時間の合計
    parse     : 構造体のパース（SDK の SetDataB）と列の取り出し
    write     : CSV への追記（バッファに積む＋書き出し）
    first_ms / last_ms : そのファイルの最初の100件・最後の100件の JVGets 1回あたりの平均（ミリ秒）
        後半ほど遅いなら、読む位置に比例して遅くなっている（ファイル内で先頭から
        たどり直しているような動き）ことが分かる。
    """

    fname: str
    record_type: str = ""
    records: int = 0
    gets_calls: int = 0
    minus3: int = 0
    wall: float = 0.0
    gets: float = 0.0
    wait_m3: float = 0.0
    parse: float = 0.0
    write: float = 0.0
    started: float = 0.0
    _first: list = field(default_factory=list, repr=False)
    _last: list = field(default_factory=list, repr=False)

    def add_gets(self, sec: float) -> None:
        self.gets += sec
        self.gets_calls += 1
        if len(self._first) < 100:
            self._first.append(sec)
        self._last.append(sec)
        if len(self._last) > 100:
            self._last.pop(0)

    @property
    def first_ms(self) -> float:
        return 1000 * sum(self._first) / len(self._first) if self._first else 0.0

    @property
    def last_ms(self) -> float:
        return 1000 * sum(self._last) / len(self._last) if self._last else 0.0

    def line(self, index: int, total: int) -> str:
        other = self.wall - self.gets - self.wait_m3 - self.parse - self.write
        return (f"  [{index:>5}/{total}] {self.fname} {self.record_type} {self.records:,}件 "
                f"{self.wall:.1f}秒 = JVGets {self.gets:.1f} / -3待ち {self.wait_m3:.1f}({self.minus3}回)"
                f" / 解析 {self.parse:.1f} / 書出 {self.write:.1f} / その他 {other:.1f}"
                f"  [JVGets 1回: 前半{self.first_ms:.1f}ms 後半{self.last_ms:.1f}ms]")

    def row(self) -> dict:
        return {"file": self.fname, "type": self.record_type, "records": self.records,
                "gets_calls": self.gets_calls, "wall_sec": round(self.wall, 3),
                "gets_sec": round(self.gets, 3), "minus3_count": self.minus3,
                "minus3_wait_sec": round(self.wait_m3, 3), "parse_sec": round(self.parse, 3),
                "write_sec": round(self.write, 3), "gets_first100_ms": round(self.first_ms, 3),
                "gets_last100_ms": round(self.last_ms, 3)}


class TimingLog:
    """ファイルごとの計時を CSV に追記し、画面にも1行ずつ出す。"""

    COLUMNS = ["logged_at", "file", "type", "records", "gets_calls", "wall_sec", "gets_sec",
               "minus3_count", "minus3_wait_sec", "parse_sec", "write_sec",
               "gets_first100_ms", "gets_last100_ms"]

    def __init__(self, path: Path | None, log=print, show_min_sec: float = 0.0):
        self.path, self.log, self.show_min_sec = path, log, show_min_sec
        self.rows: list[dict] = []

    def add(self, t: FileTiming, index: int, total: int) -> None:
        row = {"logged_at": datetime.now().isoformat(timespec="seconds"), **t.row()}
        self.rows.append(row)
        if t.wall >= self.show_min_sec:
            self.log(t.line(index, total))
        if self.path is None:
            return
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)


def fetch(dataspec: str = "RACE", fromtime: str | None = None, option: int = 1,
          out_dir: str | os.PathLike | None = None,
          record_types: tuple[str, ...] = ("RA", "SE", "HR"),
          client: JVLinkClient | None = None, struct_module: ModuleType | None = None,
          sid: str = "UNKNOWN", retry_sleep_sec: float = 1.0,
          max_retry_sec: float = 600, max_reopen: int = 3, reopen_wait_sec: float = 10.0,
          sleep=time.sleep, log=print, progress_every: int = 50_000,
          buffer_size: int | None = None, parse: bool = True,
          max_data_files: int | None = None, timing_path: str | os.PathLike | None = "auto",
          method: str = "gets", clock=time.perf_counter) -> FetchResult:
    """JV-Link から取得して、レコード種別ごとの CSV に追記する。

    Parameters
    ----------
    fromtime : "YYYYMMDDhhmmss"。省略時は前回保存した最新ファイル時刻（なければ30日前）。
               "開始-終了" 形式で終了時刻も指定できる（RACE は可。仕様書 JVOpen の項）。
               ※ これは**データの提供時刻**であってレース日ではない。
                 レース日で絞るのは読み込んだ後（jvmap 側）で行う。
    option   : 1=通常（差分更新） / 2=今週 / 4=ダイアログ無しセットアップ（初回のみダイアログ）
    retry_sleep_sec : -3（ダウンロード中）のときに待つ時間の**上限**。
               0.05秒から始めて、続くたびに倍にしていき、この値で頭打ちにする。
    max_reopen : -402/-403/-502/-503 のとき、自動で JVClose → JVOpen し直す回数。
                 読み終えたファイルは飛ばすので、やり直しても最初から読み直しにはならない。
    buffer_size : JVGets に渡すバッファの大きさ（既定 2048。BUFFER_SIZE の説明を参照）。
    method   : "gets"（既定）/ "read"（JVRead。速さの比較用）。
    parse    : False ならパースも CSV 書き出しもしない（JVGets だけの速さを測る bench 用）。
    max_data_files : 保存対象のファイルをこの数だけ読んだら止める（bench 用）。
    timing_path : ファイルごとの所要時間の記録先。"auto" なら out_dir/fetch_timing.csv。

    中断からの再開:
      同じ dataspec / fromtime / option でもう一度実行すると、読み終えたファイルを
      JVSkip で飛ばして続きから読む（Progress を参照）。
    """
    out_dir = Path(out_dir or config.JV_DATA_DIR)
    if parse:
        struct_module = struct_module or load_struct_module()
    client = client or JVLinkClient(buffer_size=buffer_size, method=method)
    if buffer_size:
        check_buffer_size(buffer_size)
        client.buffer_size = buffer_size
    client.method = method

    if fromtime is None:
        prev = load_state(out_dir).get(dataspec, {}).get("last_file_timestamp")
        fromtime = prev or (datetime.now() - timedelta(days=30)).strftime("%Y%m%d000000")

    writers = ({rt: CsvAppender(out_dir / f"{rt}.csv", ["_seq"] + columns_for(rt, struct_module))
                for rt in record_types} if parse else {})
    # 追記する CSV の通し番号（同じ作成日のレコードの前後関係を決めるのに使う）
    counter = {"seq": _next_seq(out_dir, record_types) if parse else 0, "total": 0}
    progress = Progress(out_dir, dataspec, option, fromtime)
    if progress.done:
        log(f"前回の続きから再開します（読み終えたファイル {len(progress.done)} 個を飛ばす）")
    if timing_path == "auto":
        timing_path = out_dir / "fetch_timing.csv"
    # 画面には1秒以上かかったファイルだけ出す（週次の差分更新は小さいファイルが多く、
    # 全部出すと読みにくい）。CSV には全ファイルを記録する。
    timing = TimingLog(Path(timing_path) if timing_path else None, log=log, show_min_sec=1.0)

    result = FetchResult(dataspec, fromtime, option, open_code=0)
    started = time.time()
    client.init(sid)
    try:
        attempt = 0
        while True:
            try:
                _read_all(client, dataspec, fromtime, option, writers, struct_module,
                          progress, counter, result, retry_sleep_sec, max_retry_sec,
                          sleep, log, progress_every, set(record_types), parse,
                          max_data_files, timing, clock)
                break
            except JVLinkError as e:
                if e.code not in REOPEN_CODES or attempt >= max_reopen:
                    raise
                attempt += 1
                result.reopened += 1
                log(f"  {e}\n  → JVClose して開き直します（{attempt}/{max_reopen}回目）")
                for w in writers.values():
                    w.flush()
                client.close()
                sleep(reopen_wait_sec)

        result.timings = timing.rows
        if result.open_code != -1 and not result.stopped_early:
            for w in writers.values():
                w.flush()
            _save_state(out_dir, dataspec, result)
            progress.clear()  # 完走したので再開用の記録は不要
        return result
    except KeyboardInterrupt:
        result.interrupted = True
        log("\n中断を受け付けました（Ctrl+C）。書けた分を保存して JVClose します。"
            "もう一度 Ctrl+C を押さずにお待ちください…")
        raise
    finally:
        # 途中で例外が出ても（Ctrl+C を含む）、書けた分は残し、必ず JVClose する。
        # 後始末の最中に2回目の Ctrl+C が来ると、ここが途中で打ち切られてしまうので、
        # 後始末の間だけ Ctrl+C を無視する。
        with _ignore_ctrl_c():
            flushed = 0
            for w in writers.values():
                try:
                    flushed += len(w.buffer)
                    w.flush()
                except Exception as e:  # 書き出しに失敗しても JVClose は必ず呼ぶ
                    log(f"  CSV の書き出しに失敗: {e}")
            client.close()
            result.seconds = time.time() - started
            if result.interrupted:
                log(f"  CSV に {flushed:,} 件を書き出し、JVClose しました。"
                    f"読み終えたファイル {len(progress.done):,} 個は {progress.path.name} に記録済み。\n"
                    "  同じコマンドを再実行すれば続きから読みます。")


def _read_all(client, dataspec, fromtime, option, writers, struct_module, progress,
              counter, result, retry_sleep_sec, max_retry_sec, sleep, log, progress_every,
              wanted, parse, max_data_files, timing, clock):
    """JVOpen から EOF まで読む（1回分）。"""
    log(f"JVOpen({dataspec!r}, {fromtime!r}, option={option})")
    opened = client.open(dataspec, fromtime, option)
    result.open_code = opened.code
    result.read_count = opened.read_count
    result.download_count = opened.download_count
    result.last_file_timestamp = opened.last_file_timestamp

    if opened.code == -1:
        log("該当データなし（エラーではありません）")
        return
    if opened.code < 0:
        raise JVLinkError("JVOpen", opened.code, ERROR_ADVICE.get(opened.code, ""))
    log(f"  読込対象ファイル {opened.read_count} / ダウンロード {opened.download_count}")

    wait_for_download(client, opened.download_count, sleep=sleep, log=log)
    log("  読み込み開始（1秒以上かかった保存対象ファイルは、読み終えるたびに所要時間を表示）")

    cur: FileTiming | None = None   # いま読んでいる保存対象ファイルの計時

    def finish(fname: str) -> None:
        """1ファイル読み終わり：先に CSV へ書き出してから、読み終えたと記録する。"""
        nonlocal cur
        if not fname:
            return
        t0 = clock()
        for w in writers.values():
            w.flush()
        progress.mark(fname)
        result.files_done += 1
        if cur is not None and cur.fname == fname:
            cur.write += clock() - t0
            cur.wall = clock() - cur.started
            result.data_files += 1
            timing.add(cur, result.files_done + result.resumed_files, opened.read_count)
        cur = None

    current = ""     # いま読んでいるファイル名
    m3_streak = 0    # -3 が何回続いたか（待ち時間を伸ばすのに使う）
    waited = 0.0
    pending_m3 = (0, 0.0)  # レコードを読む前の -3（どのファイルの待ちか、次のレコードで確定する）
    while True:
        t0 = clock()
        code, raw, fname = client.gets()
        dt = clock() - t0

        if code > 0:
            waited, m3_streak = 0.0, 0
            if fname != current:          # 次のファイルに入った
                finish(current)
                current = fname
            if fname in progress:         # 前回までに読み終えたファイル
                result.resumed_files += 1
                client.skip()
                current = ""
                pending_m3 = (0, 0.0)
                continue
            record_type = raw[:2].decode("ascii", errors="replace")
            if record_type not in wanted:
                # 1ファイル=1レコード種別なので、残りは読まずに次のファイルへ
                result.skipped_files[record_type] = result.skipped_files.get(record_type, 0) + 1
                client.skip()
                finish(current)
                current = ""
                pending_m3 = (0, 0.0)
                continue
            if cur is None:
                cur = FileTiming(fname, record_type, started=t0)
                cur.minus3, cur.wait_m3 = pending_m3
                pending_m3 = (0, 0.0)
            cur.add_gets(dt)
            cur.records += 1

            if len(raw) < RECORD_LENGTHS[record_type] - 2:
                result.short_records += 1
            if parse:
                t1 = clock()
                parsed = parse_record(raw, struct_module)
                t2 = clock()
                cur.parse += t2 - t1
                if parsed is not None:
                    _, row = parsed
                    row["_seq"] = counter["seq"]
                    counter["seq"] += 1
                    writers[record_type].append(row)
                    cur.write += clock() - t2
            result.records[record_type] = result.records.get(record_type, 0) + 1
            counter["total"] += 1
            if counter["total"] % progress_every == 0:
                log(f"  {counter['total']:,} 件  {result.records}  "
                    f"（ファイル {result.files_done + result.resumed_files}/{opened.read_count}）")

        elif code == -1:          # ファイルの切り替わり。エラーではない
            result.files_switched += 1
            finish(current)
            current = ""
            if max_data_files is not None and result.data_files >= max_data_files:
                result.stopped_early = True
                return
        elif code == 0:           # 全ファイル読み終わり
            finish(current)
            return
        elif code == -3:          # ダウンロード中。待って再試行（サンプルはここで止まる）
            if waited >= max_retry_sec:
                raise JVLinkError("JVGets", code,
                                  f"{max_retry_sec:.0f} 秒待ってもダウンロードが終わりません")
            # 最初は 0.05 秒、続くたびに倍にして retry_sleep_sec で頭打ち。
            # 1秒固定だと、すぐ終わる待ちでも毎回1秒捨てることになるため。
            wait = min(0.05 * (2 ** m3_streak), retry_sleep_sec)
            m3_streak += 1
            sleep(wait)
            waited += wait
            result.minus3 += 1
            if cur is not None:
                cur.minus3 += 1
                cur.wait_m3 += wait + dt
            else:
                pending_m3 = (pending_m3[0] + 1, pending_m3[1] + wait + dt)
        elif code in (-402, -403):
            client.delete_file(fname)
            raise JVLinkError("JVGets", code, ERROR_ADVICE[code] + f"（{fname}）")
        else:
            raise JVLinkError("JVGets", code, ERROR_ADVICE.get(code, ""))


def _next_seq(out_dir: Path, record_types) -> int:
    """既存 CSV の _seq の最大値 + 1 を返す（追記しても順序が保たれるように）。"""
    best = -1
    for rt in record_types:
        path = out_dir / f"{rt}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if "_seq" not in header:
                continue
            idx = header.index("_seq")
            for row in reader:
                try:
                    best = max(best, int(row[idx]))
                except (ValueError, IndexError):
                    pass
    return best + 1


# ---------------------------------------------------------------------------
# コマンドライン（Windows で 1 コマンドずつ確認するため）
# ---------------------------------------------------------------------------
def _cmd_check(args) -> int:
    print(f"Python      : {sys.version.split()[0]} ({'64bit' if sys.maxsize > 2**32 else '32bit'})")
    print(f"JVSDK_DIR   : {config.JVSDK_DIR}")
    path = struct_path()
    print(f"構造体      : {path}  存在={path.is_file()}")
    module = load_struct_module()
    for rt, cls in RECORD_STRUCTS.items():
        print(f"  {rt}: {cls}  保存列数={len(columns_for(rt, module))}")
    print(f"保存先      : {config.JV_DATA_DIR}")
    if args.no_com:
        return 0
    client = JVLinkClient()
    client.init(args.sid)
    print("JVInit      : 0（正常）")
    return 0


def _cmd_fetch(args) -> int:
    try:
        result = fetch(dataspec=args.dataspec, fromtime=args.fromtime, option=args.option,
                       out_dir=args.out, buffer_size=args.buffer_size)
    except KeyboardInterrupt:
        return 130  # 後始末とメッセージは fetch の中で済んでいる。トレースバックは出さない
    print(result.summary())
    return 0


def _cmd_bench(args) -> int:
    """遅い原因を切り分けるため、先頭の数ファイルだけを読んで内訳を出す。

    本番の CSV・進捗とは別のフォルダ（data/jvlink/bench/日時_条件）に書くので、
    本番のセットアップの再開記録には影響しない。JV-Link を同時に2つ開くのは
    避けたいので、本番のセットアップは止めてから実行すること。
    """
    import cProfile
    import io
    import pstats

    label = (f"{args.method}_buf{args.buffer_size or BUFFER_SIZE}"
             + ("" if not args.no_parse else "_noparse"))
    out = Path(args.out or config.JV_DATA_DIR) / "bench" / f"{datetime.now():%Y%m%d_%H%M%S}_{label}"
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    print(f"bench: 読み方={'JVRead' if args.method == 'read' else 'JVGets'} "
          f"種別={types} ファイル数={args.files} バッファ={args.buffer_size or BUFFER_SIZE} "
          f"解析={'しない' if args.no_parse else 'する'}  出力先={out}")

    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    result = fetch(dataspec=args.dataspec, fromtime=args.fromtime, option=args.option,
                   out_dir=out, record_types=types, buffer_size=args.buffer_size,
                   parse=not args.no_parse, max_data_files=args.files, method=args.method)
    if prof:
        prof.disable()

    print(result.summary())
    rows = result.timings
    if rows:
        recs = sum(r["records"] for r in rows)
        gets = sum(r["gets_sec"] for r in rows)
        wall = sum(r["wall_sec"] for r in rows)
        print(f"\n合計: {recs:,}件 {wall:.1f}秒  JVGets {gets:.1f}秒"
              f"（1件あたり {1000 * gets / max(recs, 1):.2f}ms）"
              f"  解析 {sum(r['parse_sec'] for r in rows):.1f}秒"
              f"  書出 {sum(r['write_sec'] for r in rows):.1f}秒"
              f"  -3待ち {sum(r['minus3_wait_sec'] for r in rows):.1f}秒"
              f"（{sum(r['minus3_count'] for r in rows)}回）")
    if prof:
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(15)
        print("\n--- cProfile（関数そのものにかかった時間の上位15）---")
        print(buf.getvalue())
    return 0


def _cmd_summary(args) -> int:
    import pandas as pd

    out = Path(args.out or config.JV_DATA_DIR)
    for rt in ("RA", "SE", "HR"):
        path = out / f"{rt}.csv"
        if not path.exists():
            print(f"[{rt}] なし")
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        print(f"\n[{rt}] {len(df):,} 行  開催年 {df['id.Year'].min()}〜{df['id.Year'].max()}")
        print("  データ区分:", df["head.DataKubun"].value_counts().sort_index().to_dict())
        print("  競馬場コード:", df["id.JyoCD"].value_counts().sort_index().to_dict())
        if rt == "RA":
            print("  グレードコード:", df["GradeCD"].value_counts().sort_index().to_dict())
            print("  トラックコード:", df["TrackCD"].value_counts().sort_index().to_dict())
            # 取得期間より明らかに古い開催年のレコード（過去レースの訂正など）を一覧する
            newest = pd.to_numeric(df["id.Year"], errors="coerce").max()
            old = df.loc[pd.to_numeric(df["id.Year"], errors="coerce") < newest - 1]
            if len(old):
                print(f"  開催年が古いレコード {len(old)} 件（データ作成日が新しければ過去レースの訂正）:")
                show = old.assign(
                    作成日=old["head.MakeDate.Year"] + old["head.MakeDate.Month"] + old["head.MakeDate.Day"])
                cols = ["id.Year", "id.MonthDay", "id.JyoCD", "id.RaceNum", "head.DataKubun",
                        "作成日", "GradeCD", "RaceInfo.Hondai"]
                print(show[cols].head(20).to_string(index=False))
        if rt == "SE":
            print("  異常区分:", df["IJyoCD"].value_counts().sort_index().to_dict())
            print("  脚質判定:", df["KyakusituKubun"].value_counts().sort_index().to_dict())
    state = load_state(out)
    if state:
        print("\nstate.json:", json.dumps(state, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m src.jvlink", description="JV-Link からの取得")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="SDK と JV-Link の疎通確認（データは取らない）")
    p.add_argument("--sid", default="UNKNOWN")
    p.add_argument("--no-com", action="store_true", help="SDK の読み込みだけ確認する")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("fetch", help="RA/SE/HR を取得して CSV に追記")
    p.add_argument("--dataspec", default="RACE")
    p.add_argument("--from", dest="fromtime", default=None,
                   help="YYYYMMDDhhmmss（省略時は前回の続き。初回は30日前）")
    p.add_argument("--option", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--buffer-size", type=int, default=None,
                   help="JVGets に渡すバッファの大きさ（既定 2048）")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("bench", help="先頭の数ファイルだけ読んで、遅い原因の内訳を測る")
    p.add_argument("--dataspec", default="RACE")
    p.add_argument("--from", dest="fromtime", default="19860101000000")
    p.add_argument("--option", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--types", default="SE", help="保存対象の種別（カンマ区切り）。既定 SE")
    p.add_argument("--files", type=int, default=1, help="保存対象のファイルを何個読んだら止めるか")
    p.add_argument("--buffer-size", type=int, default=None, help="バッファの大きさ（既定 2048）")
    p.add_argument("--method", default="gets", choices=["gets", "read"],
                   help="gets=JVGets（既定）/ read=JVRead（速さの比較用）")
    p.add_argument("--no-parse", action="store_true", help="パースと CSV 書き出しをしない")
    p.add_argument("--profile", action="store_true", help="cProfile で関数ごとの時間も出す")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("summary", help="保存済み CSV の中身を集計して表示")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
