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

# JVGets のバッファサイズ。公式サンプルと同じ値（最大レコードの O6 も収まる）
BUFFER_SIZE = 110_000


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

    def __init__(self, com=None):
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
        """JVGets。戻り値は (コード, データ本体のバイト列, ファイル名)。"""
        buff = bytearray(BUFFER_SIZE)
        ret, memview, fname = self.jv.JVGets(buff, BUFFER_SIZE, bytearray())
        code = int(ret)
        raw = b""
        if code > 0 and memview is not None:
            data = memview.tobytes() if hasattr(memview, "tobytes") else bytes(memview)
            raw = data[:code]  # 戻り値 = バッファにセットしたバイト数
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
    resumed_files: int = 0    # 前回までに読み終えていたので飛ばしたファイル数
    reopened: int = 0         # エラーで JVOpen し直した回数
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"JVOpen: code={self.open_code} 読込ファイル={self.read_count} "
            f"ダウンロード={self.download_count} 最新ファイル時刻={self.last_file_timestamp}",
            f"保存レコード: {self.records}",
            f"JVSkipしたファイル（種別ごと）: {self.skipped_files}",
            f"再開で飛ばしたファイル: {self.resumed_files}  開き直し: {self.reopened}回",
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


def fetch(dataspec: str = "RACE", fromtime: str | None = None, option: int = 1,
          out_dir: str | os.PathLike | None = None,
          record_types: tuple[str, ...] = ("RA", "SE", "HR"),
          client: JVLinkClient | None = None, struct_module: ModuleType | None = None,
          sid: str = "UNKNOWN", retry_sleep_sec: float = 1.0,
          max_retry_sec: float = 600, max_reopen: int = 3, reopen_wait_sec: float = 10.0,
          sleep=time.sleep, log=print, progress_every: int = 50_000) -> FetchResult:
    """JV-Link から取得して、レコード種別ごとの CSV に追記する。

    Parameters
    ----------
    fromtime : "YYYYMMDDhhmmss"。省略時は前回保存した最新ファイル時刻（なければ30日前）。
               "開始-終了" 形式で終了時刻も指定できる（RACE は可。仕様書 JVOpen の項）。
               ※ これは**データの提供時刻**であってレース日ではない。
                 レース日で絞るのは読み込んだ後（jvmap 側）で行う。
    option   : 1=通常（差分更新） / 2=今週 / 4=ダイアログ無しセットアップ（初回のみダイアログ）
    max_reopen : -402/-403/-502/-503 のとき、自動で JVClose → JVOpen し直す回数。
                 読み終えたファイルは飛ばすので、やり直しても最初から読み直しにはならない。

    中断からの再開:
      同じ dataspec / fromtime / option でもう一度実行すると、読み終えたファイルを
      JVSkip で飛ばして続きから読む（Progress を参照）。
    """
    out_dir = Path(out_dir or config.JV_DATA_DIR)
    struct_module = struct_module or load_struct_module()
    client = client or JVLinkClient()

    if fromtime is None:
        prev = load_state(out_dir).get(dataspec, {}).get("last_file_timestamp")
        fromtime = prev or (datetime.now() - timedelta(days=30)).strftime("%Y%m%d000000")

    writers = {rt: CsvAppender(out_dir / f"{rt}.csv", ["_seq"] + columns_for(rt, struct_module))
               for rt in record_types}
    # 追記する CSV の通し番号（同じ作成日のレコードの前後関係を決めるのに使う）
    counter = {"seq": _next_seq(out_dir, record_types), "total": 0}
    progress = Progress(out_dir, dataspec, option, fromtime)
    if progress.done:
        log(f"前回の続きから再開します（読み終えたファイル {len(progress.done)} 個を飛ばす）")

    result = FetchResult(dataspec, fromtime, option, open_code=0)
    started = time.time()
    client.init(sid)
    try:
        attempt = 0
        while True:
            try:
                _read_all(client, dataspec, fromtime, option, writers, struct_module,
                          progress, counter, result, retry_sleep_sec, max_retry_sec,
                          sleep, log, progress_every)
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

        if result.open_code != -1:
            for w in writers.values():
                w.flush()
            _save_state(out_dir, dataspec, result)
            progress.clear()  # 完走したので再開用の記録は不要
        return result
    finally:
        # 途中で例外が出ても（Ctrl+C を含む）、書けた分は残し、必ず JVClose する
        for w in writers.values():
            try:
                w.flush()
            except Exception:
                pass
        client.close()
        result.seconds = time.time() - started


def _read_all(client, dataspec, fromtime, option, writers, struct_module, progress,
              counter, result, retry_sleep_sec, max_retry_sec, sleep, log, progress_every):
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

    def finish(fname: str) -> None:
        """1ファイル読み終わり：先に CSV へ書き出してから、読み終えたと記録する。"""
        if not fname:
            return
        for w in writers.values():
            w.flush()
        progress.mark(fname)
        result.files_done += 1

    current = ""   # いま読んでいるファイル名
    waited = 0.0
    while True:
        code, raw, fname = client.gets()

        if code > 0:
            waited = 0.0
            if fname != current:          # 次のファイルに入った
                finish(current)
                current = fname
            if fname in progress:         # 前回までに読み終えたファイル
                result.resumed_files += 1
                client.skip()
                current = ""
                continue
            record_type = raw[:2].decode("ascii", errors="replace")
            if record_type not in writers:
                # 1ファイル=1レコード種別なので、残りは読まずに次のファイルへ
                result.skipped_files[record_type] = result.skipped_files.get(record_type, 0) + 1
                client.skip()
                finish(current)
                current = ""
                continue
            if len(raw) < RECORD_LENGTHS[record_type] - 2:
                result.short_records += 1
            parsed = parse_record(raw, struct_module)
            if parsed is None:
                continue
            _, row = parsed
            row["_seq"] = counter["seq"]
            counter["seq"] += 1
            writers[record_type].append(row)
            result.records[record_type] = result.records.get(record_type, 0) + 1
            counter["total"] += 1
            if counter["total"] % progress_every == 0:
                log(f"  {counter['total']:,} 件  {result.records}  "
                    f"（ファイル {result.files_done + result.resumed_files}/{opened.read_count}）")

        elif code == -1:          # ファイルの切り替わり。エラーではない
            result.files_switched += 1
            finish(current)
            current = ""
        elif code == 0:           # 全ファイル読み終わり
            finish(current)
            return
        elif code == -3:          # ダウンロード中。待って再試行（サンプルはここで止まる）
            if waited >= max_retry_sec:
                raise JVLinkError("JVGets", code,
                                  f"{max_retry_sec:.0f} 秒待ってもダウンロードが終わりません")
            sleep(retry_sleep_sec)
            waited += retry_sleep_sec
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
    result = fetch(dataspec=args.dataspec, fromtime=args.fromtime, option=args.option,
                   out_dir=args.out)
    print(result.summary())
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
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("summary", help="保存済み CSV の中身を集計して表示")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
