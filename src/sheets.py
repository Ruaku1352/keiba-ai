"""Google スプレッドシート「競馬パドック記録」への書き込み（v7）。

■ 設定（リポジトリには入れない）
  鍵ファイルのパスとシートの ID は、次の順で探す:
    1. 環境変数 KEIBA_GOOGLE_KEY（鍵ファイルのパス）と KEIBA_SHEET_ID（シートの ID）
    2. 環境変数 KEIBA_SETTINGS が指す JSON
    3. リポジトリ直下の local_settings.json（.gitignore 済み）
  JSON の形は local_settings.example.json を参照。

■ 書き込みのきまり
  - 列は位置ではなく**1行目の見出しの名前**で探す（シートごとに列の順番が違ってよい）
  - 行は (レースID, 馬名) で探す。同じレースを予測し直したら、同じ行を**上書き**する（行は増えない）。
    馬番は出馬表の確定で変わりうるので、キーにはしない
  - 予測し直したときに居なくなった馬（取消・除外）は、行を消さずに 予測順位 を「取消」にし、予測確率を空にする
    （手入力の欄が残っているかもしれないので、行は消さない）
  - **発汗・落着き・メモは書き込まない**（PROTECTED_COLUMNS。スマホで手入力する欄）。
    書き込む列の一覧にこれらが入っていたらエラーにする
  - 見出しに「更新日時」の列があれば、書き込んだ行にその時刻を入れる（無ければ何もしない）
  - レースID は先頭に ' を付けて文字列として書く（12桁の数字が指数表記や数値に化けないように）

テストでは本物のシートを使わず、`get_all_values` / `batch_update` / `append_rows` を持つ
偽物のワークシートで確かめる（tests/test_v7.py）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = REPO / "local_settings.json"
ENTRY_SHEET = "評価入力"
RECORD_SHEET = "記録"
KEY_COLUMNS = ("レースID", "馬名")
PROTECTED_COLUMNS = ("発汗", "落着き", "メモ")
UPDATED_AT_COLUMN = "更新日時"
SCRATCHED = "取消"

ENTRY_COLUMNS = ["レースID", "日付", "レース名", "馬番", "馬名", "予測確率", "予測順位"]
RECORD_COLUMNS = ENTRY_COLUMNS + ["着順", "単勝オッズ", "馬体重"]


@dataclass
class SheetSettings:
    key_file: str
    spreadsheet_id: str
    entry_sheet: str = ENTRY_SHEET
    record_sheet: str = RECORD_SHEET


def load_settings(path: str | os.PathLike | None = None) -> SheetSettings:
    """鍵ファイルとシート ID を、環境変数か手元の設定ファイルから読む。"""
    data: dict = {}
    path = path or os.environ.get("KEIBA_SETTINGS") or DEFAULT_SETTINGS
    if Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    key = os.environ.get("KEIBA_GOOGLE_KEY") or data.get("google_service_account_file")
    sheet_id = os.environ.get("KEIBA_SHEET_ID") or data.get("spreadsheet_id")
    if not key or not sheet_id:
        raise SystemExit(
            "スプレッドシートの設定がありません。リポジトリ直下に local_settings.json を作るか"
            "（local_settings.example.json を参照）、環境変数 KEIBA_GOOGLE_KEY と KEIBA_SHEET_ID を設定してください")
    if not Path(key).exists():
        raise SystemExit(f"鍵ファイルがありません: {key}")
    return SheetSettings(key, sheet_id, data.get("entry_sheet", ENTRY_SHEET),
                         data.get("record_sheet", RECORD_SHEET))


def open_worksheet(settings: SheetSettings, title: str):
    """gspread でワークシートを開く（gspread はここでだけ読み込む）。"""
    import gspread

    client = gspread.service_account(filename=settings.key_file)
    return client.open_by_key(settings.spreadsheet_id).worksheet(title)


# ---------------------------------------------------------------------------
# 見出しの名前での書き込み
# ---------------------------------------------------------------------------
def col_letter(n: int) -> str:
    """1 -> A, 27 -> AA。"""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _norm_key(v) -> str:
    """キーの比較用。数値として読まれた "202610110511.0" や前後の空白・先頭の ' を揃える。"""
    s = str(v).strip().lstrip("'")
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _cell(value, column: str):
    """シートに書く値。欠損は空文字、レースID は文字列として書く。"""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return ""
    if column == "レースID":
        return "'" + _norm_key(value)
    if hasattr(value, "item"):          # numpy の数値を Python の数値に
        value = value.item()
    return value


def _same(old: str, new) -> bool:
    if new == "" or new is None:
        return str(old).strip() == ""
    new_s = str(new).lstrip("'")
    if str(old).strip() == new_s:
        return True
    try:
        return abs(float(str(old).replace(",", "")) - float(new_s)) < 1e-9
    except ValueError:
        return False


@dataclass
class UpsertResult:
    updated_rows: int = 0
    appended_rows: int = 0
    changed_cells: int = 0
    scratched_rows: int = 0
    duplicate_keys: int = 0

    def __str__(self) -> str:
        s = (f"上書き {self.updated_rows} 行（変わったセル {self.changed_cells}）/ 追加 {self.appended_rows} 行"
             f" / 取消にした行 {self.scratched_rows}")
        if self.duplicate_keys:
            s += f" / 【注意】同じレースID・馬名の行が複数ある: {self.duplicate_keys} 件（すべて上書きした）"
        return s


def upsert_rows(ws, rows: pd.DataFrame, columns: list[str], mark_scratched: bool = True,
                now: datetime | None = None) -> UpsertResult:
    """rows を (レースID, 馬名) をキーにシートへ書く。columns の列だけを書き、他の列には触らない。

    mark_scratched=True なら、rows に含まれるレースのうち rows に居ない馬の行を「取消」にする。
    """
    bad = [c for c in columns if c in PROTECTED_COLUMNS]
    if bad:
        raise ValueError(f"手入力の列には書き込まない: {bad}")
    values = ws.get_all_values()
    if not values:
        raise SystemExit(f"シート「{ws.title}」の1行目に見出しがありません")
    header = [h.strip() for h in values[0]]
    missing = [c for c in list(KEY_COLUMNS) + columns if c not in header]
    if missing:
        raise SystemExit(f"シート「{ws.title}」の見出しに無い列: {missing}（見出し: {header}）")
    pos = {h: i for i, h in enumerate(header)}           # 0始まり
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    has_stamp = UPDATED_AT_COLUMN in pos

    existing: dict[tuple[str, str], list[int]] = {}
    for r, row in enumerate(values[1:], start=2):        # シートの行番号（1始まり、見出しが1行目）
        row = row + [""] * (len(header) - len(row))
        key = (_norm_key(row[pos["レースID"]]), str(row[pos["馬名"]]).strip())
        if key[0]:
            existing.setdefault(key, []).append(r)

    result = UpsertResult()
    updates, appends = [], []

    def set_cell(r: int, c: str, v) -> bool:
        old = values[r - 1][pos[c]] if pos[c] < len(values[r - 1]) else ""
        if _same(old, v):
            return False
        updates.append({"range": f"{col_letter(pos[c] + 1)}{r}", "values": [[v]]})
        return True

    written = set()
    for rec in rows.to_dict("records"):
        key = (_norm_key(rec["レースID"]), str(rec["馬名"]).strip())
        written.add(key)
        if key in existing:
            if len(existing[key]) > 1:
                result.duplicate_keys += 1
            for r in existing[key]:
                changed = sum(set_cell(r, c, _cell(rec.get(c), c)) for c in columns)
                if changed and has_stamp:
                    set_cell(r, UPDATED_AT_COLUMN, stamp)
                result.changed_cells += changed
                result.updated_rows += 1
        else:
            line = [""] * len(header)
            for c in columns:
                line[pos[c]] = _cell(rec.get(c), c)
            if has_stamp:
                line[pos[UPDATED_AT_COLUMN]] = stamp
            appends.append(line)
            result.appended_rows += 1

    if mark_scratched:
        races = {_norm_key(x) for x in rows["レースID"]}
        for key, rs in existing.items():
            if key[0] in races and key not in written:
                for r in rs:
                    changed = 0
                    if "予測順位" in columns:
                        changed += set_cell(r, "予測順位", SCRATCHED)
                    if "予測確率" in columns:
                        changed += set_cell(r, "予測確率", "")
                    if changed and has_stamp:
                        set_cell(r, UPDATED_AT_COLUMN, stamp)
                    result.scratched_rows += bool(changed)

    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")
    if appends:
        ws.append_rows(appends, value_input_option="USER_ENTERED", table_range="A1")
    return result


def prediction_rows(table: pd.DataFrame) -> pd.DataFrame:
    """predict.output_table の表から「評価入力」に書く列だけを取り出す。"""
    out = table[ENTRY_COLUMNS].copy()
    out["予測確率"] = out["予測確率"].astype(float).round(4)
    return out
