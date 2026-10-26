"""確定オッズ・速報オッズ（O1〜O6）の読み取りと保存（v8 4）。

■ 読み取り
  JV-Data 仕様書のレコードフォーマット（オッズ1〜6）の位置で、バイト列から直接読む。
  SDK の構造体（JVData_Struct.py）でも読めるが、3連単は1レコード 4,896 組あり、
  組ごとに Python のオブジェクトを作ると遅いので、numpy でまとめて切り出す。
  位置は SDK の構造体と同じ（tests/test_v8_odds.py で SDK がある環境では突き合わせる）。

    レコード  レコード長  組の数  1組のバイト数  組番 / オッズ / 人気 の桁
    O1 単勝      962        28        8        馬番2 / 4 / 2     （単勝の部分だけ使う。開始位置 44）
    O2 馬連     2042       153       13        4 / 6 / 3
    O3 ワイド   2654       153       17        4 / 最低5（最高5）/ 3
    O4 馬単     4031       306       13        4 / 6 / 3
    O5 3連複   12293       816       15        6 / 6 / 3
    O6 3連単   83285      4896       17        6 / 7 / 4
  オッズは 0.1 倍単位。"0000"（無投票）・"----"（発売前取消）・"****"（発売後取消）・空白（登録なし）は欠損。

■ 保存（data/jvlink/odds/）
  取得1回・レコード種別ごとに1つの圧縮した numpy ファイル（.npz）。parquet などの追加のライブラリは使わない。
    races : レースID・データ区分・作成日・発表月日時分・取得の通し番号（1レコード1行）
    rows  : 何番目のレースか・組番（"010203" → 10203 の整数）・オッズ（0.1倍単位の整数）・人気
  組が無い（登録なし）行は保存しない。3連単は1レース実際の組が数千、圧縮後 1レース約15KB。
  読むときは、同じレースの版が複数あれば データ区分の優先度（確定5 > 確定4 > 最終3 > 前日売最終2 > 中間1）→
  作成日 → 取得の順 で1つにする（jvmap.apply_data_kubun と同じ考え方）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# レコード種別: (券種, レコード長, 開始位置(1始まり), 組の数, 1組のバイト数, 組番の桁, オッズの桁, 人気の位置・桁)
LAYOUT = {
    "O1": ("単勝", 962, 44, 28, 8, 2, 4, (7, 2)),
    "O2": ("馬連", 2042, 41, 153, 13, 4, 6, (11, 3)),
    "O3": ("ワイド", 2654, 41, 153, 17, 4, 5, (15, 3)),
    "O4": ("馬単", 4031, 41, 306, 13, 4, 6, (11, 3)),
    "O5": ("3連複", 12293, 41, 816, 15, 6, 6, (13, 3)),
    "O6": ("3連単", 83285, 41, 4896, 17, 6, 7, (14, 4)),
}
ODDS_TYPES = tuple(LAYOUT)
KIND_OF = {rt: v[0] for rt, v in LAYOUT.items()}
RECORD_LENGTHS = {rt: v[1] for rt, v in LAYOUT.items()}
# 本番・検証で使う種別（単勝は SE の確定オッズがある。ワイドは今の検証では使わない）
DEFAULT_TYPES = ("O1", "O2", "O4", "O5", "O6")
KUBUN_PRIORITY = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "9": 9}
FINAL_KUBUN = {"4", "5"}             # 確定オッズ


def odds_dir(base: str | os.PathLike | None = None) -> Path:
    return Path(base or config.JV_DATA_DIR) / "odds"


@dataclass
class OddsRecord:
    record_type: str
    race_id: str           # 12桁（年4 場2 回2 日2 R2）
    date: str              # YYYYMMDD
    kubun: str
    make_date: int
    announced: str         # 発表月日時分（MMDDhhmm。中間オッズのみ）
    kumi: np.ndarray       # int32（"010203" → 10203）
    odds10: np.ndarray     # int32（0.1倍単位。欠損は -1）
    ninki: np.ndarray      # int16（欠損は -1）


def _digits(block: np.ndarray, start: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(組の数, バイト) の配列から、各組の start（0始まり）から width 桁を整数にする。数字以外は欠損。"""
    sub = block[:, start:start + width]
    is_digit = ((sub >= ord("0")) & (sub <= ord("9"))).all(axis=1)
    vals = (sub.astype(np.int64) - ord("0"))
    weights = 10 ** np.arange(width - 1, -1, -1, dtype=np.int64)
    num = (np.where(is_digit[:, None], vals, 0) * weights).sum(axis=1)
    return num, is_digit


def parse_odds(raw: bytes) -> OddsRecord | None:
    """オッズのレコード（O1〜O6）を読む。対象外・短すぎるときは None。"""
    rt = raw[:2].decode("ascii", errors="replace")
    if rt not in LAYOUT:
        return None
    kind, length, start, count, size, kw, ow, (np_pos, nw) = LAYOUT[rt]
    need = start - 1 + count * size
    if len(raw) < need:
        return None
    head = raw[:40].decode("ascii", errors="replace")
    year, md, jyo, kai, nichi, rno = head[11:15], head[15:19], head[19:21], head[21:23], head[23:25], head[25:27]
    block = np.frombuffer(raw, dtype=np.uint8, count=count * size, offset=start - 1).reshape(count, size)
    kumi, k_ok = _digits(block, 0, kw)
    odds, o_ok = _digits(block, kw, ow)
    ninki, n_ok = _digits(block, np_pos - 1, nw)
    keep = k_ok & (kumi > 0)
    odds = np.where(o_ok & (odds > 0), odds, -1)
    ninki = np.where(n_ok & (ninki > 0), ninki, -1)
    try:
        make = int(head[3:11])
    except ValueError:
        make = 0
    return OddsRecord(rt, f"{year}{jyo}{kai}{nichi}{rno}", f"{year}{md}", head[2], make, head[27:35].strip(),
                      kumi[keep].astype(np.int32), odds[keep].astype(np.int32), ninki[keep].astype(np.int16))


class OddsWriter:
    """取得1回ぶんのオッズをためて、レコード種別ごとに .npz に書く。"""

    def __init__(self, directory: str | os.PathLike, label: str = "fetch",
                 race_filter=None, kubun_filter=None):
        """directory: 書き出し先（確定オッズは odds_dir()、速報は odds_dir()/rt/日付 など）。"""
        self.out_dir = Path(directory)
        self.label = label
        self.race_filter = set(race_filter) if race_filter is not None else None
        self.kubun_filter = set(kubun_filter) if kubun_filter is not None else None
        self.buf: dict[str, list[OddsRecord]] = {}
        self.seq = 0
        self.kept = 0
        self.dropped = 0
        self.extra: dict[str, list] = {}

    def add(self, rec: OddsRecord, fetched_at: str = "") -> bool:
        if (self.race_filter is not None and rec.race_id not in self.race_filter) or \
                (self.kubun_filter is not None and rec.kubun not in self.kubun_filter):
            self.dropped += 1
            return False
        self.buf.setdefault(rec.record_type, []).append(rec)
        self.extra.setdefault(rec.record_type, []).append(fetched_at)
        self.kept += 1
        return True

    def flush(self) -> list[Path]:
        paths = []
        if not self.buf:
            return paths
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for rt, recs in self.buf.items():
            n = np.array([len(r.kumi) for r in recs])
            path = self.out_dir / f"{rt}_{self.label}_{stamp}.npz"
            np.savez_compressed(
                path,
                race_id=np.array([r.race_id for r in recs]), date=np.array([r.date for r in recs]),
                kubun=np.array([r.kubun for r in recs]), make_date=np.array([r.make_date for r in recs]),
                announced=np.array([r.announced for r in recs]),
                fetched_at=np.array(self.extra.get(rt, [""] * len(recs))),
                seq=np.arange(len(recs)) + self.seq,
                row_race=np.repeat(np.arange(len(recs)), n).astype(np.int32),
                kumi=np.concatenate([r.kumi for r in recs]) if len(recs) else np.zeros(0, np.int32),
                odds10=np.concatenate([r.odds10 for r in recs]) if len(recs) else np.zeros(0, np.int32),
                ninki=np.concatenate([r.ninki for r in recs]) if len(recs) else np.zeros(0, np.int16))
            self.seq += len(recs)
            paths.append(path)
        self.buf, self.extra = {}, {}
        return paths


RACE_KEYS = ("race_id", "date", "kubun", "make_date", "announced", "fetched_at", "seq")
ROW_KEYS = ("row_race", "kumi", "odds10", "ninki")


def _load_files(paths) -> list[dict]:
    out = []
    for fi, p in enumerate(sorted(paths)):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in RACE_KEYS + ROW_KEYS}
        d["file"] = fi
        out.append(d)
    return out


def load_odds(record_type: str, base: str | os.PathLike | None = None, race_ids=None,
              final_only: bool = True, latest: bool = True, directory=None) -> pd.DataFrame:
    """保存したオッズを読む。戻り値は1組1行（レースID・券種・データ区分・発表・取得時刻・組番・オッズ・人気）。

    latest=True なら、同じレースの版から1つ（データ区分の優先度 → 作成日 → 取得順）。
    final_only=True なら確定オッズ（データ区分 4・5）だけ。directory を渡すとそこを読む（速報など）。
    """
    d = Path(directory) if directory is not None else odds_dir(base)
    files = _load_files(d.glob(f"{record_type}_*.npz"))
    cols = ["レースID", "券種", "データ区分", "発表", "取得時刻", "組番", "オッズ", "人気"]
    if not files:
        return pd.DataFrame(columns=cols)
    races = pd.concat([pd.DataFrame({**{k: f[k] for k in RACE_KEYS}, "file": f["file"],
                                     "local": np.arange(len(f["race_id"]))}) for f in files], ignore_index=True)
    if final_only:
        races = races.loc[races["kubun"].isin(FINAL_KUBUN)]
    if race_ids is not None:
        races = races.loc[races["race_id"].isin(set(race_ids))]
    if latest and not races.empty:
        races = races.assign(_pri=races["kubun"].map(KUBUN_PRIORITY).fillna(0))
        races = races.sort_values(["race_id", "_pri", "make_date", "file", "seq"]).drop_duplicates("race_id", keep="last")
    parts = []
    for f in files:
        local = races.loc[races["file"] == f["file"], "local"].to_numpy()
        if not len(local):
            continue
        keep = np.zeros(len(f["race_id"]), dtype=bool)
        keep[local] = True
        m = keep[f["row_race"]]
        idx = f["row_race"][m]
        o = f["odds10"][m]
        parts.append(pd.DataFrame({
            "レースID": f["race_id"][idx], "券種": KIND_OF[record_type], "データ区分": f["kubun"][idx],
            "発表": f["announced"][idx], "取得時刻": f["fetched_at"][idx],
            "組番": f["kumi"][m], "オッズ": np.where(o > 0, o / 10.0, np.nan), "人気": f["ninki"][m]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def kumi_tuple(kumi: int, record_type: str) -> tuple[int, ...]:
    """10203 → (1, 2, 3)。単勝は (馬番,)。"""
    width = {"O1": 1, "O2": 2, "O3": 2, "O4": 2, "O5": 3, "O6": 3}[record_type]
    s = f"{int(kumi):0{2 * width}d}"
    return tuple(int(s[i:i + 2]) for i in range(0, 2 * width, 2))


def summary(base: str | os.PathLike | None = None) -> pd.DataFrame:
    """保存したオッズの件数（レコード種別 × データ区分）。"""
    rows = []
    for rt in ODDS_TYPES:
        paths = list(odds_dir(base).glob(f"{rt}_*.npz"))
        if not paths:
            continue
        files = _load_files(paths)
        races = pd.DataFrame({"kubun": np.concatenate([f["kubun"] for f in files]),
                              "race_id": np.concatenate([f["race_id"] for f in files])})
        size = sum(p.stat().st_size for p in paths)
        for k, g in races.groupby("kubun"):
            rows.append({"種別": rt, "券種": KIND_OF[rt], "データ区分": k, "レコード": len(g),
                         "レース": g["race_id"].nunique(), "ファイル": len(paths), "容量MB": size / 1e6})
    return pd.DataFrame(rows)
