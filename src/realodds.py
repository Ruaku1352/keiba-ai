"""v8d：本物の確定オッズで買い目を選ぶ検証（平場サンプル。重賞の成績は出さない）。

- 取り直した確定オッズ（O2 馬連・O4 馬単・O5 3連複・O6 3連単。data/jvlink/odds）を、レースごとに
  bets.mixed が使う「券種 → 100円あたりの配当の表」（市場の確率の表 Q と同じ形・同じ馬の並び）にする
- 推定した配当（100 × 控除後の割合 ÷ 市場の確率）と本物の配当の比を、全部の組で集計する

確定オッズは締切の時点のもの。本番は発走20分前のオッズで買うので、ここでの結果は「オッズが分かっていたら」の上限に近い。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import market, odds

KIND_TYPES = {"馬連": "O2", "馬単": "O4", "3連複": "O5", "3連単": "O6"}
ODDS_BINS = [0, 10, 50, 100, 300, 1000, 1e9]          # 倍
ODDS_BIN_LABELS = ["〜10倍", "10〜50倍", "50〜100倍", "100〜300倍", "300〜1000倍", "1000倍〜"]


def load_final_odds(race_ids, base=None, kinds=None) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """レースID → 券種 → (組の馬番の配列 (組, 2 or 3), 100円あたりの配当)。オッズの無い組は入れない。

    3連単は1レース数千組あるので、組ごとの表（DataFrame）にはせず、保存した配列をレースごとに切り出す。
    """
    out: dict[str, dict] = {}
    for kind, rt in KIND_TYPES.items():
        if kinds is not None and kind not in kinds:
            continue
        files = odds._load_files(odds.odds_dir(base).glob(f"{rt}_*.npz"))
        if not files:
            continue
        races = odds.select_versions(files, set(map(str, race_ids)))
        for f in files:
            sel = races.loc[races["file"] == f["file"]]
            if sel.empty:
                continue
            rr = f["row_race"]
            bounds = np.searchsorted(rr, np.arange(len(f["race_id"]) + 1))      # row_race は昇順
            for rid, local in zip(sel["race_id"], sel["local"]):
                a, b = bounds[local], bounds[local + 1]
                k = f["kumi"][a:b].astype(np.int64)
                o = f["odds10"][a:b]
                ok = o > 0
                k = k[ok]
                horses = (np.column_stack([k // 10000, (k // 100) % 100, k % 100]) if rt in ("O5", "O6")
                          else np.column_stack([k // 100, k % 100]))
                out.setdefault(str(rid), {})[kind] = (horses, o[ok].astype("float64") * 10)   # 0.1倍 → 100円あたり
        del files
    return out


def payout_tables(entry: dict, posts: np.ndarray) -> dict[str, np.ndarray] | None:
    """1レースぶんを、行の並び（posts の順）の配当の表に。4券種のどれかが無ければ None。"""
    if any(k not in entry for k in KIND_TYPES):
        return None
    n = len(posts)
    idx = np.full(100, -1)
    idx[posts] = np.arange(n)
    out = {}
    for kind, (horses, pay) in entry.items():
        ix = idx[np.clip(horses, 0, 99)]
        ok = (ix >= 0).all(axis=1)
        ix, pay = ix[ok], pay[ok]
        if kind in ("3連単", "3連複"):
            t = np.full((n, n, n), np.nan)
            perms = [(0, 1, 2)] if kind == "3連単" else [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            for a, b, c in perms:
                t[ix[:, a], ix[:, b], ix[:, c]] = pay
        else:
            t = np.full((n, n), np.nan)
            t[ix[:, 0], ix[:, 1]] = pay
            if kind == "馬連":
                t[ix[:, 1], ix[:, 0]] = pay
        out[kind] = t
    return out


KINDS = ("3連単", "3連複", "馬単", "馬連")
LOG_EDGES = np.linspace(-6, 6, 2401)                   # log(推定÷本物) の目盛り（0.005 刻み。中央値はこの精度）


def ratio_arrays(Q: dict, real: dict, P: np.ndarray):
    """全部の組（並びの重複は除く）について、券種ごとに (log(推定÷本物), 本物の配当, P からの組の確率)。"""
    Pc = market.combo_probabilities(P)
    n = P.shape[0]
    out = {}
    for kind in KINDS:
        r = real[kind]
        m = np.isfinite(r) & (r > 0)
        if kind == "3連複":
            i, j, k = np.indices((n, n, n))
            m &= (i < j) & (j < k)
        elif kind == "馬連":
            i, j = np.indices((n, n))
            m &= i < j
        q = Q[kind][m]
        ok = q > 0
        est = 100.0 * market.PAYOUT_RATE[kind] / q[ok]
        out[kind] = (np.log(est / r[m][ok]), r[m][ok], Pc[kind][m][ok])
    return out


class RatioStats:
    """推定÷本物 を、券種 × 本物のオッズの帯で集計する（全部の組をためずに、目盛りごとの数と和だけ持つ）。"""

    def __init__(self):
        nb = len(ODDS_BIN_LABELS) + 1                  # 最後は「全部」
        self.hist = {k: np.zeros((nb, len(LOG_EDGES) + 1)) for k in KINDS}
        self.wsum = {k: np.zeros(nb) for k in KINDS}
        self.wlog = {k: np.zeros(nb) for k in KINDS}

    def add(self, arrays: dict) -> None:
        for kind, (lr, pay, w) in arrays.items():
            band = np.digitize(pay / 100, ODDS_BINS[1:-1])
            pos = np.searchsorted(LOG_EDGES, lr)
            for b in (band, np.full(len(lr), len(ODDS_BIN_LABELS))):
                np.add.at(self.hist[kind], (b, pos), 1)
                self.wsum[kind] += np.bincount(b, weights=w, minlength=len(self.wsum[kind]))
                self.wlog[kind] += np.bincount(b, weights=w * lr, minlength=len(self.wlog[kind]))

    def summary(self) -> pd.DataFrame:
        rows = []
        labels = ODDS_BIN_LABELS + ["全部"]
        mid = np.r_[LOG_EDGES[0], (LOG_EDGES[1:] + LOG_EDGES[:-1]) / 2, LOG_EDGES[-1]]
        def frac(h, lo, hi):
            return h[(mid > lo) & (mid < hi)].sum() / h.sum()
        for kind in KINDS:
            for b in [len(ODDS_BIN_LABELS)] + list(range(len(ODDS_BIN_LABELS))):
                h = self.hist[kind][b]
                n = h.sum()
                if not n:
                    continue
                med = mid[np.searchsorted(np.cumsum(h), n / 2)]
                w = self.wsum[kind][b]
                rows.append({"券種": kind, "本物のオッズ": labels[b], "組の数": int(n),
                             "推定÷本物（中央値）": float(np.exp(med)),
                             "推定÷本物（確率で重みづけ）": float(np.exp(self.wlog[kind][b] / w)) if w > 0 else np.nan,
                             "±10%以内": frac(h, -np.log(1.1), np.log(1.1)),
                             "±25%以内": frac(h, -np.log(1.25), np.log(1.25)),
                             "推定が2割以上高い": frac(h, np.log(1.2), np.inf),
                             "推定が2割以上低い": frac(h, -np.inf, -np.log(1.2))})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# v10 当日の仕組み：1レースぶんのオッズ（速報のその時点の版、または確定）
# ---------------------------------------------------------------------------
SNAPSHOT_TYPES = {"単勝": "O1", **KIND_TYPES}


def load_race_odds(directory, race_id: str, before: str | None = None, final_only: bool = False) -> dict | None:
    """directory に保存したオッズ（npz）から、1レースぶんの一番新しい版を読む。

    before（"YYYY-MM-DD HH:MM:SS"）を渡すと、その時刻までに取ったものだけ（ドライランで「その時点」を再現する）。
    戻り値: {"単勝": {馬番: オッズ}, "馬連"・"馬単"・"3連複"・"3連単": (組の馬番, 100円あたりの配当),
             "発表": "MMDDhhmm", "取得時刻": ..., "データ区分": ...}。単勝が無ければ None。
    """
    from pathlib import Path

    d = Path(directory)
    out: dict = {}
    for kind, rt in SNAPSHOT_TYPES.items():
        files = odds._load_files(d.glob(f"{rt}_*.npz"))
        if not files:
            continue
        races = odds.select_versions(files, {str(race_id)}, final_only=final_only, latest=False)
        if before is not None:
            races = races.loc[races["fetched_at"].astype(str) <= before]
        if races.empty:
            continue
        races = races.assign(_pri=races["kubun"].map(odds.KUBUN_PRIORITY).fillna(0))
        last = races.sort_values(["fetched_at", "_pri", "make_date", "file", "seq"]).iloc[-1]
        f = next(x for x in files if x["file"] == last["file"])
        bounds = np.searchsorted(f["row_race"], np.arange(len(f["race_id"]) + 1))
        a, b = bounds[last["local"]], bounds[last["local"] + 1]
        k = f["kumi"][a:b].astype(np.int64)
        o = f["odds10"][a:b]
        ok = o > 0
        k, o = k[ok], o[ok].astype("float64")
        if kind == "単勝":
            out["単勝"] = {int(x): float(v) / 10 for x, v in zip(k, o)}
            out.update({"発表": str(last["announced"]), "取得時刻": str(last["fetched_at"]),
                        "データ区分": str(last["kubun"])})
        else:
            horses = (np.column_stack([k // 10000, (k // 100) % 100, k % 100]) if rt in ("O5", "O6")
                      else np.column_stack([k // 100, k % 100]))
            out[kind] = (horses, o * 10)
    return out if "単勝" in out else None
