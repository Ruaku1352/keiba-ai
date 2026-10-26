"""v8 2e：買い目の選び方（平場だけで比べる）。

どの券種の当たり外れも「1〜3着の並び」で決まるので、3連単の確率の表 P[i,j,k]（2c）から全部計算できる。
配当は市場の確率から推定する（2d。本番では締切前の実際のオッズを使う）。
評価は実際の払戻（HR の的中組。同着で複数あればそのすべて）で行う。

  S0 モデル（1段目）の上位6頭BOX     … 今の戦略（参考。平場）
  S1 混ぜた確率の上位6頭BOX
  S2 熱い組だけ：推定配当が基準以上の3連単の組から、確率の高い順に 120 点（100円ずつ）
  S3 S2 をフォーメーションのカード数枚に限定（会場のマークカードで買える形）
  S4 券種と金額を混ぜる（3連単・3連複・馬単・馬連、1点の金額も変えてよい）。予算 12,000円で
     「払戻の合計が基準以上になる確率」を大きくする組み合わせを貪欲法で選ぶ
  S4c S4 に券種ごとの点数の上限（マークカードの枚数の上限）を付けたもの
  S5 S4 を3連単だけにしたもの（金額は変えてよい）

S4 の考え方：1点（券種・組・金額）ごとに「それ1点だけで基準に届く金額」を
   金額 = ceil(基準 ÷ 推定配当) × 100円
とし、その1点が当たる並びの確率の合計 ÷ 金額 が大きいものから予算まで選ぶ。
例）人気薄3頭の3連複1点で3連単の6通りをまとめて押さえる、250倍の3連単なら200円で5万円に届く、など。
最後に「選んだ全部の推定払戻の合計が基準以上になる並び」の確率を、予測の達成確率として出す。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import market

BUDGET = 12_000
TARGET = 50_000
TOP_H = 10                         # 券種を混ぜる買い方で候補にする頭数（確率の上位）
MAX_CARDS_FORMATION = 3            # S3：フォーメーションのカードの枚数の上限
# S4c：1枚のマークカードに書ける組の数と、券種ごとのカードの枚数の上限（会場で買える形）。
# JRA の通常のマークカードは1枚に数組まで書ける。ここでは 1枚 4組・券種ごとに 2枚までと仮定した（設定で変えられる）
MARKS_PER_CARD = 4
MAX_CARDS_PER_KIND = 2
KINDS = ("3連単", "3連複", "馬単", "馬連")


@dataclass
class Bet:
    kind: str
    combo: tuple            # 馬番（3連複・馬連は小さい順）
    stake: int              # 円


@dataclass
class Plan:
    name: str
    bets: list[Bet] = field(default_factory=list)
    predicted: float = float("nan")     # 予測の達成確率

    @property
    def invest(self) -> int:
        return int(sum(b.stake for b in self.bets))


# ---------------------------------------------------------------------------
# 候補の構造（上位 H 頭の中の位置で表す。レースによらず同じなので1回だけ作る）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def coverage_structure(h: int):
    """上位 h 頭の並び（i,j,k 異なる）と、各候補（券種・組）が当たる並びの行列。"""
    outcomes = [(i, j, k) for i in range(h) for j in range(h) for k in range(h)
                if len({i, j, k}) == 3]
    o_index = {o: n for n, o in enumerate(outcomes)}
    cands, rows = [], []

    def add(kind, combo, covered):
        cands.append((kind, combo))
        rows.append([o_index[o] for o in covered])

    for o in outcomes:
        add("3連単", o, [o])
    for c in itertools.combinations(range(h), 3):
        add("3連複", c, list(itertools.permutations(c)))
    for i, j in itertools.permutations(range(h), 2):
        add("馬単", (i, j), [(i, j, k) for k in range(h) if k not in (i, j)])
    for i, j in itertools.combinations(range(h), 2):
        add("馬連", (i, j), [(a, b, k) for a, b in ((i, j), (j, i)) for k in range(h) if k not in (i, j)])
    M = np.zeros((len(cands), len(outcomes)), dtype=np.float64)   # float64：掛け算のたびに型を変えない
    for r, cols in enumerate(rows):
        M[r, cols] = 1.0
    return outcomes, cands, M


@lru_cache(maxsize=None)
def candidate_index(h: int):
    """候補ごとの券種と、上位 h 頭の中での位置（3頭ぶん。2頭の券種は3つ目を 0 で埋める）。"""
    _, cands, _ = coverage_structure(h)
    kind = np.array([KINDS.index(k) for k, _ in cands])
    pos = np.array([list(c) + [0] * (3 - len(c)) for _, c in cands])
    return kind, pos


def _subtensor(P: np.ndarray, order: np.ndarray, outcomes) -> np.ndarray:
    o = np.array(outcomes)
    return P[order[o[:, 0]], order[o[:, 1]], order[o[:, 2]]]


def _candidate_probs(Q: dict, order: np.ndarray, h: int) -> np.ndarray:
    """各候補の市場の確率（推定配当の元）。まとめて引く。"""
    kind, pos = candidate_index(h)
    horse = order[pos]                                     # (候補, 3) 実際の馬の位置
    out = np.empty(len(kind))
    for k, name in enumerate(KINDS):
        m = kind == k
        if name in ("3連単", "3連複"):
            out[m] = Q[name][horse[m, 0], horse[m, 1], horse[m, 2]]
        else:
            out[m] = Q[name][horse[m, 0], horse[m, 1]]
    return out


# ---------------------------------------------------------------------------
# 買い方
# ---------------------------------------------------------------------------
def box(p: np.ndarray, posts: np.ndarray, size: int = 6, stake: int = 100, name: str = "BOX") -> Plan:
    top = np.argsort(-p, kind="mergesort")[:size]
    bets = [Bet("3連単", tuple(int(posts[x]) for x in c), stake) for c in itertools.permutations(top, 3)]
    return Plan(name, bets)


def hot_top(P: np.ndarray, Q3: np.ndarray, posts: np.ndarray, points: int = 120,
            target: float = TARGET, stake: int = 100) -> Plan:
    """S2：推定配当（100円あたり）× stake/100 が基準以上の3連単の組を、確率の高い順に points 点。"""
    est = 100 * market.PAYOUT_RATE["3連単"] / np.where(Q3 > 0, Q3, np.nan)
    hot = (est * stake / 100 >= target) & (P > 0)
    idx = np.argwhere(hot)
    if len(idx) == 0:
        return Plan("熱い組 120点")
    pr = P[hot]
    pick = idx[np.argsort(-pr, kind="mergesort")[:points]]
    return Plan("熱い組 120点", [Bet("3連単", tuple(int(posts[x]) for x in c), stake) for c in pick])


def formation_cards(P: np.ndarray, Q3: np.ndarray, p: np.ndarray, posts: np.ndarray,
                    max_cards: int = MAX_CARDS_FORMATION, points: int = 120, target: float = TARGET,
                    h: int = TOP_H) -> tuple[Plan, list[tuple]]:
    """S3：1着・2着・3着の欄に「確率の上位 a・b・c 頭」を塗るフォーメーションを、最大 max_cards 枚。

    熱い組（推定配当 ≥ 基準）の確率の合計が、点数あたりで一番増えるカードを順に足す。
    戻り値の2つ目は、カードごとの (1着の馬番, 2着の馬番, 3着の馬番)。
    """
    n = len(p)
    h = min(h, n)
    order = np.argsort(-p, kind="mergesort")[:h]
    sub = P[np.ix_(order, order, order)]
    est = 100 * market.PAYOUT_RATE["3連単"] / np.where(Q3 > 0, Q3, np.nan)
    hot = (est[np.ix_(order, order, order)] >= target)
    ii, jj, kk = np.meshgrid(np.arange(h), np.arange(h), np.arange(h), indexing="ij")
    valid = ((ii != jj) & (jj != kk) & (ii != kk)).astype(float)
    weight = np.where(hot, sub, 0.0) * valid
    covered = np.zeros_like(valid)
    cards, remaining = [], points
    for _ in range(max_cards):
        W = weight * (1 - covered)
        S = W.cumsum(0).cumsum(1).cumsum(2)
        V = valid.cumsum(0).cumsum(1).cumsum(2)
        best, best_val = None, 0.0
        for a in range(1, min(3, h) + 1):
            for b in range(1, min(6, h) + 1):
                for c in range(1, h + 1):
                    pts = V[a - 1, b - 1, c - 1]
                    if pts == 0 or pts > remaining:
                        continue
                    val = S[a - 1, b - 1, c - 1] / pts
                    if val > best_val:
                        best, best_val = (a, b, c), val
        if best is None:
            break
        a, b, c = best
        covered[:a, :b, :c] = np.maximum(covered[:a, :b, :c], valid[:a, :b, :c])
        remaining -= int(V[a - 1, b - 1, c - 1])
        cards.append(tuple(tuple(int(posts[order[x]]) for x in range(m)) for m in (a, b, c)))
    bets = [Bet("3連単", (int(posts[order[i]]), int(posts[order[j]]), int(posts[order[k]])), 100)
            for i, j, k in np.argwhere(covered > 0)]
    return Plan(f"フォーメーション{max_cards}枚まで", bets), cards


def mixed(P: np.ndarray, Q: dict, p: np.ndarray, posts: np.ndarray, budget: int = BUDGET,
          target: float = TARGET, kinds=KINDS, max_per_kind: int | None = None,
          h: int = TOP_H, name: str = "券種混合", margin: float = 1.0,
          max_points: int | None = None) -> Plan:
    """S4 / S4c / S5：券種と金額を混ぜて、払戻の合計が基準以上になる確率を大きくする（貪欲法）。

    margin     : 1点の金額を「推定配当 × 金額 ≥ 基準 × margin」にする（1.2 なら推定より2割下振れしても届く）。
                 達成の判定（予測・実際とも）は基準のまま
    max_points : 入力する点数の合計の上限（スマッピーの1つの QR に入る点数。券種は問わない）
    """
    n = len(p)
    h = min(h, n)
    if h < 3:
        return Plan(name)
    order = np.argsort(-p, kind="mergesort")[:h]
    outcomes, cands, M = coverage_structure(h)
    po = _subtensor(P, order, outcomes)                         # 並びの予測確率
    q_c = _candidate_probs(Q, order, h)
    kind_code, _ = candidate_index(h)
    rate = np.array([market.PAYOUT_RATE[k] for k in KINDS])[kind_code]
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.where(q_c > 0, 100.0 * rate / q_c, np.nan)     # 100円あたりの推定配当
        stake = np.ceil(target * margin / est) * 100            # それ1点で基準（× 余裕）に届く金額
    kind_arr = np.array(KINDS)[kind_code]
    usable = np.isfinite(stake) & (stake <= budget) & np.isin(kind_arr, list(kinds))
    uncovered = np.ones(len(outcomes))
    chosen, spent = [], 0
    used_per_kind = {k: 0 for k in KINDS}
    gains = M @ po                                              # 各候補が新しく押さえる確率（毎回差分で更新）
    while True:
        ok = usable & (stake <= budget - spent) & (gains > 1e-15)
        if max_per_kind is not None:
            for k in KINDS:
                if used_per_kind[k] >= max_per_kind:
                    ok &= kind_arr != k
        if max_points is not None and len(chosen) >= max_points:
            break
        if not ok.any():
            break
        ratio = np.where(ok, gains / stake, -1)
        best = int(np.argmax(ratio))
        chosen.append(best)
        spent += int(stake[best])
        used_per_kind[kind_arr[best]] += 1
        newly = np.flatnonzero((M[best] > 0) & (uncovered > 0))
        if len(newly):
            gains -= M[:, newly] @ po[newly]
            uncovered[newly] = 0
        usable[best] = False
    bets = []
    for c in chosen:
        k, combo = cands[c]
        horses = tuple(int(posts[order[x]]) for x in combo)
        if k in ("3連複", "馬連"):
            horses = tuple(sorted(horses))
        bets.append(Bet(k, horses, int(stake[c])))
    plan = Plan(name, bets)
    # 予測の達成確率：選んだ全部の推定払戻の合計が基準以上になる並びの確率。
    # 候補選びは上位 h 頭の並びだけで行うが、ここは全頭の並びで計算する（馬単・3連複などは、残りの着に
    # 上位 h 頭以外の馬が来ても当たるため。上位 h 頭だけで数えると少し小さく出る）
    plan.predicted = predicted_success(plan, P, Q, posts, target) if chosen else 0.0
    return plan


def predicted_success(plan: Plan, P: np.ndarray, Q: dict, posts: np.ndarray, target: float) -> float:
    """選んだ買い目の推定払戻の合計が基準以上になる並びの確率（全頭の並びで）。"""
    if not plan.bets:
        return 0.0
    index = {int(x): i for i, x in enumerate(posts)}
    n = len(posts)
    total = np.zeros((n, n, n))
    for b in plan.bets:
        ix = [index[x] for x in b.combo]
        q = Q[b.kind][tuple(ix)] if b.kind in ("3連単", "3連複") else Q[b.kind][ix[0], ix[1]]
        pay = market.estimated_payout(float(q), b.kind) * b.stake / 100 if q > 0 else 0.0
        if b.kind == "3連単":
            total[ix[0], ix[1], ix[2]] += pay
        elif b.kind == "3連複":
            for perm in itertools.permutations(ix):
                total[perm] += pay
        elif b.kind == "馬単":
            total[ix[0], ix[1], :] += pay
        else:
            total[ix[0], ix[1], :] += pay
            total[ix[1], ix[0], :] += pay
    return float(P[total >= target].sum())


# ---------------------------------------------------------------------------
# 実際の払戻での評価
# ---------------------------------------------------------------------------
def winning_combos(pay_row: pd.Series) -> dict[str, dict[tuple, float]]:
    """券種 → {的中組: 100円あたりの払戻}（同着なら複数）。"""
    out = {}
    for kind in KINDS:
        kumis = str(pay_row.get(f"{kind}組番一覧", "") or "").split(";")
        pays = str(pay_row.get(f"{kind}払戻一覧", "") or "").split(";")
        d = {}
        for k, v in zip(kumis, pays):
            if k and v:
                combo = market.parse_kumi(k, kind)
                d[tuple(sorted(combo)) if kind in ("3連複", "馬連") else combo] = float(v)
        out[kind] = d
    return out


def realized(plan: Plan, wins: dict) -> float:
    """実際の払戻の合計（円）。"""
    total = 0.0
    for b in plan.bets:
        pay = wins.get(b.kind, {}).get(b.combo)
        if pay:
            total += pay * b.stake / 100
    return total


def summarize(results: pd.DataFrame, target: float = TARGET, keys=("買い方",)) -> pd.DataFrame:
    """買い方ごとの成績（平場）。results は1レース・1買い方1行（買い方, 投資, 払戻, 予測の達成確率）。
    keys で集計の単位を変えられる（例: ["確率", "買い方"]）。"""
    rows = []
    keys = list(keys)
    for name, g in results.groupby(keys, sort=False):
        name = name if isinstance(name, tuple) else (name,)
        n = len(g)
        hit = int((g["払戻"] > 0).sum())
        ach = int((g["払戻"] >= target).sum())
        lo, hi = market.wilson(ach, n)
        rows.append({**dict(zip(keys, name)), "レース数": n, "的中率": hit / n if n else np.nan,
                     "達成率": ach / n if n else np.nan, "下限": lo, "上限": hi,
                     "予測の達成確率（平均）": float(g["予測の達成確率"].mean()),
                     "平均投資": float(g["投資"].mean()),
                     "回収率": float(g["払戻"].sum() / g["投資"].sum()) if g["投資"].sum() else np.nan})
    return pd.DataFrame(rows)


def card_text(cards: list[tuple]) -> str:
    """フォーメーションのカードを、1着・2着・3着の欄に塗る馬番の形で。"""
    return " / ".join(f"[1着 {'・'.join(map(str, a))} | 2着 {'・'.join(map(str, b))} | 3着 {'・'.join(map(str, c))}]"
                      for a, b, c in cards)
