"""依頼C：3連単の買い方シミュレーションと重賞限定の検証。

■ このプロジェクトの目的関数（v2で変更された点）
  回収率の最大化ではなく **「熱い当たりが出る確率」の最大化**。

      予算1,000円 → 10,000円
      予算3,000円 → 20,000円
      予算5,000円 → 30,000円
      予算10,000円 → 50,000円

  この基準を超える払戻が出たレースの割合を「達成率」と呼び、これを最大化する。
  回収率60%でも達成率が高ければ勝ちという設計。

■ シミュレーションの計算量について
  3連単のBOXは点数が多い（5頭BOXで60点）が、**当たる組み合わせは必ず1つだけ**
  （実際の1-2-3着の並び）なので、全点を展開する必要はない。

      当たり判定 = 実際の1着馬の予測順位 ≤ n1
                かつ 2着馬の予測順位 ≤ n2
                かつ 3着馬の予測順位 ≤ n3

  これだけで済むので、22,956レースでも一瞬で終わる。

■ 点数の数え方
  上位n1頭から1着、上位n2頭から2着、上位n3頭から3着を選ぶ（n1≤n2≤n3、入れ子）。
  1着に n1 通り、2着は1着で使った1頭を除いて n2-1 通り、
  3着はさらに2頭除いて n3-2 通り。よって

      点数 = n1 × (n2 - 1) × (n3 - 2)

  BOX(k) は n1=n2=n3=k なので k(k-1)(k-2)。3頭BOX=6点、4頭BOX=24点、5頭BOX=60点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, leakfree

BET_UNIT = 100  # 1点100円


# ---------------------------------------------------------------------------
# 買い方の定義
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Strategy:
    """買い方。上位 n1/n2/n3 頭から1着/2着/3着を選ぶ。

    kind="3連単" なら着順どおりに当てる必要がある。
    kind="3連複" は着順不問なので、n1=n2=n3=k の BOX としてだけ使う。
    """

    name: str
    n1: int
    n2: int
    n3: int
    kind: str = "3連単"

    @property
    def points(self) -> int:
        """点数（出走頭数が足りている場合）。"""
        if self.kind == "3連複":
            k = self.n3
            return max(k * (k - 1) * (k - 2) // 6, 0)  # C(k,3)
        return max(self.n1 * (self.n2 - 1) * (self.n3 - 2), 0)

    @property
    def payout_column(self) -> str:
        return f"{self.kind}払戻"


def box(k: int) -> Strategy:
    """3連単・上位k頭のBOX買い（k(k-1)(k-2) 点）。"""
    return Strategy(f"上位{k}頭BOX", k, k, k)


def formation(n1: int, n2: int, n3: int) -> Strategy:
    """3連単フォーメーション（1着候補n1頭・2着候補n2頭・3着候補n3頭）。"""
    return Strategy(f"F{n1}→{n2}→{n3}", n1, n2, n3)


def trio(k: int) -> Strategy:
    """3連複・上位k頭のBOX買い（C(k,3) 点）。

    着順を問わないので、上位k頭の中に1〜3着が全部入っていれば的中。
    3連単の同じk頭BOXと比べて点数が 1/6 になり的中率は6倍になる。
    その代わり配当は下がるので、「熱い基準」を満たせるかが焦点。
    """
    return Strategy(f"3連複{k}頭BOX", k, k, k, kind="3連複")


def multi2(n3: int) -> Strategy:
    """軸2頭マルチ（1着・2着を上位2頭で入れ替え、3着は上位n3頭）。

    入れ子フォーメーションの n1=n2=2 と同じ形。点数は 2×1×(n3-2)。
    """
    return Strategy(f"軸2頭マルチ→{n3}", 2, 2, n3)


# v2 までの比較セット（進捗まとめ 3.3 の表に対応）
DEFAULT_STRATEGIES = [
    formation(1, 3, 4), box(3), formation(1, 4, 6), formation(1, 4, 8),
    box(4), formation(2, 5, 6), formation(2, 5, 8), box(5),
    formation(2, 6, 10), box(6),
]

# v3 依頼Dで追加した買い方
EXTRA_STRATEGIES = [
    box(7), box(8),
    formation(1, 5, 8), formation(1, 5, 10), formation(2, 4, 8),
    formation(3, 5, 8), formation(3, 6, 8),
    multi2(8), multi2(10),
    trio(4), trio(5), trio(6), trio(7), trio(8),
]

# 総当たり用（v2 + v3）
ALL_STRATEGIES = DEFAULT_STRATEGIES + EXTRA_STRATEGIES


# ---------------------------------------------------------------------------
# 熱さの基準
# ---------------------------------------------------------------------------
def hot_threshold(investment: float) -> int:
    """投資額に対する「熱い」基準額を返す。

    開発者の設定した4段階（1,000/3,000/5,000/10,000円）のうち、
    投資額に **最も近い** 段階の基準を使う。
    例）投資2,400円 → 3,000円の段が近い → 基準20,000円
    """
    tiers = np.array(sorted(config.HOT_CRITERIA))
    nearest = tiers[np.argmin(np.abs(tiers - investment))]
    return config.HOT_CRITERIA[int(nearest)]


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """達成率の信頼区間（Wilson score interval）。

    達成率は数%と非常に小さいので、教科書どおりの
    p ± 1.96*sqrt(p(1-p)/n) だと下限が負に飛び出して意味を失う。
    Wilson 区間は 0 件でも [0, 上限] という妥当な区間を返すので、
    重賞のように母数が小さい場合はこちらが適切。
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _normal_cdf(x: float) -> float:
    """標準正規分布の累積分布関数（scipy を使わずに済ませるため）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_test(hits_a: int, n_a: int, hits_b: int, n_b: int) -> dict:
    """2つの割合の差の検定（両側 z 検定）＋ Newcombe 法による差の信頼区間。

    「重賞と平場で達成率が違う」を言うために使う。

    信頼区間が重ならないことは差があることの**十分条件だが必要条件ではない**。
    つまり区間の重なりだけで判定すると、本当は差があるケースを見逃す。
    差そのものを検定した方が検出力が高いので、こちらを主に使う。

    差の区間には Newcombe のスコア法を使う。個々の Wilson 区間から組み立てるので、
    達成率が数%と0に近い状況でも下限が負に飛び出しにくい。
    """
    if n_a == 0 or n_b == 0:
        return {"差": float("nan"), "z": float("nan"), "p値": float("nan"),
                "差下限": float("nan"), "差上限": float("nan"), "有意": False}

    p_a, p_b = hits_a / n_a, hits_b / n_b
    # プール比率で標準誤差を作る（帰無仮説「両者は同じ割合」のもとでの検定）
    p_pool = (hits_a + hits_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se if se > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    # Newcombe: 各群の Wilson 区間 [l, u] を使って差の区間を作る
    l_a, u_a = wilson_interval(hits_a, n_a)
    l_b, u_b = wilson_interval(hits_b, n_b)
    lower = (p_a - p_b) - math.sqrt((p_a - l_a) ** 2 + (u_b - p_b) ** 2)
    upper = (p_a - p_b) + math.sqrt((u_a - p_a) ** 2 + (p_b - l_b) ** 2)

    return {"差": p_a - p_b, "z": z, "p値": p_value,
            "差下限": lower, "差上限": upper, "有意": bool(p_value < 0.05)}


def paired_proportion_test(x, y, z: float = 1.96) -> dict:
    """同じレースを2通りで判定したときの割合の差（x − y）の検定と信頼区間。

    例）同じ検証レースで、学習開始 1995年のモデルと 2003年のモデルの「熱い当たり」を比べる。
    同じレースなので2つの結果は強く相関する。two_proportion_test（別々の標本を仮定）は
    この相関を無視するので区間が広すぎる。ここでは対応のある比較を使う:
      差の区間: Newcombe (1998) の方法10（各割合の Wilson 区間と、2つの判定の相関 φ から組み立てる）
      p値     : 正確な McNemar 検定（片方だけが当たったレース b, c の二項検定）
    z は区間の幅（1.96 = 95%。複数の案を比べるときに厳しくする。例: 2.394 = 98.3%）
    """
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    n = len(x)
    if n == 0 or len(y) != n:
        return {"差": float("nan"), "差下限": float("nan"), "差上限": float("nan"),
                "p値": float("nan"), "n": n, "xだけ": 0, "yだけ": 0, "有意": False}
    a = int((x & y).sum())
    b = int((x & ~y).sum())
    c = int((~x & y).sum())
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_interval(a + b, n, z)
    l2, u2 = wilson_interval(a + c, n, z)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (max(num - n / 2, 0.0) if num > 0 else num) / denom
    delta = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    eps = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    diff = p1 - p2

    # 正確な McNemar（両側）：b ~ Binomial(b + c, 1/2)
    m = b + c
    if m == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
        p_value = min(1.0, 2 * tail)
    return {"差": diff, "差下限": diff - delta, "差上限": diff + eps, "p値": p_value,
            "n": n, "xだけ": b, "yだけ": c, "両方": a, "有意": bool(p_value < 0.05)}


def normal_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """依頼書に書かれていた素朴な二項近似（比較用に残してある）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


# ---------------------------------------------------------------------------
# レース単位の表を作る
# ---------------------------------------------------------------------------
def build_race_table(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                     payout_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """1行1レースの検証用テーブルを作る。

    列: レースID, 日付, 出走頭数, 1着予測順位, 2着予測順位, 3着予測順位,
        3連単払戻, 重賞
    「1着予測順位」= 実際に1着だった馬が、モデルの予測で何番目に評価されていたか。
    """
    cols = config.resolve_columns(df)
    race_id, rank, date = cols["race_id"], cols["rank"], cols["date"]

    base = pd.DataFrame({
        # payout 側と型を揃えるため normalize_id を通す
        "レースID": leakfree.normalize_id(df[race_id]).to_numpy(),
        "日付": df[date].to_numpy(),
        "着順": df[rank].to_numpy(),
        "pred": np.asarray(pred, dtype="float64"),
    })
    if "grade" in cols:
        base["_grade"] = df[cols["grade"]].to_numpy()

    # 予測確率の高い順に 1,2,3... と順位を振る
    base["予測順位"] = base.groupby("レースID")["pred"].rank(
        ascending=False, method="first"
    ).astype("int16")

    race = base.groupby("レースID", sort=False).agg(
        日付=("日付", "first"), 出走頭数=("着順", "size")
    ).reset_index()

    # 実際の1〜3着馬の予測順位を横に並べる
    for k in (1, 2, 3):
        # 同着があると1レースに複数行出るため min で1つに潰す
        s = (base.loc[base["着順"] == k]
             .groupby("レースID")["予測順位"].min().rename(f"{k}着予測順位"))
        race = race.merge(s, on="レースID", how="left")

    if "_grade" in base.columns:
        grade = base.groupby("レースID")["_grade"].first().rename("格付け")
        race = race.merge(grade, on="レースID", how="left")
        race["重賞"] = race["格付け"].isin(config.GRADED_VALUES)
    else:
        race["重賞"] = False

    if payout_df is not None:
        # 3連単払戻・3連複払戻など、渡された払戻列をまとめて貼る
        pay = payout_df.rename(columns={"レースID": "_pid"})
        race = race.merge(pay, left_on="レースID", right_on="_pid", how="left")
        race = race.drop(columns=[c for c in ["_pid"] if c in race.columns])
    if "3連単払戻" not in race.columns:
        race["3連単払戻"] = np.nan

    # 3着まで揃っていないレース（出走3頭未満・同着で欠けるなど）は検証対象外
    race = race.dropna(subset=["1着予測順位", "2着予測順位", "3着予測順位"])
    return race.sort_values("日付").reset_index(drop=True)


# ---------------------------------------------------------------------------
# シミュレーション
# ---------------------------------------------------------------------------
def simulate(race: pd.DataFrame, strategy: Strategy,
             require_payout: bool = True,
             threshold: float | None = None) -> dict:
    """1つの買い方について成績を計算する。

    threshold を渡すと「熱い」基準額を上書きできる（依頼Cの感度分析用）。
    省略時は投資額から自動で決まる。
    """
    pay_col = strategy.payout_column
    r = race
    if pay_col not in r.columns:
        return {"買い方": strategy.name, "レース数": 0, "券種": strategy.kind}
    if require_payout:
        r = r.loc[r[pay_col].notna()]
    if r.empty:
        return {"買い方": strategy.name, "レース数": 0, "券種": strategy.kind}

    m = r["出走頭数"].to_numpy()
    # 出走頭数が候補数に満たない場合は、実際に買える点数まで縮める
    n1 = np.minimum(strategy.n1, m)
    n2 = np.minimum(strategy.n2, m)
    n3 = np.minimum(strategy.n3, m)

    r1 = r["1着予測順位"].to_numpy()
    r2 = r["2着予測順位"].to_numpy()
    r3 = r["3着予測順位"].to_numpy()

    if strategy.kind == "3連複":
        # 着順不問。上位k頭の中に1〜3着が全部入っていれば的中
        k = n3
        points = np.clip(k * (k - 1) * (k - 2) // 6, 0, None)
        hit = (r1 <= k) & (r2 <= k) & (r3 <= k)
    else:
        points = np.clip(n1 * (n2 - 1) * (n3 - 2), 0, None)
        hit = (r1 <= n1) & (r2 <= n2) & (r3 <= n3)

    invest = points * BET_UNIT
    payout = np.where(hit, r[pay_col].to_numpy(), 0.0)

    # 「熱い」基準はその買い方の1レースあたり投資額で決まる
    typical_invest = float(np.median(invest))
    threshold = hot_threshold(typical_invest) if threshold is None else float(threshold)
    hot = payout >= threshold

    n_races = len(r)
    n_hits = int(hit.sum())
    n_hot = int(hot.sum())
    lo, hi = wilson_interval(n_hot, n_races)

    hit_payouts = payout[hit]
    return {
        "買い方": strategy.name,
        "券種": strategy.kind,
        "点数": int(np.median(points)),
        "投資/R": int(typical_invest),
        "レース数": n_races,
        "的中数": n_hits,
        "的中率": n_hits / n_races,
        "配当中央値": float(np.median(hit_payouts)) if n_hits else np.nan,
        "熱い基準": threshold,
        "熱い当たり": n_hot,
        "達成率": n_hot / n_races,
        "達成率下限": lo,
        "達成率上限": hi,
        "回収率": float(payout.sum() / invest.sum()) if invest.sum() else np.nan,
        "収支": float(payout.sum() - invest.sum()),
    }


def strategy_table(race: pd.DataFrame, strategies: list[Strategy] | None = None,
                   graded_only: bool = False) -> pd.DataFrame:
    """買い方の総当たり表を作る（依頼Cのメイン出力）。

    graded_only=True で重賞（G1/G2/G3/G）だけに絞る。
    学習データは絞らず、**評価だけ**を重賞に絞るのがポイント。
    """
    strategies = strategies or DEFAULT_STRATEGIES
    target = race.loc[race["重賞"]] if graded_only else race
    rows = [simulate(target, s) for s in strategies]
    out = pd.DataFrame(rows)
    return out.sort_values("投資/R").reset_index(drop=True) if "投資/R" in out else out


def compare_graded_vs_flat(race: pd.DataFrame,
                           strategies: list[Strategy] | None = None) -> pd.DataFrame:
    """重賞と平場を並べて比較する。

    重賞は出走馬の実力が拮抗して人気が割れやすいので、
    平場と傾向が違う可能性がある。母数が小さいため信頼区間を必ず見ること。
    """
    strategies = strategies or DEFAULT_STRATEGIES
    graded = strategy_table(race.loc[race["重賞"]], strategies)
    flat = strategy_table(race.loc[~race["重賞"]], strategies)

    keep = ["買い方", "点数", "レース数", "的中率", "配当中央値", "達成率",
            "達成率下限", "達成率上限", "回収率"]
    merged = graded[keep].merge(
        flat[keep].drop(columns=["点数"]), on="買い方", suffixes=("_重賞", "_平場")
    )
    # 重賞の方が達成率が高ければプラス
    merged["達成率差"] = merged["達成率_重賞"] - merged["達成率_平場"]
    # 信頼区間が重ならない＝統計的に差があると言える
    merged["有意"] = (
        (merged["達成率下限_重賞"] > merged["達成率上限_平場"])
        | (merged["達成率上限_重賞"] < merged["達成率下限_平場"])
    )
    return merged


def simulate_gap_box(df: pd.DataFrame, pred: np.ndarray | pd.Series,
                     payout_df: pd.DataFrame | None = None,
                     n_gap: int = 2, graded_only: bool = False) -> dict:
    """乖離ベースの買い方（v2 セクション3.4 の再現・改善判定用）。

    軸1頭 = 予測1位。残り n_gap 頭 = 乖離 gap の上位。この3頭BOX（6点）。

        gap = 予測確率 ÷ 市場勝率 = 予測確率 × 単勝オッズ

    「モデルは高く評価しているのに人気がない馬」を拾う狙い。
    v2時点では達成率0.065%（上位3頭BOXの1/15）で完敗している。
    **新特徴量でここが改善するかが最重要の判定基準。**
    """
    cols = config.resolve_columns(df)
    race_id, rank, odds = cols["race_id"], cols["rank"], cols["win_odds"]

    base = pd.DataFrame({
        "レースID": leakfree.normalize_id(df[race_id]).to_numpy(),
        "着順": df[rank].to_numpy(),
        "pred": np.asarray(pred, dtype="float64"),
        "odds": pd.to_numeric(df[odds], errors="coerce").to_numpy(),
    })
    if "grade" in cols:
        base["重賞"] = pd.Series(df[cols["grade"]].to_numpy()).isin(config.GRADED_VALUES).to_numpy()
    else:
        base["重賞"] = False
    if graded_only:
        base = base.loc[base["重賞"]]

    base = base.loc[base["odds"].notna() & (base["odds"] > 0)]
    if base.empty:
        return {"買い方": f"乖離BOX(1+{n_gap})", "レース数": 0}

    g = base.groupby("レースID", sort=False)
    base["予測順位"] = g["pred"].rank(ascending=False, method="first")
    # 乖離 = 予測確率 × オッズ（市場勝率の逆数を掛けるのと同じ）
    base["gap"] = base["pred"] * base["odds"]

    is_axis = base["予測順位"] == 1
    # 軸を除いた中での乖離順位
    base["gap順位"] = base.loc[~is_axis].groupby("レースID", sort=False)["gap"].rank(
        ascending=False, method="first"
    )
    selected = is_axis | (base["gap順位"] <= n_gap)
    base["選択"] = selected.astype(int)

    # 実際の1〜3着が、選んだ3頭とちょうど一致すれば的中（BOXなので順序は不問）
    top3 = base.loc[base["着順"] <= 3]
    matched = top3.groupby("レースID")["選択"].sum()
    n_top3 = top3.groupby("レースID")["選択"].size()
    hit = ((matched == 3) & (n_top3 == 3))

    points = 1 + n_gap
    points = points * (points - 1) * (points - 2)  # 3頭BOX = 6点
    invest_per_race = points * BET_UNIT

    race = pd.DataFrame({"レースID": hit.index, "hit": hit.to_numpy()})
    if payout_df is not None:
        race = race.merge(payout_df, on="レースID", how="left")
        race = race.dropna(subset=["3連単払戻"])
    else:
        race["3連単払戻"] = np.nan

    if race.empty:
        return {"買い方": f"乖離BOX(1+{n_gap})", "レース数": 0}

    payout = np.where(race["hit"].to_numpy(), race["3連単払戻"].to_numpy(), 0.0)
    threshold = hot_threshold(invest_per_race)
    n_races = len(race)
    n_hits = int(race["hit"].sum())
    n_hot = int((payout >= threshold).sum())
    lo, hi = wilson_interval(n_hot, n_races)

    return {
        "買い方": f"乖離BOX(1+{n_gap})",
        "点数": points,
        "投資/R": invest_per_race,
        "レース数": n_races,
        "的中数": n_hits,
        "的中率": n_hits / n_races,
        "配当中央値": float(np.median(payout[payout > 0])) if n_hits else np.nan,
        "熱い基準": threshold,
        "熱い当たり": n_hot,
        "達成率": n_hot / n_races,
        "達成率下限": lo,
        "達成率上限": hi,
        "回収率": float(payout.sum() / (invest_per_race * n_races)),
        "収支": float(payout.sum() - invest_per_race * n_races),
    }


# v2 時点のベースライン（改善したかを機械的に判定するため）
V2_BASELINE = {
    "valid AUC": 0.7787,
    "上位3頭BOX 達成率": 0.01015,
    "上位5頭BOX 達成率": 0.02919,
    "乖離BOX 達成率": 0.00065,
}


def verdict(auc: float | None, table: pd.DataFrame, gap_result: dict) -> pd.DataFrame:
    """v2 のベースラインと比べて改善したかを判定する（依頼書セクション6）。"""
    def achievement(name: str) -> float:
        row = table.loc[table["買い方"] == name]
        return float(row["達成率"].iloc[0]) if len(row) else float("nan")

    rows = [
        ("valid AUC", auc, V2_BASELINE["valid AUC"]),
        ("上位3頭BOX 達成率", achievement("上位3頭BOX"), V2_BASELINE["上位3頭BOX 達成率"]),
        ("上位5頭BOX 達成率", achievement("上位5頭BOX"), V2_BASELINE["上位5頭BOX 達成率"]),
        ("乖離BOX 達成率", gap_result.get("達成率", float("nan")),
         V2_BASELINE["乖離BOX 達成率"]),
    ]
    out = pd.DataFrame(rows, columns=["指標", "今回", "v2ベースライン"])
    out["改善"] = out["今回"] > out["v2ベースライン"]
    out["差"] = out["今回"] - out["v2ベースライン"]
    return out


def _is_rate_column(col: str) -> bool:
    """パーセント表記にすべき列か。

    「倍率」は割合ではなく倍数なので除外する。
    「指標下限/指標上限」のような層の境界値もそのまま表示したいので、
    率の信頼区間だけを名前で拾う。
    """
    if col in ("倍率", "残す割合"):
        return False
    if col in ("達成率下限", "達成率上限"):
        return True
    return col.endswith("率") or col.endswith("率差")


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """パーセント表記に整えて見やすくする（表示用）。"""
    out = table.copy()
    for col in out.columns:
        base = col.split("_")[0]  # 「達成率_重賞」のような接尾辞つきに対応
        if _is_rate_column(col) or _is_rate_column(base):
            out[col] = (pd.to_numeric(out[col], errors="coerce") * 100).round(3)
    return out
