"""v4：時系列クロスバリデーションによる再現性の検証。

■ このモジュールの目的は「新しい良い数字を出すこと」ではない
  v3で160通りの条件を探索した結果、達成率10%超えの条件がいくつか出た。
  しかし160通りも試せば、真に効果がなくても偶然良く見える条件は必ず出る。

  ここでやるのは、v3で得た**4つの仮説を別の期間に当てて、再現するかを見る**こと。

      H1  重賞は平場より達成率が高い
      H2  G1はG2/G3より達成率が高い
      H3  上位6頭BOX(120点)が予算1万円台で最良
      H4  レース選別では達成率は上がらない

■ 探索は一切しない
  H4で使う選別条件は v3 のトップ3を **定数として固定**してある（FIXED_CONDITIONS）。
  新しい条件を試せば多重比較の問題が悪化するだけなので、増やさないこと。

■ なぜ walk-forward なのか
  競馬は年々レース体系・馬場・馬の質が変わる。ランダムに分割する通常の CV は
  「未来を学習して過去を予測する」ことになり、実戦と条件が違ってしまう。
  そこで学習期間を伸ばしながら検証期間を前に進める walk-forward を使う。
  各 fold は「その時点で持っていたデータだけで学習し、次の数年を予測する」
  という実戦そのものの形になる。

■ fold は独立に扱う
  難易度指標の分位点も、fold ごとにその fold の検証データだけから計算する。
  全期間の分位点を使うと、fold 間で情報が共有されてしまう。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, features, preprocess, selection, trifecta


# ---------------------------------------------------------------------------
# fold の定義
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """学習期間と検証期間（どちらも年で指定、両端を含む）。"""

    name: str
    train_years: tuple[int, int]
    test_years: tuple[int, int]

    @property
    def label(self) -> str:
        return (f"{self.train_years[0] % 100:02d}-{self.train_years[1] % 100:02d}"
                f" → {self.test_years[0] % 100:02d}-{self.test_years[1] % 100:02d}")


DEFAULT_FOLDS = [
    Fold("1", (2003, 2010), (2011, 2013)),
    Fold("2", (2003, 2013), (2014, 2016)),
    Fold("3", (2003, 2016), (2017, 2019)),
    Fold("4", (2003, 2019), (2020, 2021)),
]


# ---------------------------------------------------------------------------
# H4 で検証する選別条件（v3のトップ3。**ここに条件を足さないこと**）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """複合選別条件。(指標, 高い側を残すか, 残す割合) の組。"""

    name: str
    rules: tuple[tuple[str, bool, float], ...] = field(default_factory=tuple)

    def apply(self, race: pd.DataFrame) -> pd.DataFrame:
        """条件を順に適用して、残ったレースを返す。

        分位点は **渡された race の中だけ**から計算する。
        fold ごとに呼べば、fold 間で情報が混ざらない。
        """
        out = race
        for metric, higher, keep in self.rules:
            if metric not in out.columns:
                return out.iloc[0:0]
            out = selection._select(out, metric, keep, higher)
        return out


FIXED_CONDITIONS = [
    Condition("エントロピー高50% × 最大確率低50%",
              (("エントロピー", True, 0.5), ("最大確率", False, 0.5))),
    Condition("1位2位差低75% × 出走頭数高50%",
              (("1位2位差", False, 0.75), ("出走頭数", True, 0.5))),
    Condition("エントロピー高75% × 出走頭数高50%",
              (("エントロピー", True, 0.75), ("出走頭数", True, 0.5))),
]

# H3 で比較する買い方（予算1万円台。ここも固定）
H3_STRATEGIES = [
    trifecta.formation(2, 6, 10),   # 80点 / 8,000円
    trifecta.formation(3, 6, 8),    # 90点 / 9,000円
    trifecta.box(6),                # 120点 / 12,000円
]

MAIN_STRATEGY = trifecta.box(6)          # 主役の買い方
CHEAP_STRATEGY = trifecta.formation(1, 4, 8)  # コスパ枠


# ---------------------------------------------------------------------------
# fold ごとの実行
# ---------------------------------------------------------------------------
def _slice_years(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    cols = config.resolve_columns(df)
    year = df[cols["date"]].dt.year
    return df.loc[(year >= years[0]) & (year <= years[1])]


def _default_model_fn(train_df, test_df, feature_cols):
    """既定の学習器（LightGBM）。戻り値は (予測確率, valid AUC, gain重要度)。"""
    from . import train as train_mod

    model = train_mod.train_lgb(train_df, test_df, feature_cols)
    pred = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    auc = None
    try:
        auc = float(model.best_score["valid"]["auc"])
    except (KeyError, TypeError):
        pass
    return pred, auc, train_mod.importance_table(model, feature_cols)


def run_fold(df: pd.DataFrame, fold: Fold, payout_df: pd.DataFrame,
             feature_cols: list[str], model_fn=None, verbose: bool = True) -> dict:
    """1 fold を実行し、検証用のレーステーブルまで作って返す。"""
    model_fn = model_fn or _default_model_fn

    train_df = _slice_years(df, fold.train_years)
    test_df = _slice_years(df, fold.test_years)
    if verbose:
        print(f"  fold{fold.name} {fold.label}: "
              f"train {len(train_df):,} / test {len(test_df):,}")
    if train_df.empty or test_df.empty:
        return {"fold": fold, "race": pd.DataFrame(), "auc": None}

    # 学習器は (pred, auc) か (pred, auc, 重要度) を返す（テスト用の偽物は前者でよい）
    out = model_fn(train_df, test_df, feature_cols)
    pred, auc = out[0], out[1]
    importance = out[2] if len(out) > 2 else None

    race = trifecta.build_race_table(test_df, pred, payout_df)
    race = selection.attach_difficulty(race, test_df, pred)
    return {"fold": fold, "race": race, "auc": auc, "importance": importance,
            "n_train": len(train_df), "n_test": len(test_df)}


def run_time_series_cv(df: pd.DataFrame, payout_df: pd.DataFrame,
                       folds: list[Fold] | None = None,
                       corner_df: pd.DataFrame | None = None,
                       lap_df: pd.DataFrame | None = None,
                       feature_cols: list[str] | None = None,
                       model_fn=None, verbose: bool = True) -> list[dict]:
    """全 fold を実行する（依頼A）。

    **特徴量は全期間から1度だけ作る。** fold ごとに作り直すと、
    序盤の馬の通算成績がリセットされて別物になってしまう。
    切るのは学習・検証の窓だけ。
    """
    folds = folds or DEFAULT_FOLDS

    # まだ特徴量が付いていなければここで付ける（付いていれば再利用）
    if "通算勝率" not in df.columns:
        if verbose:
            print("[v4] 前処理と特徴量作成（全期間で1回だけ）...")
        df = preprocess.basic_clean(df)
        df = features.add_all_features(df, corner_df=corner_df, lap_df=lap_df)
        df = preprocess.downcast(df)
        df = preprocess.to_category(df)

    feature_cols = feature_cols or features.feature_columns(df)
    if verbose:
        print(f"[v4] 特徴量 {len(feature_cols)} 個 / {len(folds)} fold を実行")

    return [run_fold(df, f, payout_df, feature_cols, model_fn, verbose) for f in folds]


# ---------------------------------------------------------------------------
# fold ごとの成績表（依頼A の出力）
# ---------------------------------------------------------------------------
def _rate_row(race: pd.DataFrame, strategy: trifecta.Strategy) -> dict:
    res = trifecta.simulate(race, strategy)
    return {
        "レース数": res.get("レース数", 0),
        "的中率": res.get("的中率"),
        "達成率": res.get("達成率"),
        "下限": res.get("達成率下限"),
        "上限": res.get("達成率上限"),
        "回収率": res.get("回収率"),
        "熱い当たり": res.get("熱い当たり", 0),
    }


def fold_summary(results: list[dict]) -> pd.DataFrame:
    """fold ごとの主要指標を1行にまとめる（依頼Aの出力形式）。"""
    rows = []
    for r in results:
        race = r["race"]
        if race.empty:
            continue
        graded = race.loc[race["重賞"]]
        flat = race.loc[~race["重賞"]]

        g = _rate_row(graded, MAIN_STRATEGY)
        f = _rate_row(flat, MAIN_STRATEGY)
        cheap = _rate_row(graded, CHEAP_STRATEGY)
        g1 = _rate_row(race.loc[race.get("格付け").eq("G1")], MAIN_STRATEGY) \
            if "格付け" in race.columns else {}

        rows.append({
            "fold": r["fold"].name,
            "学習": f"{r['fold'].train_years[0]}-{r['fold'].train_years[1]}",
            "検証": f"{r['fold'].test_years[0]}-{r['fold'].test_years[1]}",
            "AUC": r.get("auc"),
            "重賞R数": g["レース数"], "重賞達成率": g["達成率"], "重賞下限": g["下限"],
            "平場R数": f["レース数"], "平場達成率": f["達成率"], "平場下限": f["下限"],
            "G1R数": g1.get("レース数"), "G1達成率": g1.get("達成率"),
            "G1下限": g1.get("下限"),
            "F148達成率": cheap["達成率"], "F148下限": cheap["下限"],
        })
    return pd.DataFrame(rows)


def fold_grade_table(results: list[dict]) -> pd.DataFrame:
    """fold × グレード別の達成率（依頼A）。"""
    rows = []
    for r in results:
        race = r["race"]
        if race.empty or "格付け" not in race.columns:
            continue
        for grade in config.GRADED_VALUES:
            subset = race.loc[race["格付け"] == grade]
            if subset.empty:
                continue
            row = _rate_row(subset, MAIN_STRATEGY)
            row.update({"fold": r["fold"].name, "格付け": grade})
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cols = ["fold", "格付け", "レース数", "的中率", "達成率", "下限", "上限", "回収率"]
    return out[cols]


# ---------------------------------------------------------------------------
# 依頼B：仮説ごとの再現性判定
# ---------------------------------------------------------------------------
def _pool(results: list[dict], mask_fn, strategy: trifecta.Strategy) -> tuple[int, int]:
    """全 fold をプールして (熱い当たり数, レース数) を返す。"""
    hits = races = 0
    for r in results:
        race = r["race"]
        if race.empty:
            continue
        subset = mask_fn(race)
        if subset.empty:
            continue
        res = trifecta.simulate(subset, strategy)
        hits += res.get("熱い当たり", 0)
        races += res.get("レース数", 0)
    return hits, races


def test_h1(results: list[dict]) -> dict:
    """H1：重賞は平場より達成率が高いか。"""
    per_fold = []
    for r in results:
        race = r["race"]
        if race.empty:
            continue
        g = _rate_row(race.loc[race["重賞"]], MAIN_STRATEGY)
        f = _rate_row(race.loc[~race["重賞"]], MAIN_STRATEGY)
        per_fold.append({
            "fold": r["fold"].name,
            "重賞達成率": g["達成率"], "重賞R数": g["レース数"],
            "平場達成率": f["達成率"], "平場R数": f["レース数"],
            "重賞が上": bool((g["達成率"] or 0) > (f["達成率"] or 0)),
        })

    gh, gn = _pool(results, lambda x: x.loc[x["重賞"]], MAIN_STRATEGY)
    fh, fn = _pool(results, lambda x: x.loc[~x["重賞"]], MAIN_STRATEGY)
    test = trifecta.two_proportion_test(gh, gn, fh, fn)

    table = pd.DataFrame(per_fold)
    wins = int(table["重賞が上"].sum()) if not table.empty else 0
    return {
        "仮説": "H1: 重賞 > 平場",
        "per_fold": table,
        "再現fold数": f"{wins}/{len(table)}",
        "プール重賞": f"{gh}/{gn}",
        "プール平場": f"{fh}/{fn}",
        "プール重賞達成率": gh / gn if gn else np.nan,
        "プール平場達成率": fh / fn if fn else np.nan,
        **test,
    }


def test_h2(results: list[dict]) -> dict:
    """H2：G1はG2/G3より達成率が高いか。

    G1は fold あたり30〜60レース程度しかない。
    Wilson 区間は必ず±数%以上の幅になるので、
    「何fold でG1が最高だったか」という粗い見方が主になる。
    """
    grade_table = fold_grade_table(results)
    per_fold = []
    if not grade_table.empty:
        for fold_name, g in grade_table.groupby("fold", sort=True):
            sub = g.loc[g["格付け"].isin(["G1", "G2", "G3"])].dropna(subset=["達成率"])
            if sub.empty:
                continue
            best = sub.loc[sub["達成率"].idxmax(), "格付け"]
            row = {"fold": fold_name, "最高": best}
            for grade in ["G1", "G2", "G3"]:
                one = sub.loc[sub["格付け"] == grade]
                row[f"{grade}達成率"] = float(one["達成率"].iloc[0]) if len(one) else np.nan
                row[f"{grade}R数"] = int(one["レース数"].iloc[0]) if len(one) else 0
            per_fold.append(row)

    table = pd.DataFrame(per_fold)
    g1_best = int((table["最高"] == "G1").sum()) if not table.empty else 0

    # プールして G1 vs G2+G3 を検定
    def g1_mask(race):
        return race.loc[race.get("格付け").eq("G1")] if "格付け" in race else race.iloc[0:0]

    def other_mask(race):
        return race.loc[race.get("格付け").isin(["G2", "G3"])] if "格付け" in race else race.iloc[0:0]

    ah, an = _pool(results, g1_mask, MAIN_STRATEGY)
    bh, bn = _pool(results, other_mask, MAIN_STRATEGY)
    test = trifecta.two_proportion_test(ah, an, bh, bn)

    return {
        "仮説": "H2: G1 > G2/G3",
        "per_fold": table,
        "G1が最高だったfold": f"{g1_best}/{len(table)}",
        "プールG1": f"{ah}/{an}",
        "プールG2G3": f"{bh}/{bn}",
        "プールG1達成率": ah / an if an else np.nan,
        "プールG2G3達成率": bh / bn if bn else np.nan,
        **test,
    }


def test_h3(results: list[dict]) -> dict:
    """H3：予算1万円台で上位6頭BOXが最良か。"""
    rows = []
    for r in results:
        race = r["race"]
        if race.empty:
            continue
        graded = race.loc[race["重賞"]]
        for s in H3_STRATEGIES:
            row = _rate_row(graded, s)
            row.update({"fold": r["fold"].name, "買い方": s.name, "点数": s.points})
            rows.append(row)

    table = pd.DataFrame(rows)
    # 重賞が1つもない fold は達成率が NaN になるので、比較対象から外す
    valid = table.dropna(subset=["達成率", "下限"]) if not table.empty else table
    if valid.empty:
        return {"仮説": "H3: 上位6頭BOXが最良（予算1万円台）", "per_fold": table,
                "達成率で最良だったfold": "0/0", "下限で最良だったfold": "0/0",
                "各foldの最良（達成率）": {}, "各foldの最良（下限）": {}}

    # 各 fold で「達成率」「達成率下限」それぞれの最良を見る
    best_rate = valid.loc[valid.groupby("fold")["達成率"].idxmax()]
    best_lower = valid.loc[valid.groupby("fold")["下限"].idxmax()]
    n_folds = valid["fold"].nunique()
    wins_rate = int((best_rate["買い方"] == MAIN_STRATEGY.name).sum())
    wins_lower = int((best_lower["買い方"] == MAIN_STRATEGY.name).sum())

    return {
        "仮説": "H3: 上位6頭BOXが最良（予算1万円台）",
        "per_fold": table[["fold", "買い方", "点数", "レース数", "的中率",
                           "達成率", "下限", "上限", "回収率"]],
        "達成率で最良だったfold": f"{wins_rate}/{n_folds}",
        "下限で最良だったfold": f"{wins_lower}/{n_folds}",
        "各foldの最良（達成率）": dict(zip(best_rate["fold"], best_rate["買い方"])),
        "各foldの最良（下限）": dict(zip(best_lower["fold"], best_lower["買い方"])),
    }


def test_h4(results: list[dict],
            conditions: list[Condition] | None = None) -> dict:
    """H4：固定した選別条件で達成率が上がるか（依頼B）。

    判定の考え方:
      条件ごとに「基準（全重賞）より達成率下限が上がった fold 数」を数える。
      fold1では条件Aが良く fold2では条件Bが良い、という状態は
      「どの条件も一貫して効いていない」＝ノイズを拾っていた証拠になる。
    """
    conditions = conditions or FIXED_CONDITIONS
    rows = []
    for r in results:
        race = r["race"]
        if race.empty:
            continue
        graded = race.loc[race["重賞"]]
        base = _rate_row(graded, MAIN_STRATEGY)
        base.update({"fold": r["fold"].name, "条件": "全重賞（基準）"})
        rows.append(base)

        for cond in conditions:
            subset = cond.apply(graded)
            row = _rate_row(subset, MAIN_STRATEGY)
            row.update({"fold": r["fold"].name, "条件": cond.name})
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return {"仮説": "H4: 選別は効かない", "per_fold": table}

    # 条件ごとに、基準を上回った fold 数を数える
    summary = []
    baseline = table.loc[table["条件"] == "全重賞（基準）"].set_index("fold")
    for cond in conditions:
        sub = table.loc[table["条件"] == cond.name].set_index("fold")
        common = sub.index.intersection(baseline.index)
        better_lower = int((sub.loc[common, "下限"] > baseline.loc[common, "下限"]).sum())
        better_rate = int((sub.loc[common, "達成率"] > baseline.loc[common, "達成率"]).sum())
        summary.append({
            "条件": cond.name,
            "下限が改善したfold": f"{better_lower}/{len(common)}",
            "達成率が改善したfold": f"{better_rate}/{len(common)}",
            "平均レース数": float(sub.loc[common, "レース数"].mean()),
        })

    summary_df = pd.DataFrame(summary)
    # 全 fold で下限が改善した条件があるか
    consistent = [s["条件"] for s in summary
                  if s["下限が改善したfold"].split("/")[0]
                  == s["下限が改善したfold"].split("/")[1]
                  and s["下限が改善したfold"].split("/")[1] != "0"]

    return {
        "仮説": "H4: レース選別では上がらない",
        "per_fold": table[["fold", "条件", "レース数", "的中率", "達成率",
                           "下限", "上限", "回収率"]],
        "summary": summary_df,
        "全foldで下限が改善した条件": consistent,
        "結論": ("選別は一貫して効いていない（H4を支持）" if not consistent
                 else f"一貫して効いた条件あり: {consistent}"),
    }


# ---------------------------------------------------------------------------
# 依頼C：結論の要約
# ---------------------------------------------------------------------------
def _last_test_date(results: list[dict]) -> str:
    """検証に使ったレースの最終日（限界の説明に使う）。"""
    dates = [pd.to_datetime(r["race"]["日付"]).max()
             for r in results if not r["race"].empty and "日付" in r["race"]]
    return f"{max(dates):%Y-%m-%d}" if dates else "不明"


def recommend(results: list[dict], h1: dict, h2: dict, h3: dict, h4: dict) -> dict:
    """検証結果から実戦戦略を1つに絞る（依頼C）。

    ここは「数字を見て人が決める」部分を、明示的なルールとして書き下したもの。
    ルールが気に入らなければ、判定の根拠（h1〜h4の中身）を見て自分で覆せる。
    """
    # --- 買う対象 ---------------------------------------------------------
    if h1.get("有意") and (h1.get("差") or 0) > 0:
        target = "重賞のみ"
        target_reason = (f"重賞 {h1['プール重賞達成率']:.2%} vs "
                         f"平場 {h1['プール平場達成率']:.2%}、"
                         f"p={h1['p値']:.4f} で有意")
    else:
        target = "重賞のみ（暫定）"
        target_reason = (f"プールしても有意差なし（p={h1.get('p値', float('nan')):.4f}）。"
                         "重賞優位は再現しなかった")

    # G1に絞るべきか
    if h2.get("有意") and (h2.get("差") or 0) > 0:
        target += "、とくにG1"
        g1_note = f"G1 {h2['プールG1達成率']:.2%} vs G2/G3 {h2['プールG2G3達成率']:.2%} で有意"
    else:
        g1_note = ("G1に絞る根拠は得られなかった（母数が小さく区間が広い）。"
                   "有馬記念を狙うことと、G1が統計的に有利なことは別問題")

    # --- 買い方 -----------------------------------------------------------
    best_by_lower = h3.get("各foldの最良（下限）", {})
    votes = pd.Series(list(best_by_lower.values())).value_counts() if best_by_lower else pd.Series(dtype=int)
    if not votes.empty:
        strategy_name = votes.index[0]
        counts = {k: int(v) for k, v in votes.items()}  # numpy 型を素の int に
        strategy_reason = f"下限で最良だったfold数: {counts}"
    else:
        strategy_name = MAIN_STRATEGY.name
        strategy_reason = "fold結果が取れなかったため既定"

    # --- プールした達成率と区間 --------------------------------------------
    strategy = next((s for s in H3_STRATEGIES if s.name == strategy_name), MAIN_STRATEGY)
    hits, races = _pool(results, lambda x: x.loc[x["重賞"]], strategy)
    lo, hi = trifecta.wilson_interval(hits, races)

    return {
        "買う対象": target,
        "対象の根拠": target_reason,
        "G1について": g1_note,
        "買い方": strategy_name,
        "点数": strategy.points,
        "投資額": strategy.points * trifecta.BET_UNIT,
        "買い方の根拠": strategy_reason,
        "期待達成率（全fold プール）": hits / races if races else np.nan,
        "信頼区間": (lo, hi),
        "検証レース数": races,
        "選別条件を使うか": ("使わない" if not h4.get("全foldで下限が改善した条件")
                             else f"使う: {h4['全foldで下限が改善した条件']}"),
        "限界": [
            "3連単の発売は2004年から。それ以前の期間は検証できていない",
            "パドック評価の効果は未検証（過去データが存在しないため）",
            f"検証データは {_last_test_date(results)} まで。それ以降の馬場・レース体系の変化は反映されていない",
            "達成率の区間はfoldをプールした値。fold間のばらつきは別途 fold_summary を見ること",
            "3連単の配当は分布の裾が重い。少数の大穴が達成率を左右する点に注意",
        ],
    }


# ---------------------------------------------------------------------------
# まとめて実行
# ---------------------------------------------------------------------------
def run_v4(df: pd.DataFrame, payout_df: pd.DataFrame | None = None,
           folds: list[Fold] | None = None,
           corner_df: pd.DataFrame | None = None,
           lap_df: pd.DataFrame | None = None,
           corner_path: str | None = None, lap_path: str | None = None,
           odds_path: str | None = None,
           model_fn=None, verbose: bool = True) -> dict:
    """依頼A〜Cを一括で実行する。

    補助データはパスでも DataFrame でも渡せる。
    fold ごとにモデルを学習し直すので、実行時間は run_pipeline の fold 数倍かかる。
    重い場合は folds を3つに減らしてよい（依頼書でも最低3foldとされている）。
    """
    from . import corner as corner_mod
    from . import pace as pace_mod
    from . import payout as payout_mod

    if corner_df is None and corner_path is not None:
        corner_df = corner_mod.load_corner(corner_path)
    if lap_df is None and lap_path is not None:
        lap_df = pace_mod.load_laptime(lap_path)
    if payout_df is None and odds_path is not None:
        payout_df = payout_mod.load_payouts(odds_path)
    if payout_df is None:
        raise ValueError("payout_df か odds_path のどちらかが必要です（配当なしでは検証できません）")

    results = run_time_series_cv(df, payout_df, folds=folds, corner_df=corner_df,
                                 lap_df=lap_df, model_fn=model_fn, verbose=verbose)

    h1, h2, h3, h4 = test_h1(results), test_h2(results), test_h3(results), test_h4(results)
    out = {
        "results": results,
        "fold_summary": fold_summary(results),
        "grade_table": fold_grade_table(results),
        "H1": h1, "H2": h2, "H3": h3, "H4": h4,
        "recommendation": recommend(results, h1, h2, h3, h4),
    }
    if verbose:
        print_v4(out)
    return out


def print_v4(out: dict) -> None:
    """v4の結果を表示する。"""
    fmt = trifecta.format_table

    print("\n=== 依頼A：fold ごとの成績 ===")
    summary = out["fold_summary"].copy()
    for col in [c for c in summary.columns if "達成率" in c or "下限" in c]:
        summary[col] = (summary[col] * 100).round(3)
    print(summary.to_string(index=False))

    if not out["grade_table"].empty:
        print("\n=== 依頼A：fold × グレード別 ===")
        print(fmt(out["grade_table"].rename(
            columns={"下限": "達成率下限", "上限": "達成率上限"})).to_string(index=False))

    for key in ["H1", "H2", "H3", "H4"]:
        h = out[key]
        print(f"\n=== 依頼B：{h['仮説']} ===")
        table = h.get("per_fold")
        if table is not None and not table.empty:
            shown = table.copy()
            for col in [c for c in shown.columns
                        if any(k in c for k in ("達成率", "下限", "上限", "的中率", "回収率"))]:
                shown[col] = (pd.to_numeric(shown[col], errors="coerce") * 100).round(3)
            print(shown.to_string(index=False))
        for k, v in h.items():
            if k in ("仮説", "per_fold", "summary"):
                continue
            print(f"  {k}: {v}")
        if isinstance(h.get("summary"), pd.DataFrame) and not h["summary"].empty:
            print(h["summary"].to_string(index=False))

    print("\n=== 依頼C：実戦で採用する戦略 ===")
    rec = out["recommendation"]
    for k, v in rec.items():
        if k == "限界":
            print("  限界:")
            for line in v:
                print(f"    - {line}")
        elif k == "信頼区間":
            print(f"  {k}: {v[0]:.2%} 〜 {v[1]:.2%}")
        elif k == "期待達成率（全fold プール）":
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")


# ===========================================================================
# v5：JRA-VAN データでの再検証
# ===========================================================================
# データ期間が 2026 年まで延びたので fold を1つ増やす（依頼書 v5 の例どおり）。
# 学習開始年は v4 と同じ 2003 年に揃える（比較のため）。
V5_FOLDS = [
    Fold("1", (2003, 2010), (2011, 2013)),
    Fold("2", (2003, 2013), (2014, 2016)),
    Fold("3", (2003, 2016), (2017, 2019)),
    Fold("4", (2003, 2019), (2020, 2022)),
    Fold("5", (2003, 2022), (2023, 2026)),
]

# v4（Kaggle データ・4fold）で確定した値。依頼書 v5 の表より。
# 達成率の熱い当たり数は 6.13% × 1,337 から逆算した 82（82/1337 = 6.133%）。
# 正確な件数が手元にあれば V4_REFERENCE["pooled_hits"] を書き換えること。
V4_REFERENCE = {
    "pooled_hits": 82,
    "pooled_n": 1337,
    "pooled_rate": 0.0613,
    "pooled_ci": (0.0497, 0.0755),
    "H1_graded": 0.0613,
    "H1_flat": 0.0347,
    "H1_p": 2.7e-07,
    "H2_G1": 0.0574,
    "H2_G2G3": 0.0627,
    "H3": "4/4",
    "H4_consistent": 0,
}

# 自前の脚質特徴量と、JRA-VAN 公式の脚質判定から作った特徴量（重要度の比較用）
OWN_STYLE_FEATURES = ["脚質スコア", "脚質スコア直近3走", "前走4角相対位置",
                      "好走時脚質スコア", "脚質スコア_レース内平均差", "脚質安定度"]


def recent_period_check(results: list[dict], since: str = "2021-08-01",
                        v4_hits: int | None = None, v4_n: int | None = None) -> dict:
    """依頼書 v5 セクション8 の成功判定。

    2021年8月以降（Kaggle データに無かった期間）の重賞・上位6頭BOXの達成率が、
    v4 のプール値より**有意に低くないか**を、v4 と同じ two_proportion_test で判定する。

    「下限5%を超えたか」のような固定値では判定しない（依頼書の指示）。
    比べる相手は v4 の点推定と母数で、差の検定を行う。
    """
    v4_hits = V4_REFERENCE["pooled_hits"] if v4_hits is None else v4_hits
    v4_n = V4_REFERENCE["pooled_n"] if v4_n is None else v4_n
    cutoff = pd.Timestamp(since)

    def recent(race: pd.DataFrame) -> pd.DataFrame:
        return race.loc[pd.to_datetime(race["日付"]) >= cutoff]

    gh, gn = _pool(results, lambda r: recent(r).loc[lambda x: x["重賞"]], MAIN_STRATEGY)
    fh, fn = _pool(results, lambda r: recent(r).loc[lambda x: ~x["重賞"]], MAIN_STRATEGY)

    vs_v4 = trifecta.two_proportion_test(gh, gn, v4_hits, v4_n)
    h1_recent = trifecta.two_proportion_test(gh, gn, fh, fn)
    lo, hi = trifecta.wilson_interval(gh, gn)

    collapsed = bool(vs_v4["有意"] and vs_v4["差"] < 0)
    h1_kept = bool(h1_recent["差"] > 0) if gn and fn else False
    return {
        "期間": f"{since} 以降",
        "重賞レース数": gn,
        "熱い当たり": gh,
        "達成率": gh / gn if gn else np.nan,
        "信頼区間": (lo, hi),
        "v4プール": f"{v4_hits}/{v4_n} = {v4_hits / v4_n:.2%}",
        "v4との差": vs_v4["差"],
        "v4との差の区間": (vs_v4["差下限"], vs_v4["差上限"]),
        "v4との差のp値": vs_v4["p値"],
        "判定（達成率）": ("崩れた：v4より有意に低い" if collapsed
                       else "維持：v4より有意に低くはない"),
        "平場達成率": fh / fn if fn else np.nan,
        "平場レース数": fn,
        "H1（重賞>平場）のp値": h1_recent["p値"],
        "判定（H1）": ("維持：重賞が平場を上回る（有意）" if h1_kept and h1_recent["有意"]
                     else "維持（ただし有意差なし）" if h1_kept
                     else "崩れた：重賞が平場を上回っていない"),
    }


def style_importance_table(results: list[dict]) -> pd.DataFrame:
    """自前の脚質特徴量 vs 公式脚質判定の特徴量を、gain 重要度で比べる（依頼C-3）。

    fold ごとの gain を合計し、全特徴量の中での順位も出す。
    どちらを残すかは「データで決める」方針なので、ここでは並べるだけ。
    """
    from . import corner as corner_mod

    frames = [r["importance"].assign(fold=r["fold"].name)
              for r in results if r.get("importance") is not None]
    if not frames:
        return pd.DataFrame()
    imp = pd.concat(frames, ignore_index=True)
    total = imp.groupby("feature", as_index=False)["gain"].sum()
    total["順位"] = total["gain"].rank(ascending=False, method="min").astype(int)
    total["全体に占める割合"] = total["gain"] / total["gain"].sum()

    official = set(corner_mod.official_style_feature_columns())
    own = set(OWN_STYLE_FEATURES)
    total["系統"] = np.where(total["feature"].isin(official), "公式脚質判定",
                           np.where(total["feature"].isin(own), "自前の脚質", None))
    out = total.loc[total["系統"].notna()].sort_values("gain", ascending=False)
    return out[["系統", "feature", "gain", "順位", "全体に占める割合"]].reset_index(drop=True)


def compare_with_v4(out: dict) -> pd.DataFrame:
    """v4（Kaggle）と v5（JRA-VAN）の主要な数字を並べる（依頼D）。"""
    h1, h2, h3, h4 = out["H1"], out["H2"], out["H3"], out["H4"]
    rec = out["recommendation"]
    lo, hi = rec["信頼区間"]
    aucs = [r.get("auc") for r in out["results"] if r.get("auc") is not None]
    rows = [
        ("重賞・上位6頭BOX 達成率（プール）",
         f"{V4_REFERENCE['pooled_rate']:.2%} [{V4_REFERENCE['pooled_ci'][0]:.2%}, {V4_REFERENCE['pooled_ci'][1]:.2%}]",
         f"{rec['期待達成率（全fold プール）']:.2%} [{lo:.2%}, {hi:.2%}]"),
        ("検証レース数（重賞）", f"{V4_REFERENCE['pooled_n']:,}", f"{rec['検証レース数']:,}"),
        ("H1 重賞 vs 平場",
         f"{V4_REFERENCE['H1_graded']:.2%} vs {V4_REFERENCE['H1_flat']:.2%}（p={V4_REFERENCE['H1_p']:.1e}）",
         f"{h1['プール重賞達成率']:.2%} vs {h1['プール平場達成率']:.2%}（p={h1['p値']:.1e}）"
         f"／再現 {h1['再現fold数']}"),
        ("H2 G1 vs G2/G3",
         f"{V4_REFERENCE['H2_G1']:.2%} vs {V4_REFERENCE['H2_G2G3']:.2%}（否定）",
         f"{h2['プールG1達成率']:.2%} vs {h2['プールG2G3達成率']:.2%}（p={h2['p値']:.2f}）"),
        ("H3 上位6頭BOXが最良（下限）", V4_REFERENCE["H3"], h3.get("下限で最良だったfold", "")),
        ("H4 全foldで効いた選別条件", str(V4_REFERENCE["H4_consistent"]),
         str(len(h4.get("全foldで下限が改善した条件", [])))),
        ("valid AUC（fold平均）", "0.7821（v3・単一分割）",
         f"{np.mean(aucs):.4f}" if aucs else "—"),
    ]
    return pd.DataFrame(rows, columns=["項目", "v4（Kaggle）", "v5（JRA-VAN）"])


def run_v5(dataset, folds: list[Fold] | None = None, since: str = "2021-08-01",
           ablation: bool = False, model_fn=None, verbose: bool = True) -> dict:
    """JRA-VAN データで v4 と同じ検証を回し、v4 と比較する（依頼D）。

    Parameters
    ----------
    dataset  : jvmap.build_dataset() の戻り値
    ablation : True なら「公式脚質特徴量なし」でも CV を回して AUC を比べる
               （学習が fold 数ぶん追加で走るので時間は約2倍）
    """
    from . import corner as corner_mod

    folds = folds or V5_FOLDS
    if verbose:
        print(dataset.summary())
        print("[v5] 前処理と特徴量作成（全期間で1回だけ）...")
    df = preprocess.basic_clean(dataset.race_result)
    df = features.add_all_features(df, lap_df=dataset.lap_df)
    df = preprocess.downcast(df)
    df = preprocess.to_category(df)

    out = run_v4(df, payout_df=dataset.payout_df, folds=folds,
                 model_fn=model_fn, verbose=False)
    out["recent"] = recent_period_check(out["results"], since=since)
    out["style_importance"] = style_importance_table(out["results"])
    out["comparison"] = compare_with_v4(out)

    if ablation:
        official = set(corner_mod.official_style_feature_columns())
        base_cols = [c for c in features.feature_columns(df) if c not in official]
        base = run_time_series_cv(df, dataset.payout_df, folds=folds,
                                  feature_cols=base_cols, model_fn=model_fn, verbose=False)
        out["ablation"] = pd.DataFrame([
            {"fold": a["fold"].name, "AUC（公式脚質あり）": a.get("auc"),
             "AUC（公式脚質なし）": b.get("auc")}
            for a, b in zip(out["results"], base)
        ])
        out["ablation"]["差"] = out["ablation"]["AUC（公式脚質あり）"] - out["ablation"]["AUC（公式脚質なし）"]

    if verbose:
        print_v5(out)
    return out


def print_v5(out: dict) -> None:
    print_v4(out)

    print("\n=== v5：v4 との比較 ===")
    print(out["comparison"].to_string(index=False))

    print("\n=== v5 成功判定：2021年8月以降 ===")
    for k, v in out["recent"].items():
        if isinstance(v, tuple):
            print(f"  {k}: {v[0]:.2%} 〜 {v[1]:.2%}")
        elif isinstance(v, float) and ("率" in k or "差" == k[-1:]):
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")

    if not out["style_importance"].empty:
        print("\n=== 依頼C-3：自前の脚質 vs 公式脚質判定（gain 重要度、全fold合計） ===")
        print(out["style_importance"].to_string(index=False))
    if "ablation" in out:
        print("\n=== 公式脚質特徴量の有無による AUC ===")
        print(out["ablation"].to_string(index=False))
