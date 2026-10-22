"""v8 の実行入口：有馬記念に向けた最後の改良の検証（毎週のコマンドとは別）。

    py -m src.v8 baseline          1   今のモデル・単勝人気・ランダムの上位6頭BOX（重賞・平場）
    py -m src.v8 odds-coverage     2a  単勝オッズ・各券種の払戻が年ごとにどこまでそろっているか
    py -m src.v8 levels            1.5 格・着差・推移・タイム指数（平場で AUC・log-loss・キャリブレーション）
    py -m src.v8 market            2b  モデルと市場の確率を混ぜる（Benter の2段階）と、オッズを特徴量に入れたモデル
    py -m src.v8 trifecta          2c・2d  3連単の確率（Harville・Henery）と、配当の推定の当たり具合
    py -m src.v8 bets              2e  買い目の選び方（平場）
    py -m src.v8 trackbias         3   当日の馬場の傾向（平場の後半のレース）

**v8 の新しい案（1.5・2・3）は平場だけで評価し、重賞の成績は出さない**（experiment.exploration_only）。
例外は 1（新しい案ではない）と、確率のキャリブレーション（全レース）。
学習の結果（fold ごとの予測）は data/v8/ に保存し、次のコマンドで使い回す（--refresh で作り直す）。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import bets, config, experiment, jvmap, market, predict, trifecta

TRIALS: list[str] = []          # このフェーズで平場で試した案（報告用）


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.t0 = time.perf_counter()

    def step(self, what: str) -> None:
        print(f"[{time.perf_counter() - self.t0:7.0f}秒] {what}", flush=True)


def _load(args, clock):
    clock.step("生データを読み込んで変換中 ...")
    ds = jvmap.build_dataset(args.data)
    rr = ds.race_result
    print(f"         確定レース: {rr['レース日付'].min():%Y-%m-%d} 〜 {rr['レース日付'].max():%Y-%m-%d}"
          f"（{rr['レースID'].nunique():,} レース）", flush=True)
    return ds


def _folds(args):
    from . import validate
    return validate.V5_FOLDS[: args.folds] if args.folds else validate.V5_FOLDS


def _oof(args, ds, variant):
    return experiment.load_or_run(ds, variant, folds=_folds(args), refresh=args.refresh,
                                  rounds=args.rounds)


def _show(table: pd.DataFrame, pct=(), money=(), digits=4) -> None:
    t = table.copy()
    for c in t.columns:
        if not pd.api.types.is_numeric_dtype(t[c]):
            continue
        if any(k in str(c) for k in pct):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        elif any(k in str(c) for k in money):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
        elif t[c].dtype.kind == "f" and t[c].dropna().apply(float.is_integer).all():
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")      # 件数など
        elif t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    with pd.option_context("display.max_rows", 400, "display.max_columns", 40, "display.width", 250):
        print(t.to_string(index=False))


def _save(args, table: pd.DataFrame, name: str) -> None:
    path = experiment.v8_dir() / f"{name}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存: {path}")


def _flat(t: pd.DataFrame) -> pd.DataFrame:
    return t.loc[~t["重賞"]]


def _trials_note() -> None:
    print(f"\n[記録] このコマンドで平場で比べた案: {len(TRIALS)} 個")
    for x in TRIALS:
        print(f"   - {x}")


# ---------------------------------------------------------------------------
# 1 基準との比較（新しい案ではないので重賞でも出す）
# ---------------------------------------------------------------------------
def _race_table(t: pd.DataFrame, pred, payout) -> pd.DataFrame:
    return trifecta.build_race_table(t, np.asarray(pred, dtype="float64"), payout)


def _box_metrics(race: pd.DataFrame) -> dict:
    r = trifecta.simulate(race, trifecta.box(6))
    hits, hot = r.get("的中数", 0), r.get("熱い当たり", 0)
    return {"レース数": r.get("レース数", 0), "的中率": r.get("的中率"), "達成率": r.get("達成率"),
            "下限": r.get("達成率下限"), "上限": r.get("達成率上限"),
            "的中のうち熱い割合": hot / hits if hits else np.nan, "回収率": r.get("回収率")}


def _hot_flags(race: pd.DataFrame) -> pd.Series:
    g = race.loc[race["3連単払戻"].notna()]
    k = np.minimum(6, g["出走頭数"])
    hit = (g["1着予測順位"] <= k) & (g["2着予測順位"] <= k) & (g["3着予測順位"] <= k)
    th = trifecta.hot_threshold(float(np.median(np.clip(k * (k - 1) * (k - 2), 0, None) * trifecta.BET_UNIT)))
    return pd.Series((hit & (g["3連単払戻"] >= th)).to_numpy(), index=g["レースID"].to_numpy())


def baseline_compare(t: pd.DataFrame, payout: pd.DataFrame, reps: int = 20, seed: int = 0):
    """A: モデルの上位6頭 / B: 単勝人気の上位6頭 / C: ランダムな6頭（reps 回の平均）。"""
    rng = np.random.default_rng(seed)
    pop = pd.to_numeric(t["人気"], errors="coerce").fillna(99)
    rows, paired = [], []
    for label, mask in [("重賞", t["重賞"]), ("平場", ~t["重賞"])]:
        sub = t.loc[mask]
        if sub.empty:
            continue
        ra = _race_table(sub, sub["score"], payout)
        rb = _race_table(sub, -pop.loc[sub.index], payout)
        rows.append({"区分": label, "買い方": "A 今のモデル", **_box_metrics(ra)})
        rows.append({"区分": label, "買い方": "B 単勝人気", **_box_metrics(rb)})
        rand = []
        for _ in range(reps):
            rand.append(_box_metrics(_race_table(sub, rng.random(len(sub)), payout)))
        rd = pd.DataFrame(rand)
        row = {"区分": label, "買い方": f"C ランダム（{reps}回の平均）", **rd.mean(numeric_only=True).to_dict()}
        row["下限"], row["上限"] = rd["達成率"].quantile(0.025), rd["達成率"].quantile(0.975)
        rows.append(row)
        fa, fb = _hot_flags(ra), _hot_flags(rb)
        common = fa.index.intersection(fb.index)
        pt = trifecta.paired_proportion_test(fa.loc[common].to_numpy(), fb.loc[common].to_numpy())
        paired.append({"区分": label, "レース数": pt["n"], "Aだけ熱い当たり": pt["xだけ"],
                       "Bだけ熱い当たり": pt["yだけ"], "両方": pt["両方"], "差（A−B）": pt["差"],
                       "差の下限": pt["差下限"], "差の上限": pt["差上限"], "p値": pt["p値"]})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def _cmd_baseline(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    oof = _oof(args, ds, "base")
    clock.step("比較中 ...")
    table, paired = baseline_compare(oof.table, ds.payout_df, reps=args.reps)
    print("\n=== 1 上位6頭BOX：今のモデル（A）・単勝人気（B）・ランダム（C）  as_of=day・馬体重は代用・454ラウンド ===")
    print("    C の下限・上限は、ランダムを繰り返したときの達成率の 2.5%〜97.5% の範囲")
    _show(table, pct=("率", "限", "割合"))
    print("\n=== A と B の対応のある比較（同じレースで、片方だけ熱い当たりになったレースの数） ===")
    _show(paired, pct=("差",), digits=4)
    _save(args, table, "baseline")
    _save(args, paired, "baseline_paired")
    return 0


# ---------------------------------------------------------------------------
# 2a データの確認
# ---------------------------------------------------------------------------
SPEC_ODDS = pd.DataFrame([
    ("単勝・複勝・枠連", "O1", "1993年6月〜", "0B31（単複枠）/ 0B30（全券種）/ 0B41（時系列・単複枠）"),
    ("馬連", "O2", "1993年6月〜", "0B32 / 0B30 / 0B42（時系列・馬連）"),
    ("ワイド", "O3", "1999年10月〜", "0B33 / 0B30"),
    ("馬単", "O4", "2002年6月〜", "0B34 / 0B30"),
    ("3連複", "O5", "2002年6月〜", "0B35 / 0B30"),
    ("3連単", "O6", "2004年8月〜", "0B36 / 0B30"),
], columns=["券種", "蓄積系のレコード", "セットアップで取れる期間（仕様書）", "当日の発売中のオッズ（速報系 JVRTOpen）"])


def odds_coverage(ds) -> pd.DataFrame:
    rr = ds.race_result
    year = pd.to_datetime(rr["レース日付"]).dt.year
    win = rr["単勝オッズ"].notna().groupby(year).mean()
    pay = ds.payout_df.copy()
    races = rr.drop_duplicates("レースID")[["レースID", "レース日付"]]
    pay = races.merge(pay, on="レースID", how="left")
    py = pd.to_datetime(pay["レース日付"]).dt.year
    out = pd.DataFrame({"単勝オッズ（馬ごと、SE）": win})
    for kind in ("3連単", "3連複", "馬単", "馬連"):
        col = f"{kind}組番一覧"
        has = pay[col].fillna("").astype(str).ne("") if col in pay.columns else pd.Series(False, index=pay.index)
        out[f"{kind}の払戻（レースごと、HR）"] = has.groupby(py).mean()
    out.index.name = "年"
    return out.reset_index()


def _cmd_odds_coverage(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    t = odds_coverage(ds)
    clock.step("完了\n")
    print("=== 2a 手元のデータで、年ごとに値がある割合 ===")
    _show(t, pct=("（",))
    print("\n=== 2a 手元に無いもの：確定オッズ（O1〜O6）は取得時に読み飛ばしている。仕様書で取れる期間 ===")
    _show(SPEC_ODDS)
    print("  ※ 当日の発売中のオッズは速報系の JVRTOpen(データ種別, キー)。キーはレース単位 \"YYYYMMDDJJKKHHRR\"。"
          "提供は発売開始以降（PAT の金曜発売があれば金曜19:00〜）、保存期間は1週間（docs/v8_validation.md）")
    _save(args, t, "odds_coverage")
    return 0


# ---------------------------------------------------------------------------
# 1.5 格・着差・推移・タイム指数
# ---------------------------------------------------------------------------
def compare_probabilities(tables: dict[str, pd.DataFrame], prob_col: str = "prob") -> pd.DataFrame:
    rows = []
    for name, t in tables.items():
        experiment.exploration_only(t, name)
        rows.append(market.summary_row(t, prob_col, name))
    return pd.DataFrame(rows)


def _importance(oof: experiment.OOF, new_cols: list[str], top: int = 30) -> pd.DataFrame:
    imp = oof.importance.groupby("特徴量", sort=False)["gain"].mean().sort_values(ascending=False)
    share = imp / imp.sum()
    t = pd.DataFrame({"順位": np.arange(1, len(imp) + 1), "特徴量": imp.index, "gainの割合": share.to_numpy()})
    t["新しい特徴量"] = t["特徴量"].isin(new_cols).map({True: "◯", False: ""})
    return t.loc[(t["順位"] <= top) | (t["新しい特徴量"] == "◯")]


def _cmd_levels(args) -> int:
    from . import levels
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    lev = _oof(args, ds, "levels")
    clock.step("比較中 ...\n")
    TRIALS.append("1.5 格・着差・推移・タイム指数を全部入れたモデル（levels）")
    tables = {"今のモデル（base）": _flat(base.table), "1.5 を足したモデル（levels）": _flat(lev.table)}
    comp = compare_probabilities(tables)
    print("=== 1.5 平場：入れる前と後（log-loss は小さいほど良い。AUC は大きいほど良い） ===")
    _show(comp)
    for name, t in [("base", base.table), ("levels", lev.table)]:
        print(f"\n--- キャリブレーション（全レース。重賞を含む・達成率ではない）: {name} ---")
        _show(market.calibration_table(t, "prob"), pct=("平均", "勝率"))
    print("\n=== gain 重要度（fold 平均、上位30と新しい特徴量の順位） ===")
    _show(_importance(lev, levels.level_feature_columns()), pct=("割合",))
    _save(args, comp, "levels_compare")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 2b Benter
# ---------------------------------------------------------------------------
def _stage1(args):
    return args.stage1


def market_tables(args, ds):
    s1 = _oof(args, ds, _stage1(args))
    odds_variant = {"base": "base_odds", "levels": "levels_odds"}[_stage1(args)]
    od = _oof(args, ds, odds_variant)
    blended, coefs = market.benter_walk_forward(s1.table)
    od_t = od.table.set_index(["レースID", "血統登録番号"])["prob"]
    blended["odds_model"] = od_t.reindex(pd.MultiIndex.from_frame(
        blended[["レースID", "血統登録番号"]])).to_numpy()
    blended["odds_model"] = market.normalize(blended["odds_model"], blended["レースID"])
    blended.attrs["stage1"] = s1.table
    return blended, coefs, od


def _cmd_market(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    blended, coefs, od = market_tables(args, ds)
    clock.step("比較中 ...\n")
    TRIALS.extend(["2b モデルと市場を混ぜる（Benter の2段階）",
                   f"2b 市場確率も特徴量に入れた1つのモデル（{od.variant}）"])
    print(f"=== 2b 2段目の係数（1段目 = {_stage1(args)}。推定はその fold より前の fold の平場） ===")
    print("    P ∝ exp(α·log モデル + β·log 市場)。α が 0 に近ければ、モデルは市場以上のことを知らない")
    _show(coefs)
    flat = _flat(blended)
    rows = [market.summary_row(experiment.exploration_only(flat, k), c, k) for k, c in
            [("モデルだけ", "prob"), ("市場だけ", "市場確率"), ("混ぜたもの（2段階）", "blend"),
             ("オッズも特徴量に入れたモデル", "odds_model")]]
    print("\n=== 2b 平場（fold 2 以降。2段目を推定できる fold）：log-loss・AUC・キャリブレーション誤差 ===")
    _show(pd.DataFrame(rows))
    print("\n=== キャリブレーション（全レース。重賞を含む・達成率ではない） ===")
    for k, c in [("混ぜたもの", "blend"), ("オッズも特徴量", "odds_model"), ("市場だけ", "市場確率")]:
        print(f"--- {k} ---")
        _show(market.calibration_table(blended, c), pct=("平均", "勝率"))
    print("\n=== オッズも特徴量に入れたモデルの gain 重要度（上位20） ===")
    _show(_importance(od, ["市場確率_log"], top=20), pct=("割合",))
    _save(args, pd.DataFrame(rows), "market_compare")
    _save(args, coefs, "market_coefs")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 2c・2d 3連単の確率と配当の推定
# ---------------------------------------------------------------------------
def fold_discounts(t: pd.DataFrame, prob_col: str) -> dict:
    """fold ごとに、それより前の fold の平場で γ・δ を推定する（walk-forward）。

    混ぜた確率（blend）は最初の fold には無い（2段目を推定できないため）。その fold の分は
    1段目の確率（prob）で代わりに推定に使う（γ・δ は「1着の確率から2・3着を出すときの割引」なので、
    確率の出どころが少し違っても大きくは変わらない）。
    """
    stage1 = t.attrs.get("stage1")
    full = t
    if stage1 is not None and prob_col == "blend":
        early = stage1.loc[~stage1["fold"].isin(set(t["fold"]))].copy()
        early["blend"] = early["prob"]
        full = pd.concat([early, t], ignore_index=True)
    elif stage1 is not None:
        early = stage1.loc[~stage1["fold"].isin(set(t["fold"]))]
        full = pd.concat([early, t], ignore_index=True)
    folds = list(dict.fromkeys(full["fold"]))
    out = {}
    for i, f in enumerate(folds):
        if f not in set(t["fold"]) or i == 0:
            continue
        fit = full.loc[full["fold"].isin(folds[:i]) & ~full["重賞"]]
        if fit.empty:
            continue
        out[f] = market.fit_discount(fit, prob_col)
    return out


def _cmd_trifecta(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    blended, _, _ = market_tables(args, ds)
    TRIALS.extend(["2c Harville", "2c Henery 型（γ・δ の割引）"])
    clock.step("γ・δ を推定中（混ぜた確率・市場の確率）...")
    rows, cal_rows, check_parts = [], [], []
    disc_b = fold_discounts(blended, "blend")
    disc_q = fold_discounts(blended, "市場確率")
    folds = [f for f in dict.fromkeys(blended["fold"]) if f in disc_b and f in disc_q]
    for f in folds:
        ft = blended.loc[blended["fold"] == f]
        flat = experiment.exploration_only(_flat(ft), "2c")
        g, d, n = disc_b[f]
        _, nll_h, n_h = market.trifecta_calibration(flat, "blend", 1.0, 1.0)
        _, nll_d, _ = market.trifecta_calibration(flat, "blend", g, d)
        rows.append({"検証fold": f, "γ": g, "δ": d, "推定レース数": n, "平場レース数": n_h,
                     "−log P（Harville）": nll_h, "−log P（Henery）": nll_d})
        gq, dq, _ = disc_q[f]
        check_parts.append(market.payout_estimate_check(flat, ds.payout_df, gq, dq))
        cal_rows.append((f, ft, g, d))
    print("\n=== 2c 実際の1〜3着の並びに付けた確率の −log の平均（平場。小さいほど良い） ===")
    _show(pd.DataFrame(rows))
    print("\n=== 2c 3連単の確率のキャリブレーション（全レース。重賞を含む・達成率ではない） ===")
    allt = pd.concat([x[1] for x in cal_rows])
    g_avg = float(np.mean([x[2] for x in cal_rows])) if cal_rows else 1.0
    d_avg = float(np.mean([x[3] for x in cal_rows])) if cal_rows else 1.0
    for name, gg, dd in [("Harville", 1.0, 1.0), (f"Henery（γ={g_avg:.2f}, δ={d_avg:.2f}、fold 平均）", g_avg, d_avg)]:
        cal, _, n = market.trifecta_calibration(allt, "blend", gg, dd)
        print(f"--- {name}（{n:,} レース）---")
        _show(cal[["区間", "組の数", "予測の平均", "実際の的中率", "実際の的中"]], digits=6)
    check = pd.concat(check_parts, ignore_index=True) if check_parts else pd.DataFrame()
    if not check.empty:
        print("\n=== 2d 推定配当（市場の確率・Henery 型・払戻率）と実際の払戻（平場の実際の的中組） ===")
        print("    比 = 実際 ÷ 推定。1 に近く、±50%以内が多いほど良い")
        _show(market.payout_check_summary(check), pct=("以内",))
        c3 = check.loc[check["券種"] == "3連単"].copy()
        c3["推定配当の帯"] = market.odds_band(c3["推定配当"])
        print("\n--- 3連単：推定配当の帯ごと ---")
        _show(market.payout_check_summary(c3, by="推定配当の帯"), pct=("以内",))
        print("\n--- 3連単：年ごと（払戻率の変更などで比がずれていないか） ---")
        _show(market.payout_check_summary(c3, by="年"), pct=("以内",))
        _save(args, market.payout_check_summary(check), "payout_estimate")
    _save(args, pd.DataFrame(rows), "trifecta_nll")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 2e 買い目
# ---------------------------------------------------------------------------
def evaluate_bets(blended: pd.DataFrame, payout: pd.DataFrame, disc_b: dict, disc_q: dict,
                  sample: int = 0, seed: int = 0, targets=(20_000, 30_000, 50_000, 100_000),
                  verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """平場のレースで各買い方を実際の払戻で評価する。戻り値: (成績, 熱さの一覧, 例)。"""
    flat = experiment.exploration_only(_flat(blended), "2e")
    flat = flat.loc[flat["fold"].isin(list(disc_b)) & flat["fold"].isin(list(disc_q))]
    rids = flat["レースID"].unique()
    if sample and len(rids) > sample:
        rids = np.random.default_rng(seed).choice(rids, size=sample, replace=False)
        flat = flat.loc[flat["レースID"].isin(set(rids))]
    t, starts = market.race_slices(flat)
    pay = payout.set_index("レースID")
    rows, heat, examples = [], [], []
    for n_done, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        rid = t["レースID"].iloc[a]
        if rid not in pay.index or b - a < 4:
            continue
        r = t.iloc[a:b]
        p = r["blend"].to_numpy(dtype="float64")
        q = r["市場確率"].to_numpy(dtype="float64")
        s1 = r["prob"].to_numpy(dtype="float64")
        posts = pd.to_numeric(r["馬番"], errors="coerce").to_numpy()
        if not (np.isfinite(p).all() and np.isfinite(q).all() and (q > 0).all() and np.isfinite(posts).all()):
            continue
        posts = posts.astype(int)
        wins = bets.winning_combos(pay.loc[rid])
        if not wins["3連単"]:
            continue
        f = r["fold"].iloc[0]
        gb, db, _ = disc_b[f]
        gq, dq, _ = disc_q[f]
        P = market.trifecta_tensor(p, gb, db)
        Q = market.combo_probabilities(market.trifecta_tensor(q, gq, dq))
        plans = [bets.box(s1, posts, name="S0 モデル（1段目）の上位6頭BOX"),
                 bets.box(p, posts, name="S1 混ぜた確率の上位6頭BOX"),
                 bets.hot_top(P, Q["3連単"], posts)]
        plans[2].name = "S2 熱い組だけ 120点"
        form, cards = bets.formation_cards(P, Q["3連単"], p, posts)
        form.name = f"S3 熱い組・フォーメーション{bets.MAX_CARDS_FORMATION}枚まで"
        plans.append(form)
        plans.append(bets.mixed(P, Q, p, posts, name="S4 券種と金額を混ぜる"))
        plans.append(bets.mixed(P, Q, p, posts, max_per_kind=bets.MARKS_PER_CARD * bets.MAX_CARDS_PER_KIND,
                                name=f"S4c 券種混合・券種ごと{bets.MARKS_PER_CARD * bets.MAX_CARDS_PER_KIND}点まで"))
        plans.append(bets.mixed(P, Q, p, posts, kinds=("3連単",), name="S5 3連単だけ（金額は変えてよい）"))
        for plan in plans:
            if np.isnan(plan.predicted):
                plan.predicted = bets.predicted_success(plan, P, Q, posts, bets.TARGET)
            rows.append({"レースID": rid, "買い方": plan.name, "投資": plan.invest,
                         "払戻": bets.realized(plan, wins), "予測の達成確率": plan.predicted,
                         "点数": len(plan.bets)})
        for tg in targets:
            plan = bets.mixed(P, Q, p, posts, target=tg, name=f"券種混合（基準 {tg:,}円）")
            heat.append({"基準": tg, "予測の達成確率": plan.predicted, "実際の払戻": bets.realized(plan, wins),
                         "投資": plan.invest})
        if len(examples) < 3:
            m = plans[4]
            examples.append({"レースID": rid, "S3 のカード": bets.card_text(cards),
                             "S4 の買い目": " / ".join(f"{b.kind} {'-'.join(map(str, b.combo))} {b.stake:,}円"
                                                      for b in m.bets)})
        if verbose and (n_done + 1) % 2000 == 0:
            print(f"    {n_done + 1:,} / {len(starts) - 1:,} レース", flush=True)
    res = pd.DataFrame(rows)
    summary = bets.summarize(res) if not res.empty else pd.DataFrame()
    h = pd.DataFrame(heat)
    heat_t = pd.DataFrame()
    if not h.empty:
        heat_t = h.groupby("基準").apply(lambda g: pd.Series({
            "レース数": len(g), "予測の達成確率（平均）": g["予測の達成確率"].mean(),
            "実際の達成率": (g["実際の払戻"] >= g.name).mean(),
            "下限": market.wilson(int((g["実際の払戻"] >= g.name).sum()), len(g))[0],
            "上限": market.wilson(int((g["実際の払戻"] >= g.name).sum()), len(g))[1],
            "平均投資": g["投資"].mean()}), include_groups=False).reset_index()
    return summary, heat_t, pd.DataFrame(examples)


def _cmd_bets(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    blended, _, _ = market_tables(args, ds)
    clock.step("γ・δ を推定中 ...")
    disc_b = fold_discounts(blended, "blend")
    disc_q = fold_discounts(blended, "市場確率")
    clock.step(f"買い目を計算中（平場、最大 {args.sample:,} レース）...")
    summary, heat, ex = evaluate_bets(blended, ds.payout_df, disc_b, disc_q, sample=args.sample)
    clock.step("完了\n")
    TRIALS.extend(["S1 混ぜた確率の上位6頭BOX", "S2 熱い組だけ 120点", "S3 フォーメーション3枚まで",
                   "S4 券種と金額を混ぜる", "S4c S4 に点数の上限", "S5 3連単だけ（金額可変）"])
    print("=== 2e 平場：買い方ごとの成績（予算 12,000円・基準 50,000円。配当は市場の確率から推定、評価は実際の払戻） ===")
    _show(summary, pct=("率", "限", "確率"), money=("投資",))
    print("\n=== 2e 熱さの一覧（券種混合。狙う払戻ごとに、予測の達成確率と実際） ===")
    _show(heat, pct=("確率", "達成率", "限"), money=("投資", "基準"))
    print("\n=== 例（会場で写す形） ===")
    for _, r in ex.iterrows():
        print(f"  {r['レースID']}: S3 {r['S3 のカード']}")
        print(f"            S4 {r['S4 の買い目']}")
    _save(args, summary, "bets_summary")
    _save(args, heat, "bets_heat")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 3 当日の馬場の傾向
# ---------------------------------------------------------------------------
LATE_RACE_FROM = 7          # 「後半のレース」= 7R 以降


def _race_no(t: pd.DataFrame) -> pd.Series:
    return t["レースID"].astype(str).str[-2:].astype(int)


def _cmd_trackbias(args) -> int:
    from . import trackbias
    clock = _Clock()
    ds = _load(args, clock)
    out = {v: _oof(args, ds, v) for v in ("base", "trackbias40", "trackbias80")}
    clock.step("比較中 ...\n")
    TRIALS.extend(["3 当日の傾向（40分前までに発走したレース）", "3 当日の傾向（打ち切り60分前＝80分前まで）"])
    rows = []
    for v, o in out.items():
        t = _flat(o.table)
        late = experiment.exploration_only(t.loc[_race_no(t) >= LATE_RACE_FROM], v)
        row = market.summary_row(late, "prob", f"{v}（{experiment.VARIANTS[v]['説明']}）")
        row.update(_box_metrics(_race_table(late, late["score"], ds.payout_df)))
        rows.append(row)
    comp = pd.DataFrame(rows)
    print(f"=== 3 平場の {LATE_RACE_FROM}R 以降：当日の傾向を入れない / 入れる（40分前・80分前） ===")
    _show(comp, pct=("率", "限", "割合"))
    print("\n=== 当日の傾向の特徴量の gain 重要度（40分前のモデル） ===")
    _show(_importance(out["trackbias40"], trackbias.track_bias_feature_columns(), top=20), pct=("割合",))
    _save(args, comp, "trackbias_compare")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m src.v8")
    parser.add_argument("--data", default=None, help="JV-Data CSV の場所（既定: data/jvlink）")
    parser.add_argument("--folds", type=int, default=None, help="先頭から何 fold だけ（動作確認用）")
    parser.add_argument("--refresh", action="store_true", help="保存済みの fold の予測を使わず作り直す")
    parser.add_argument("--rounds", type=int, default=predict.DEFAULT_ROUNDS, help="ラウンド数（既定 454）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="1 今のモデル・人気・ランダムの上位6頭BOX")
    p.add_argument("--reps", type=int, default=20, help="ランダムを何回繰り返すか")
    p.set_defaults(func=_cmd_baseline)
    sub.add_parser("odds-coverage", help="2a オッズ・払戻のそろい具合").set_defaults(func=_cmd_odds_coverage)
    sub.add_parser("levels", help="1.5 格・着差・推移・タイム指数（平場）").set_defaults(func=_cmd_levels)
    for name, func, hlp in [("market", _cmd_market, "2b モデルと市場の確率を混ぜる（平場）"),
                            ("trifecta", _cmd_trifecta, "2c・2d 3連単の確率・配当の推定"),
                            ("bets", _cmd_bets, "2e 買い目の選び方（平場）")]:
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--stage1", default="levels", choices=["base", "levels"],
                       help="1段目のモデル（既定 levels＝1.5 を足したもの）")
        if name == "bets":
            q.add_argument("--sample", type=int, default=10_000, help="平場から何レースを使うか（0 で全部）")
        q.set_defaults(func=func)
    sub.add_parser("trackbias", help="3 当日の馬場の傾向（平場の後半）").set_defaults(func=_cmd_trackbias)
    args = parser.parse_args(argv)
    TRIALS.clear()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
