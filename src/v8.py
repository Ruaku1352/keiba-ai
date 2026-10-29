"""v8 の実行入口：有馬記念に向けた最後の改良の検証（毎週のコマンドとは別）。

    py -m src.v8 baseline          1   今のモデル・単勝人気・ランダムの上位6頭BOX（重賞・平場）
    py -m src.v8 odds-coverage     2a  単勝オッズ・各券種の払戻が年ごとにどこまでそろっているか
    py -m src.v8 levels            1.5 格・着差・推移・タイム指数（平場で AUC・log-loss・キャリブレーション）
    py -m src.v8 market            2b  モデルと市場の確率を混ぜる（Benter の2段階）と、オッズを特徴量に入れたモデル
    py -m src.v8 trifecta          2c・2d  3連単の確率（Harville・Henery）と、配当の推定の当たり具合
    py -m src.v8 bets              2e  買い目の選び方（平場）
    py -m src.v8 trackbias         3   当日の馬場の傾向（平場の後半のレース）
    py -m src.v8 tilt              3b・3c  市場の確率に当日の傾向・開催の進み具合の傾向で傾きをつける（3b+3c も）
    py -m src.v8 stakes            券種混合の金額の余裕・スマッピーの点数の上限
    py -m src.v8 rating            v8c レーティング（Plackett-Luce 型・AtCoder 型）を市場の確率への傾きに
    py -m src.v8 final-compare     今の本命の買い方で、確率の出し方を平場の全レースで比べる（夜に回す）
    py -m src.v8 real-odds         v8d 本物の確定オッズで券種混合の金額を決める（平場サンプル）
    py -m src.v8 growth-check      v8d 市場＋3b＋伸び と 市場＋3b を平場の全レースで（採用の判定つき）
    py -m src.v8 odds-targets      過去の確定オッズを取り直すレースの一覧（重賞＋平場の検証用5,000）
    py -m src.v8 odds-drift        速報オッズ（rt-odds）と確定オッズ（fetch-odds）のずれ

**v8 の新しい案（1.5・2・3）は平場だけで評価し、重賞の成績は出さない**（experiment.exploration_only）。
例外は 1（新しい案ではない）と、確率のキャリブレーション（全レース）。
学習の結果（fold ごとの予測）は data/v8/ に保存し、次のコマンドで使い回す（--refresh で作り直す）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass

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


@dataclass
class MarketTables:
    blended: pd.DataFrame        # fold 2 以降。prob（1段目）・市場確率・blend・odds_model
    coefs: pd.DataFrame          # 2段目の係数（fold ごと）
    odds_oof: experiment.OOF     # オッズも特徴量に入れたモデル
    stage1: pd.DataFrame         # 1段目の全 fold（fold 1 を含む。γ・δ の推定に使う）


def market_tables(args, ds) -> MarketTables:
    """2b の表。※ DataFrame を attrs に入れない（pd.concat が attrs を == で比べて落ちるため）。"""
    s1 = _oof(args, ds, _stage1(args))
    odds_variant = {"base": "base_odds", "levels": "levels_odds"}[_stage1(args)]
    od = _oof(args, ds, odds_variant)
    blended, coefs = market.benter_walk_forward(s1.table)
    od_t = od.table.set_index(["レースID", "血統登録番号"])["prob"]
    blended["odds_model"] = od_t.reindex(pd.MultiIndex.from_frame(
        blended[["レースID", "血統登録番号"]])).to_numpy()
    blended["odds_model"] = market.normalize(blended["odds_model"], blended["レースID"])
    return MarketTables(blended, coefs, od, s1.table)


def _cmd_market(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    mt = market_tables(args, ds)
    blended, coefs, od = mt.blended, mt.coefs, mt.odds_oof
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
def fold_discounts(t: pd.DataFrame, prob_col: str, stage1: pd.DataFrame | None = None) -> dict:
    """fold ごとに、それより前の fold の平場で γ・δ を推定する（walk-forward）。

    混ぜた確率（blend）は最初の fold には無い（2段目を推定できないため）。その fold の分は
    1段目の確率（prob）で代わりに推定に使う（γ・δ は「1着の確率から2・3着を出すときの割引」なので、
    確率の出どころが少し違っても大きくは変わらない）。
    レースごとの対数尤度は1回だけ計算し、fold ごとの推定はその行を足すだけにする（速さのため）。
    """
    full = t
    if stage1 is not None:
        early = stage1.loc[~stage1["fold"].isin(set(t["fold"]))].copy()
        if prob_col == "blend":
            early["blend"] = early["prob"]
        full = pd.concat([early, t], ignore_index=True)
    flat = full.loc[~full["重賞"]]
    rid, L2, L3 = market.place_loglik(flat, prob_col)
    fold_of = flat.drop_duplicates("レースID").set_index("レースID")["fold"]
    race_fold = fold_of.reindex(rid).to_numpy()
    folds = list(dict.fromkeys(full["fold"]))
    targets = set(t["fold"])
    out = {}
    for i, f in enumerate(folds):
        if f not in targets or i == 0:
            continue
        rows = np.isin(race_fold, folds[:i])
        g, d, n = market.discount_from_loglik(L2, L3, rows)
        if n:
            out[f] = (g, d, n)
    return out


def _cache_key(args, extra: str = "") -> str:
    """保存した fold の予測（data/v8/oof_*.json）が変われば、2c の保存も作り直す。"""
    parts = [extra, str(_stage1(args)), str(args.folds), str(args.rounds)]
    for v in (_stage1(args), {"base": "base_odds", "levels": "levels_odds"}[_stage1(args)]):
        meta = experiment.v8_dir() / f"oof_{v}.json"
        pkl = experiment.v8_dir() / f"oof_{v}.pkl"
        parts.append(meta.read_text(encoding="utf-8") if meta.exists() else "")
        parts.append(str(pkl.stat().st_mtime_ns) if pkl.exists() else "")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def trifecta_results(args, ds, mt: MarketTables, verbose: bool = True) -> dict:
    """2c・2d の計算。結果（γ・δ・表）は data/v8/trifecta_<1段目>.pkl に保存し、同じ条件なら読み直すだけ。"""
    path = experiment.v8_dir() / f"trifecta_{_stage1(args)}.pkl"
    key = _cache_key(args, "trifecta-v2")
    if path.exists() and not args.refresh:
        saved = pd.read_pickle(path)
        if saved.get("key") == key:
            if verbose:
                print(f"[v8] 保存済みの γ・δ と 2c・2d の結果を使います: {path.name}（作り直すには --refresh）")
            return saved
    blended = mt.blended
    if verbose:
        print("[v8] γ・δ を推定中（混ぜた確率・市場の確率）...", flush=True)
    disc_b = fold_discounts(blended, "blend", mt.stage1)
    disc_q = fold_discounts(blended, "市場確率", mt.stage1)
    folds = [f for f in dict.fromkeys(blended["fold"]) if f in disc_b and f in disc_q]
    rows, check_parts, gs, ds_ = [], [], [], []
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
        gs.append(g)
        ds_.append(d)
    g_avg = float(np.mean(gs)) if gs else 1.0
    d_avg = float(np.mean(ds_)) if ds_ else 1.0
    allt = blended.loc[blended["fold"].isin(folds)]
    cal = {}
    for name, gg, dd in [("Harville", 1.0, 1.0),
                         (f"Henery（γ={g_avg:.2f}, δ={d_avg:.2f}、fold 平均）", g_avg, d_avg)]:
        c, _, n = market.trifecta_calibration(allt, "blend", gg, dd)
        cal[name] = (c, n)
    check = pd.concat(check_parts, ignore_index=True) if check_parts else pd.DataFrame()
    out = {"key": key, "disc_b": disc_b, "disc_q": disc_q, "nll": pd.DataFrame(rows),
           "calibration": cal, "check": check}
    pd.to_pickle(out, path)
    return out


def _cmd_trifecta(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    mt = market_tables(args, ds)
    TRIALS.extend(["2c Harville", "2c Henery 型（γ・δ の割引）"])
    res = trifecta_results(args, ds, mt)
    clock.step("表を作成中 ...")
    print("\n=== 2c 実際の1〜3着の並びに付けた確率の −log の平均（平場。小さいほど良い） ===")
    _show(res["nll"])
    print("\n=== 2c 3連単の確率のキャリブレーション（全レース。重賞を含む・達成率ではない） ===")
    for name, (cal, n) in res["calibration"].items():
        print(f"--- {name}（{n:,} レース）---")
        _show(cal[["区間", "組の数", "予測の平均", "実際の的中率", "実際の的中"]], digits=6)
    check = res["check"]
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
    _save(args, res["nll"], "trifecta_nll")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 2e 買い目
# ---------------------------------------------------------------------------
# 「当たる確率」の出し方（配当はどれも市場の確率＝オッズから推定する）
MIX_WEIGHTS = (0.25, 0.5, 0.75)
PROB_SOURCES = ([("モデルだけ", "prob"), ("市場だけ", "市場確率"), ("混ぜたもの（2b の推定）", "blend")]
                + [(f"混ぜたもの（モデル {w:.2f}）", f"mix{int(round(w * 100))}") for w in MIX_WEIGHTS])
HEAT_SOURCES = ("blend", "prob")       # 熱さの一覧を出す確率（全部だと時間がかかるため2つ）
STRATEGIES = ["S1 上位6頭BOX", "S2 熱い組だけ 120点", f"S3 熱い組・フォーメーション{bets.MAX_CARDS_FORMATION}枚まで",
              "S4 券種と金額を混ぜる",
              f"S4c 券種混合・券種ごと{bets.MARKS_PER_CARD * bets.MAX_CARDS_PER_KIND}点まで",
              "S5 3連単だけ（金額は変えてよい）"]


def add_mix_columns(t: pd.DataFrame, weights=MIX_WEIGHTS) -> pd.DataFrame:
    """モデルの割合を固定して市場と混ぜた確率：P ∝ exp(w·log モデル + (1−w)·log 市場)（レース内で正規化）。

    2b の推定（1着の log-loss を最小にする割合）とは別に、割合を決め打ちした版。
    """
    t = t.copy()
    lp = np.log(t["prob"].astype("float64").clip(lower=1e-6))
    lq = np.log(t["市場確率"].astype("float64").clip(lower=1e-6))
    for w in weights:
        u = w * lp + (1 - w) * lq
        e = np.exp(u - u.groupby(t["レースID"]).transform("max"))
        t[f"mix{int(round(w * 100))}"] = e / e.groupby(t["レースID"]).transform("sum")
    return t


def evaluate_bets(blended: pd.DataFrame, payout: pd.DataFrame, discs: dict, disc_q: dict,
                  sources=PROB_SOURCES, sample: int = 0, seed: int = 0,
                  targets=(20_000, 30_000, 50_000, 100_000), heat_sources=HEAT_SOURCES,
                  strategies=None, verbose: bool = True, return_rows: bool = False):
    """平場のレースで、確率の出し方 × 買い方 を実際の払戻で評価する。戻り値: (成績, 熱さの一覧, 例)。

    discs: 確率の列 → {fold: (γ, δ, n)}。disc_q: 市場の確率の γ・δ（配当の推定に使う）。
    """
    flat = experiment.exploration_only(_flat(blended), "2e")
    folds_ok = set(disc_q)
    for _, col in sources:
        folds_ok &= set(discs[col])
    flat = flat.loc[flat["fold"].isin(folds_ok)]
    rids = flat["レースID"].unique()
    if sample and len(rids) > sample:
        rids = np.random.default_rng(seed).choice(rids, size=sample, replace=False)
        flat = flat.loc[flat["レースID"].isin(set(rids))]
    t, starts = market.race_slices(flat)
    pay = payout.set_index("レースID")
    cap = bets.MARKS_PER_CARD * bets.MAX_CARDS_PER_KIND
    rows, heat, examples = [], [], []
    for n_done, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        rid = t["レースID"].iloc[a]
        if rid not in pay.index or b - a < 4:
            continue
        r = t.iloc[a:b]
        q = r["市場確率"].to_numpy(dtype="float64")
        posts = pd.to_numeric(r["馬番"], errors="coerce").to_numpy()
        probs = {col: r[col].to_numpy(dtype="float64") for _, col in sources}
        if not (np.isfinite(q).all() and (q > 0).all() and np.isfinite(posts).all()
                and all(np.isfinite(v).all() for v in probs.values())):
            continue
        posts = posts.astype(int)
        wins = bets.winning_combos(pay.loc[rid])
        if not wins["3連単"]:
            continue
        f = r["fold"].iloc[0]
        gq, dq, _ = disc_q[f]
        Q = market.combo_probabilities(market.trifecta_tensor(q, gq, dq))
        for src_name, col in sources:
            p = probs[col]
            g, d, _ = discs[col][f]
            P = market.trifecta_tensor(p, g, d)
            cards = []

            def formation():
                plan, cs = bets.formation_cards(P, Q["3連単"], p, posts)
                cards.extend(cs)
                return plan
            makers = dict(zip(STRATEGIES, [
                lambda: bets.box(p, posts), lambda: bets.hot_top(P, Q["3連単"], posts), formation,
                lambda: bets.mixed(P, Q, p, posts), lambda: bets.mixed(P, Q, p, posts, max_per_kind=cap),
                lambda: bets.mixed(P, Q, p, posts, kinds=("3連単",))]))
            names = [n for n in STRATEGIES if strategies is None or n in strategies]
            plans = [makers[n]() for n in names]
            for name, plan in zip(names, plans):
                if np.isnan(plan.predicted):
                    plan.predicted = bets.predicted_success(plan, P, Q, posts, bets.TARGET)
                rows.append({"レースID": rid, "確率": src_name, "買い方": name, "投資": plan.invest,
                             "払戻": bets.realized(plan, wins), "予測の達成確率": plan.predicted,
                             "点数": len(plan.bets)})
            if col in heat_sources:
                for tg in targets:
                    plan = bets.mixed(P, Q, p, posts, target=tg)
                    heat.append({"確率": src_name, "基準": tg, "予測の達成確率": plan.predicted,
                                 "実際の払戻": bets.realized(plan, wins), "投資": plan.invest})
            if col == "blend" and len(examples) < 3 and strategies is None:
                examples.append({"レースID": rid, "S3 のカード": bets.card_text(cards),
                                 "S4 の買い目": " / ".join(f"{x.kind} {'-'.join(map(str, x.combo))} {x.stake:,}円"
                                                          for x in plans[3].bets)})
        if verbose and (n_done + 1) % 1000 == 0:
            print(f"    {n_done + 1:,} / {len(starts) - 1:,} レース", flush=True)
    res = pd.DataFrame(rows)
    summary = bets.summarize(res, keys=["確率", "買い方"]) if not res.empty else pd.DataFrame()
    h = pd.DataFrame(heat)
    heat_t = pd.DataFrame()
    if not h.empty:
        heat_rows = []
        for (src_name, tg), g in h.groupby(["確率", "基準"], sort=False):
            ach = int((g["実際の払戻"] >= tg).sum())
            lo, hi = market.wilson(ach, len(g))
            heat_rows.append({"確率": src_name, "基準": tg, "レース数": len(g),
                              "予測の達成確率（平均）": g["予測の達成確率"].mean(),
                              "実際の達成率": ach / len(g), "下限": lo, "上限": hi, "平均投資": g["投資"].mean()})
        heat_t = pd.DataFrame(heat_rows)
    if return_rows:
        return summary, heat_t, pd.DataFrame(examples), res
    return summary, heat_t, pd.DataFrame(examples)


def bets_discounts(args, ds, mt: MarketTables, res: dict, blended: pd.DataFrame,
                   stage1: pd.DataFrame) -> dict:
    """確率の出し方ごとの γ・δ（walk-forward）。2c の保存（blend・市場）は使い回し、残りはここで推定する（数秒〜数十秒）。"""
    discs = {"blend": res["disc_b"], "市場確率": res["disc_q"]}
    for _, col in PROB_SOURCES:
        if col not in discs:
            discs[col] = fold_discounts(blended, col, stage1)
    return discs


def _cmd_bets(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    mt = market_tables(args, ds)
    res = trifecta_results(args, ds, mt)          # blend・市場の γ・δ（保存があれば読み直すだけ）
    blended = add_mix_columns(mt.blended)
    stage1 = add_mix_columns(mt.stage1)
    clock.step("確率の出し方ごとに γ・δ を推定中 ...")
    discs = bets_discounts(args, ds, mt, res, blended, stage1)
    clock.step(f"買い目を計算中（平場、最大 {args.sample:,} レース × 確率 {len(PROB_SOURCES)} 通り × 買い方 {len(STRATEGIES)}）...")
    summary, heat, ex = evaluate_bets(blended, ds.payout_df, discs, res["disc_q"], sample=args.sample)
    clock.step("完了\n")
    for src_name, _ in PROB_SOURCES:
        for st in STRATEGIES:
            TRIALS.append(f"{st}｜{src_name}")
    c = mt.coefs
    if not c.empty:
        share = c["α（モデル）"] / (c["α（モデル）"] + c["β（市場）"])
        print("参考：2b の推定（fold ごと）  α（モデル） " + " / ".join(f"{x:.3f}" for x in c["α（モデル）"])
              + "  β（市場） " + " / ".join(f"{x:.3f}" for x in c["β（市場）"])
              + "  → モデルの割合 α/(α+β) " + " / ".join(f"{x:.2f}" for x in share))
    print("    固定の混ぜ方は P ∝ exp(w·log モデル + (1−w)·log 市場)、w = " + " / ".join(f"{w:.2f}" for w in MIX_WEIGHTS))
    print("\n=== 2e 平場：確率の出し方 × 買い方（予算 12,000円・基準 50,000円。配当はどれも市場の確率から推定、評価は実際の払戻） ===")
    _show(summary, pct=("率", "限", "確率（"), money=("投資",))
    if not summary.empty:
        print("\n=== 2e 達成率の一覧（行 = 買い方、列 = 確率の出し方。カッコ内は Wilson 95% 下限） ===")
        cell = summary.assign(表示=[f"{a:.2%}（{lo:.2%}）" for a, lo in zip(summary["達成率"], summary["下限"])])
        piv = cell.pivot(index="買い方", columns="確率", values="表示")
        piv = piv.reindex(index=STRATEGIES, columns=[n for n, _ in PROB_SOURCES])
        with pd.option_context("display.width", 300, "display.max_columns", 20):
            print(piv.to_string())
    print("\n=== 2e 熱さの一覧（券種混合 S4。狙う払戻ごとに、予測の達成確率と実際） ===")
    _show(heat, pct=("確率（", "達成率", "限"), money=("投資", "基準"))
    print("\n=== 例（混ぜたもの（2b）・会場で写す形） ===")
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
# 3b・3c 市場の確率に傾向で傾きをつける
# ---------------------------------------------------------------------------
S4C = STRATEGIES[4]


def rating_frame(ds, verbose: bool = True) -> pd.DataFrame:
    """v8c のレーティング2方式（Plackett-Luce・AtCoder 型。1986年から順番に。data/v8/ratings.pkl に保存して使い回す）。"""
    from . import preprocess, ratings
    rr = preprocess.basic_clean(ds.race_result)
    key = hashlib.sha1("|".join(["ratings-v2", str(len(rr)), str(rr["レースID"].nunique()),
                                 str(rr["レース日付"].min()), str(rr["レース日付"].max())]).encode()).hexdigest()
    path = experiment.v8_dir() / "ratings.pkl"
    if path.exists():
        saved = pd.read_pickle(path)
        if saved.get("key") == key:
            return saved["table"]
    if verbose:
        print(f"[v8] レーティングを計算中（{rr['レースID'].nunique():,} レースを日付順に。数分）...", flush=True)
    t0 = time.perf_counter()
    table = ratings.compute_ratings(rr, verbose=verbose)
    ac = ratings.compute_atcoder_ratings(rr, verbose=verbose)
    table = table.merge(ac, on=["レースID", "血統登録番号"], how="left")
    if verbose:
        print(f"[v8] レーティング完了: {time.perf_counter() - t0:.0f}秒", flush=True)
    pd.to_pickle({"key": key, "table": table}, path)
    return table


def tilt_tables(args, ds, base: experiment.OOF, verbose: bool = True):
    """市場だけ / 3b / 3c / 3b+3c / R / 3b+R / A / 3b+A の確率（fold 2 以降）と係数。data/v8/tilt_tables.pkl に保存して使い回す。"""
    from . import ratings, tilt
    path = experiment.v8_dir() / "tilt_tables.pkl"
    meta = experiment.v8_dir() / "oof_base.json"
    pkl = experiment.v8_dir() / "oof_base.pkl"
    key = hashlib.sha1("|".join(["tilt-v4", str(args.folds), str(args.rounds),
                                 meta.read_text(encoding="utf-8") if meta.exists() else "",
                                 str(pkl.stat().st_mtime_ns) if pkl.exists() else ""]).encode()).hexdigest()
    if path.exists() and not args.refresh:
        saved = pd.read_pickle(path)
        if saved.get("key") == key:
            if verbose:
                print(f"[v8] 保存済みの傾きの表を使います: {path.name}（作り直すには --refresh）")
            return saved["merged"], saved["coefs"], saved["full"]
    if verbose:
        print("[v8] 傾向の項を作成し、係数を推定中 ...", flush=True)
    terms = tilt.build_terms(ds)
    k = ["レースID", "血統登録番号"]
    rt = rating_frame(ds, verbose=verbose)[k + ["μ", "σ", "出走数", "AtCoder型_レーティング", "AtCoder型_出走数"]
                                           + ratings.RATING_TERMS + ratings.AC_TERMS]
    t = base.table.merge(terms, on=k, how="left").merge(rt, on=k, how="left")
    specs = [("3b", tilt.DAY_TERMS, "tilt3b"), ("3c", tilt.MEETING_TERMS, "tilt3c"),
             ("3b+3c", tilt.DAY_TERMS + tilt.MEETING_TERMS, "tilt3bc"),
             ("R", ratings.RATING_TERMS, "tiltR"), ("3b+R", tilt.DAY_TERMS + ratings.RATING_TERMS, "tilt3bR"),
             ("A", ratings.AC_TERMS, "tiltA"), ("3b+A", tilt.DAY_TERMS + ratings.AC_TERMS, "tilt3bA")]
    merged, coefs = None, {}
    for name, cols, out_col in specs:
        out, coefs[name] = tilt.tilt_walk_forward(t, cols, out_col)
        merged = out if merged is None else merged.merge(out[k + [out_col]], on=k, how="inner")
    pd.to_pickle({"key": key, "merged": merged, "coefs": coefs, "full": t}, path)
    return merged, coefs, t


TILT_SOURCES = [("市場だけ", "市場確率"), ("市場＋当日の傾向（3b）", "tilt3b"), ("市場＋開催の傾向（3c）", "tilt3c"),
                ("市場＋当日＋開催（3b+3c）", "tilt3bc")]


def paired_vs(rows: pd.DataFrame, base_name: str, by: str = "確率", target: float = bets.TARGET) -> pd.DataFrame:
    """同じレースで、base_name と他の条件の「熱い当たり（払戻 ≥ 基準）」を対応のある比較で。"""
    from . import trifecta as tri
    hot = rows.assign(熱い=rows["払戻"] >= target).pivot_table(index="レースID", columns=by, values="熱い",
                                                               aggfunc="first")
    out = []
    for name in hot.columns:
        if name == base_name:
            continue
        both = hot[[name, base_name]].dropna()
        t = tri.paired_proportion_test(both[name].astype(bool).to_numpy(), both[base_name].astype(bool).to_numpy())
        out.append({by: name, "レース数": t["n"], "こちらだけ熱い当たり": t["xだけ"],
                    f"{base_name}だけ熱い当たり": t["yだけ"], "両方": t["両方"],
                    "差": t["差"], "差の下限": t["差下限"], "差の上限": t["差上限"], "p値": t["p値"]})
    return pd.DataFrame(out)


def _cmd_tilt(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    merged, coefs, full = tilt_tables(args, ds, base)
    TRIALS.extend(["3b 市場＋当日の傾向（発走40分前まで）で傾き", "3c 市場＋開催の進み具合の傾向で傾き",
                   "3b+3c 両方の項を一緒に入れた傾き"])
    print("\n=== 3b・3c・3b+3c 係数（その fold より前の fold の平場で推定）。θ が 0 に近ければ市場は織り込み済み ===")
    for name, c in coefs.items():
        print(f"--- {name} ---")
        _show(c)
    from . import tilt as tilt_mod
    sd = _flat(full)[tilt_mod.DAY_TERMS + tilt_mod.MEETING_TERMS].astype("float64").std()
    print("    項の標準偏差（平場）: " + " / ".join(f"{k} {v:.4f}" for k, v in sd.items())
          + "。θ × 標準偏差 = 項が1標準偏差変わったときの log の変化（0.05 で約5%の確率の差）")
    flat = experiment.exploration_only(_flat(merged), "3b・3c")
    rows = [market.summary_row(flat, c, n) for n, c in TILT_SOURCES]
    print("\n=== 平場（fold 2 以降）：log-loss・AUC・キャリブレーション誤差 ===")
    _show(pd.DataFrame(rows))
    print("\n=== キャリブレーション（全レース。重賞を含む・達成率ではない） ===")
    for n, c in TILT_SOURCES[1:]:
        print(f"--- {n} ---")
        _show(market.calibration_table(merged, c), pct=("平均", "勝率"))
    clock.step("市場の γ・δ を推定中 ...")
    disc_q = fold_discounts(merged, "市場確率", base.table)
    discs = {c: disc_q for _, c in TILT_SOURCES}     # 傾きは市場の小さな補正なので、γ・δ は市場のものを使う
    clock.step(f"{S4C} を計算中（平場、最大 {args.sample:,} レース）...")
    summary, _, _, race_rows = evaluate_bets(merged, ds.payout_df, discs, disc_q, sources=TILT_SOURCES,
                                             sample=args.sample, heat_sources=(), strategies=[S4C],
                                             return_rows=True)
    print(f"\n=== 平場：{S4C}（予算 12,000円・基準 50,000円。配当はどれも市場の確率から推定、評価は実際の払戻） ===")
    _show(summary, pct=("率", "限", "確率（"), money=("投資",))
    paired = paired_vs(race_rows, "市場だけ")
    print(f"\n=== {S4C} の達成率の差（同じレースの対応のある比較。差 = こちら − 市場だけ） ===")
    _show(paired, pct=("差",))
    _save(args, pd.DataFrame(rows), "tilt_compare")
    _save(args, pd.concat([c.assign(案=n) for n, c in coefs.items()], ignore_index=True), "tilt_coefs")
    _save(args, summary, "tilt_s4c")
    _save(args, paired, "tilt_s4c_paired")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 金額の余裕（推定配当の下振れ対策）とスマッピーの点数の上限
# ---------------------------------------------------------------------------
MARGINS = (1.0, 1.2, 1.4)
POINT_LIMITS = (None, 10, 15, 20)


def _best_tilt_source(args) -> tuple[str, str]:
    """tilt の S4c の結果から、達成率が一番良い確率の出し方（--source で指定もできる）。"""
    if args.source:
        for n, c in TILT_SOURCES:
            if args.source in (n, c):
                return n, c
        raise SystemExit(f"--source は {[c for _, c in TILT_SOURCES]} のどれか")
    path = experiment.v8_dir() / "tilt_s4c.csv"
    if not path.exists():
        raise SystemExit("先に `py -m src.v8 tilt` を実行してください（どの確率が一番良いかを決めるため）")
    t = pd.read_csv(path, encoding="utf-8-sig")
    best = t.sort_values("達成率", ascending=False).iloc[0]["確率"]
    return best, dict(TILT_SOURCES)[best]


def _plan_races(table: pd.DataFrame, payout: pd.DataFrame, cols: list[str], disc_q: dict, sample: int, seed: int,
                what: str, verbose: bool):
    """平場（γ・δ のある fold）のレースを1つずつ：(レースID, fold, 馬番, 当たりの組, 市場の組の確率 Q, {列: (P, p)})。"""
    flat = experiment.exploration_only(_flat(table), what)
    flat = flat.loc[flat["fold"].isin(set(disc_q))]
    rids = flat["レースID"].unique()
    if sample and len(rids) > sample:
        rids = np.random.default_rng(seed).choice(rids, size=sample, replace=False)
        flat = flat.loc[flat["レースID"].isin(set(rids))]
    t, starts = market.race_slices(flat)
    pay = payout.set_index("レースID")
    t0, n_all = time.perf_counter(), len(starts) - 1
    for n_done, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        if verbose and n_done and n_done % 1000 == 0:
            left = (time.perf_counter() - t0) / n_done * (n_all - n_done)
            print(f"    {n_done:,} / {n_all:,} レース（残り約 {left / 60:.0f} 分）", flush=True)
        rid = t["レースID"].iloc[a]
        if rid not in pay.index or b - a < 4:
            continue
        r = t.iloc[a:b]
        q = r["市場確率"].to_numpy(dtype="float64")
        ps = {c: r[c].to_numpy(dtype="float64") for c in cols}
        posts = pd.to_numeric(r["馬番"], errors="coerce").to_numpy()
        if not (np.isfinite(q).all() and (q > 0).all() and np.isfinite(posts).all()
                and all(np.isfinite(p).all() for p in ps.values())):
            continue
        wins = bets.winning_combos(pay.loc[rid])
        if not wins["3連単"]:
            continue
        g, d, _ = disc_q[r["fold"].iloc[0]]
        Q = market.combo_probabilities(market.trifecta_tensor(q, g, d))
        yield rid, r["fold"].iloc[0], posts.astype(int), wins, Q, {c: (market.trifecta_tensor(p, g, d), p)
                                                                   for c, p in ps.items()}


def evaluate_stakes(table: pd.DataFrame, payout: pd.DataFrame, col: str, disc_q: dict, sample: int = 0,
                    seed: int = 0, margins=MARGINS, limits=POINT_LIMITS, verbose: bool = True) -> pd.DataFrame:
    """券種混合（S4）を、金額の余裕 × 点数の上限 の組み合わせで実際の払戻で評価する（1レース・1条件1行）。"""
    rows = []
    for rid, _, posts, wins, Q, ps in _plan_races(table, payout, [col], disc_q, sample, seed,
                                                  "金額の余裕・点数の上限", verbose):
        P, p = ps[col]
        for m in margins:
            for lim in limits:
                plan = bets.mixed(P, Q, p, posts, margin=m, max_points=lim)
                rows.append({"レースID": rid, "金額の余裕": m, "点数の上限": "なし" if lim is None else lim,
                             "条件": f"余裕×{m:.1f}・" + ("上限なし" if lim is None else f"{lim}点まで"),
                             "投資": plan.invest, "払戻": bets.realized(plan, wins),
                             "予測の達成確率": plan.predicted, "点数": len(plan.bets)})
    return pd.DataFrame(rows)


# 今の本命の買い方（v8b の stakes の結果から）：券種混合・金額の余裕 ×1.4・スマッピー 15点まで
FAVORITE_MARGIN = 1.4
FAVORITE_LIMIT = 15
FAVORITE_PLAN = f"券種混合・金額の余裕×{FAVORITE_MARGIN:.1f}・{FAVORITE_LIMIT}点まで"


def evaluate_plan(table: pd.DataFrame, payout: pd.DataFrame, sources: list[tuple[str, str]], disc_q: dict,
                  sample: int = 0, seed: int = 0, margin: float = FAVORITE_MARGIN, limit: int | None = FAVORITE_LIMIT,
                  verbose: bool = True) -> pd.DataFrame:
    """今の本命の買い方を、確率の出し方ごとに同じレースで実際の払戻で評価する（1レース・1確率1行）。"""
    rows = []
    for rid, fold, posts, wins, Q, ps in _plan_races(table, payout, [c for _, c in sources], disc_q, sample, seed,
                                                     FAVORITE_PLAN, verbose):
        for name, col in sources:
            P, p = ps[col]
            plan = bets.mixed(P, Q, p, posts, margin=margin, max_points=limit)
            rows.append({"レースID": rid, "fold": fold, "確率": name, "投資": plan.invest,
                         "払戻": bets.realized(plan, wins), "予測の達成確率": plan.predicted, "点数": len(plan.bets)})
    return pd.DataFrame(rows)


def plan_summary(rows: pd.DataFrame, by=("確率",), target: float = bets.TARGET) -> pd.DataFrame:
    out = []
    for key, g in rows.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        ach = int((g["払戻"] >= target).sum())
        lo, hi = market.wilson(ach, n)
        pred = float(g["予測の達成確率"].mean())
        out.append({**dict(zip(by, key)), "レース数": n, "的中率": float((g["払戻"] > 0).mean()),
                    "達成率": ach / n, "下限": lo, "上限": hi, "予測の達成確率（平均）": pred,
                    "予測 − 実際": pred - ach / n, "平均点数": float(g["点数"].mean()),
                    "平均投資": float(g["投資"].mean()),
                    "回収率": float(g["払戻"].sum() / g["投資"].sum()) if g["投資"].sum() else np.nan})
    return pd.DataFrame(out)


def stakes_summary(rows: pd.DataFrame, target: float = bets.TARGET) -> pd.DataFrame:
    out = []
    for (m, lim), g in rows.groupby(["金額の余裕", "点数の上限"], sort=False):
        n = len(g)
        ach = int((g["払戻"] >= target).sum())
        lo, hi = market.wilson(ach, n)
        pred = float(g["予測の達成確率"].mean())
        out.append({"金額の余裕": m, "点数の上限": lim, "レース数": n, "的中率": float((g["払戻"] > 0).mean()),
                    "達成率": ach / n, "下限": lo, "上限": hi, "予測の達成確率（平均）": pred,
                    "予測 − 実際": pred - ach / n, "平均点数": float(g["点数"].mean()),
                    "平均投資": float(g["投資"].mean()),
                    "回収率": float(g["払戻"].sum() / g["投資"].sum()) if g["投資"].sum() else np.nan})
    return pd.DataFrame(out)


def _cmd_stakes(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    merged, _, _ = tilt_tables(args, ds, base)
    name, col = _best_tilt_source(args)
    print(f"確率: {name}（{'指定' if args.source else 'tilt の S4c で達成率が一番良かったもの'}）")
    clock.step("市場の γ・δ を推定中 ...")
    disc_q = fold_discounts(merged, "市場確率", base.table)
    n_cond = len(MARGINS) * len(POINT_LIMITS)
    clock.step(f"券種混合を {n_cond} 条件で計算中（平場、最大 {args.sample:,} レース）...")
    rows = evaluate_stakes(merged, ds.payout_df, col, disc_q, sample=args.sample)
    clock.step("完了\n")
    for m in MARGINS:
        for lim in POINT_LIMITS:
            TRIALS.append(f"券種混合｜{name}｜金額の余裕×{m:.1f}｜" + ("点数の上限なし" if lim is None else f"{lim}点まで"))
    summ = stakes_summary(rows)
    print("=== 券種混合（S4）：金額の余裕 × 点数の上限（スマッピー）。予算 12,000円・達成の判定は 50,000円のまま ===")
    print("    金額の余裕 = 1点の金額を「推定配当 × 金額 ≥ 50,000円 × 余裕」にする。点数の上限 = 入力する点数の合計")
    _show(summ, pct=("率", "限", "確率（", "予測 −"), money=("投資",))
    base_cond = "余裕×1.0・上限なし"
    paired = paired_vs(rows, base_cond, by="条件")
    print(f"\n=== 対応のある比較（同じレース。差 = その条件 − {base_cond}） ===")
    _show(paired, pct=("差",))
    _save(args, summ, "stakes_summary")
    _save(args, paired, "stakes_paired")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# v8c レーティング（能力が時間とともに変わる）を市場の確率への傾きに
# ---------------------------------------------------------------------------
# R = Plackett-Luce 型（ベイズ）、A = AtCoder 型（パフォーマンスの重み付き平均）
RATING_ONLY = [("市場＋レーティング", "tiltR"), ("市場＋3b＋レーティング", "tilt3bR"),
               ("市場＋AtCoder型", "tiltA"), ("市場＋3b＋AtCoder型", "tilt3bA")]
RATING_SOURCES = [("市場だけ", "市場確率"), ("市場＋当日の傾向（3b）", "tilt3b")] + RATING_ONLY
FINAL_SOURCES = TILT_SOURCES + RATING_ONLY


def _coef_with_sd(coef: pd.DataFrame, sd: pd.Series) -> pd.DataFrame:
    """係数（fold ごと）に、θ × 標準偏差（項が1標準偏差変わったときの log の変化）の列を足す。"""
    c = coef.copy()
    for term, s in sd.items():
        col = f"θ（{term}）"
        if col in c.columns:
            c[f"θ×SD（{term}）"] = c[col] * s
    return c


def _paired_report(rows: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    out = []
    for b in bases:
        p = paired_vs(rows, b)
        print(f"\n=== {FAVORITE_PLAN} の達成率の差（同じレースの対応のある比較。差 = こちら − {b}） ===")
        _show(p, pct=("差",))
        out.append(p.assign(比べる相手=b))
    return pd.concat(out, ignore_index=True)


def _cmd_rating(args) -> int:
    from . import ratings, tilt
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    merged, coefs, full = tilt_tables(args, ds, base)
    TRIALS.extend(["A 市場＋AtCoder型（相対・伸び・出走の少なさ・ばらつき）で傾き",
                   "3b+A 市場＋当日の傾向＋AtCoder型で傾き"])
    print("[記録] R・3b+R（Plackett-Luce 型）は v8c で数えた2個。今回の新しい案は AtCoder 型の2個")
    sd = _flat(full)[tilt.DAY_TERMS + ratings.RATING_TERMS + ratings.AC_TERMS].astype("float64").std()
    print("\n=== 係数（その fold より前の fold の平場で推定）。θ×SD = 項が1標準偏差変わったときの log の変化"
          "（0.05 で約5%の確率の差）。θ が 0 に近ければ市場は織り込み済み ===")
    for name in ("R", "3b+R", "A", "3b+A"):
        print(f"--- {name} ---")
        _show(_coef_with_sd(coefs[name], sd))
    print("    項の標準偏差（平場）: " + " / ".join(f"{k} {v:.4f}" for k, v in sd.items()))
    flat = experiment.exploration_only(_flat(merged), "レーティング")
    rows = [market.summary_row(flat, c, n) for n, c in RATING_SOURCES]
    print("\n=== 平場（fold 2 以降）：log-loss・AUC・キャリブレーション誤差 ===")
    _show(pd.DataFrame(rows))
    print("\n=== 参考：1着の予測の AUC（平場・fold 2 以降。レーティングだけ＝傾きにせず、そのまま順位づけ） ===")
    ref = [{"予測": "今のモデル（base）", "AUC": market.auc(flat["win"], flat["prob"])},
           {"予測": "市場（単勝オッズ）", "AUC": market.auc(flat["win"], flat["市場確率"])},
           {"予測": "レーティングだけ（μ）", "AUC": market.auc(flat["win"], flat["μ"])},
           {"予測": "レーティングだけ（μ − 2σ。不確かな馬を割り引く）",
            "AUC": market.auc(flat["win"], flat["μ"] - 2 * flat["σ"])},
           {"予測": "AtCoder型レーティングだけ（出走が少ない馬の割り引きあり）",
            "AUC": market.auc(flat["win"], flat["AtCoder型_レーティング"])}]
    _show(pd.DataFrame(ref))
    print("\n=== キャリブレーション（全レース。重賞を含む・達成率ではない） ===")
    for n, c in RATING_ONLY:
        print(f"--- {n} ---")
        _show(market.calibration_table(merged, c), pct=("平均", "勝率"))
    clock.step("市場の γ・δ を推定中 ...")
    disc_q = fold_discounts(merged, "市場確率", base.table)
    clock.step(f"{FAVORITE_PLAN} を計算中（平場、最大 {args.sample:,} レース × {len(RATING_SOURCES)} 通り）...")
    plan_rows = evaluate_plan(merged, ds.payout_df, RATING_SOURCES, disc_q, sample=args.sample)
    clock.step("完了\n")
    summ = plan_summary(plan_rows)
    print(f"=== 平場：{FAVORITE_PLAN}（予算 12,000円・基準 50,000円。配当は市場の確率から推定、評価は実際の払戻） ===")
    _show(summ, pct=("率", "限", "確率（", "予測 −"), money=("投資",))
    paired = _paired_report(plan_rows, ["市場だけ", "市場＋当日の傾向（3b）"])
    _save(args, pd.DataFrame(rows), "rating_compare")
    _save(args, pd.DataFrame(ref), "rating_auc")
    _save(args, pd.concat([_coef_with_sd(coefs[n], sd).assign(案=n) for n in ("R", "3b+R", "A", "3b+A")],
                          ignore_index=True),
          "rating_coefs")
    _save(args, summ, "rating_plan")
    _save(args, paired, "rating_plan_paired")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# v8d 「市場＋3b＋伸び」：AtCoder 型の伸びだけを 3b に足す（採用のルールは結果を見る前に決めたもの）
# ---------------------------------------------------------------------------
GROWTH_TERM = "AtCoder型_伸び"
GROWTH_SOURCES = [("市場だけ", "市場確率"), ("市場＋当日の傾向（3b）", "tilt3b"), ("市場＋3b＋伸び", "tilt3bG")]
GROWTH_RULE = ("「市場＋3b＋伸び」−「市場＋3b」の達成率の差の 95% 区間の下限が 0 を超えたら、重賞のテストの F の確率を"
               "「市場＋3b＋伸び」に差し替える。超えなければ F は「市場＋当日の傾向（3b）」のまま")


def growth_judgement(paired: pd.DataFrame) -> tuple[bool, dict]:
    """採用のルールでの判定（True = 差し替える）。paired は paired_vs(..., '市場＋当日の傾向（3b）') の結果。"""
    row = paired.loc[paired["確率"] == GROWTH_SOURCES[2][0]].iloc[0].to_dict()
    return bool(row["差の下限"] > 0), row


def _cmd_growth_check(args) -> int:
    from . import tilt
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    merged, _, full = tilt_tables(args, ds, base)
    terms = tilt.DAY_TERMS + [GROWTH_TERM]
    clock.step("「市場＋3b＋伸び」の係数を推定中（その fold より前の fold の平場で）...")
    out, coef = tilt.tilt_walk_forward(full, terms, "tilt3bG")
    k = ["レースID", "血統登録番号"]
    merged = merged.merge(out[k + ["tilt3bG"]], on=k, how="inner")
    TRIALS.append("3b+伸び 市場＋当日の傾向＋AtCoder 型の伸びだけで傾き")
    clock.step("市場の γ・δ を推定中 ...")
    disc_q = fold_discounts(merged, "市場確率", base.table)
    n = merged.loc[~merged["重賞"] & merged["fold"].isin(set(disc_q)), "レースID"].nunique()
    n = min(n, args.sample) if args.sample else n
    clock.step(f"{FAVORITE_PLAN} を計算中（平場 {n:,} レース × {len(GROWTH_SOURCES)} 通り）...")
    rows = evaluate_plan(merged, ds.payout_df, GROWTH_SOURCES, disc_q, sample=args.sample)
    clock.step("完了\n")
    vs3b = paired_vs(rows, GROWTH_SOURCES[1][0])
    adopt, row = growth_judgement(vs3b)
    verdict = "差し替える（F の確率を「市場＋3b＋伸び」に）" if adopt else "差し替えない（F は「市場＋当日の傾向（3b）」のまま）"
    print("=" * 100)
    print(f"■ 判定：{verdict}")
    print(f"  差（市場＋3b＋伸び − 市場＋3b）= {row['差']:+.2%}  95% 区間 [{row['差の下限']:+.2%}, {row['差の上限']:+.2%}]"
          f"  p値 {row['p値']:.4f}  レース数 {int(row['レース数']):,}")
    print(f"  ルール（結果を見る前に決めたもの）：{GROWTH_RULE}")
    print("=" * 100)
    print(f"\n=== {FAVORITE_PLAN} の達成率の差（同じレースの対応のある比較。差 = こちら − 市場＋当日の傾向（3b）） ===")
    _show(vs3b, pct=("差",))
    vsm = paired_vs(rows, "市場だけ")
    print(f"\n=== 参考：差 = こちら − 市場だけ ===")
    _show(vsm, pct=("差",))
    summ = plan_summary(rows)
    print(f"\n=== 平場（fold 2 以降）：{FAVORITE_PLAN}（予算 12,000円・基準 50,000円） ===")
    _show(summ, pct=("率", "限", "確率（", "予測 −"), money=("投資",))
    by_fold = plan_summary(rows, by=("fold", "確率"))
    print("\n=== fold ごとの達成率 ===")
    _show(by_fold[["fold", "確率", "レース数", "達成率", "下限", "上限", "予測の達成確率（平均）"]],
          pct=("率", "限", "確率（"))
    flat = experiment.exploration_only(_flat(merged), "3b+伸び")
    metrics = pd.DataFrame([market.summary_row(flat, c, nm) for nm, c in GROWTH_SOURCES])
    print("\n=== 平場（fold 2 以降）：log-loss・AUC・キャリブレーション誤差 ===")
    _show(metrics)
    print("\n=== キャリブレーション（平場） ===")
    for nm, c in GROWTH_SOURCES[1:]:
        print(f"--- {nm} ---")
        _show(market.calibration_table(flat, c), pct=("平均", "勝率"))
    sd = _flat(full)[terms].astype("float64").std()
    print("\n=== 係数（fold ごと。その fold より前の fold の平場で推定）と θ×SD ===")
    _show(_coef_with_sd(coef, sd))
    print("    項の標準偏差（平場）: " + " / ".join(f"{a} {b:.4f}" for a, b in sd.items()))
    judge = pd.DataFrame([{"判定": verdict, "差し替える": adopt, **{c: row[c] for c in
                                                                ("レース数", "差", "差の下限", "差の上限", "p値")},
                           "ルール": GROWTH_RULE, "サンプル": args.sample}])
    _save(args, judge, "growth_judgement")
    _save(args, pd.concat([vs3b.assign(比べる相手="市場＋当日の傾向（3b）"), vsm.assign(比べる相手="市場だけ")],
                          ignore_index=True), "growth_paired")
    _save(args, summ, "growth_summary")
    _save(args, by_fold, "growth_by_fold")
    _save(args, metrics, "growth_metrics")
    _save(args, _coef_with_sd(coef, sd), "growth_coefs")
    _trials_note()
    print(f"\n■ 判定（再掲）：{verdict}")
    if args.sample:
        print(f"  ※ --sample {args.sample} で一部のレースだけ。判定には --sample 0（既定）の結果を使ってください")
    return 0


def _cmd_final_compare(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    merged, _, _ = tilt_tables(args, ds, base)
    sources = FINAL_SOURCES if args.with_rating else TILT_SOURCES
    clock.step("市場の γ・δ を推定中 ...")
    disc_q = fold_discounts(merged, "市場確率", base.table)
    n = merged.loc[~merged["重賞"] & merged["fold"].isin(set(disc_q)), "レースID"].nunique()
    n = min(n, args.sample) if args.sample else n
    clock.step(f"{FAVORITE_PLAN} を計算中（平場 {n:,} レース × {len(sources)} 通り）...")
    rows = evaluate_plan(merged, ds.payout_df, sources, disc_q, sample=args.sample)
    clock.step("完了\n")
    print("[記録] 新しい案ではなく、これまでに試した案を全レースで見直すだけ（試した案の数は増やさない）")
    summ = plan_summary(rows)
    print(f"\n=== 平場の全レース：{FAVORITE_PLAN}（予算 12,000円・基準 50,000円） ===")
    _show(summ, pct=("率", "限", "確率（", "予測 −"), money=("投資",))
    by_fold = plan_summary(rows, by=("fold", "確率"))
    print("\n=== fold ごと（年によって順位が入れ替わらないか） ===")
    _show(by_fold[["fold", "確率", "レース数", "達成率", "下限", "上限", "予測の達成確率（平均）"]],
          pct=("率", "限", "確率（"))
    bases = ["市場だけ", "市場＋当日の傾向（3b）"]
    paired = _paired_report(rows, bases)
    _save(args, summ, "final_compare")
    _save(args, by_fold, "final_compare_by_fold")
    _save(args, paired, "final_compare_paired")
    return 0


# ---------------------------------------------------------------------------
# v8d 本物の確定オッズで買い目を選ぶ（平場サンプル）
# ---------------------------------------------------------------------------
REAL_MARGINS = (1.0, 1.1, 1.2, 1.4)
ESTIMATED_COND = f"推定の配当・余裕×{FAVORITE_MARGIN:.1f}（今の本命）"
PRED_BINS = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0001]


def _real_cond(m: float) -> str:
    return f"本物のオッズ・余裕×{m:.1f}"


def evaluate_real_odds(table: pd.DataFrame, payout: pd.DataFrame, col: str, disc_q: dict, real: dict,
                       margins=REAL_MARGINS, limit: int | None = FAVORITE_LIMIT, verbose: bool = True):
    """同じレースで、推定の配当（余裕×1.4）と本物の確定オッズ（余裕 margins）で券種混合を選び、実際の払戻で評価する。

    戻り値: (1レース・1条件1行, 推定÷本物 の集計, 推定の配当で選んだ組の本物のオッズでの確認, 確定オッズと払戻の突き合わせ)
    """
    from . import realodds
    table = table.loc[table["レースID"].astype(str).isin(set(real))]
    rows, chosen = [], []
    stats = realodds.RatioStats()
    check = {"的中組": 0, "一致": 0}
    for rid, fold, posts, wins, Q, ps in _plan_races(table, payout, [col], disc_q, 0, 0, "本物のオッズ", verbose):
        T = realodds.payout_tables(real[str(rid)], posts)
        if T is None:
            continue
        P, p = ps[col]
        stats.add(realodds.ratio_arrays(Q, T, P))
        index = {int(x): i for i, x in enumerate(posts)}
        for kind, d in wins.items():                         # 確定オッズ × 100 と実際の払戻（同着でないとき一致するはず）
            if len(d) != 1:
                continue
            (combo, pay), = d.items()
            if all(x in index for x in combo):
                ix = [index[x] for x in combo]
                v = T[kind][tuple(ix)]
                if np.isfinite(v):
                    check["的中組"] += 1
                    check["一致"] += int(abs(v - pay) < 10)     # オッズは 0.1倍刻み＝払戻 10円刻み
        plans = [(ESTIMATED_COND, bets.mixed(P, Q, p, posts, margin=FAVORITE_MARGIN, max_points=limit))]
        plans += [(_real_cond(m), bets.mixed(P, Q, p, posts, margin=m, max_points=limit, odds=T)) for m in margins]
        for name, plan in plans:
            rows.append({"レースID": rid, "fold": fold, "条件": name, "投資": plan.invest,
                         "払戻": bets.realized(plan, wins), "予測の達成確率": plan.predicted, "点数": len(plan.bets)})
        for b in plans[0][1].bets:                            # 推定の配当で選んだ組は、本物のオッズだといくらだったか
            ix = [index[x] for x in b.combo]
            q = Q[b.kind][tuple(ix)] if b.kind in ("3連単", "3連複") else Q[b.kind][ix[0], ix[1]]
            r = T[b.kind][tuple(ix)] if b.kind in ("3連単", "3連複") else T[b.kind][ix[0], ix[1]]
            chosen.append({"券種": b.kind, "推定": market.estimated_payout(float(q), b.kind), "本物": float(r),
                           "金額": b.stake})
    return pd.DataFrame(rows), stats.summary(), pd.DataFrame(chosen), check


def chosen_summary(chosen: pd.DataFrame, target: float = bets.TARGET) -> pd.DataFrame:
    """推定の配当で選んだ組：本物のオッズだと、当たっても基準に届かなかった割合など（券種ごと）。"""
    out = []
    c = chosen.loc[np.isfinite(chosen["本物"])]
    for kind, g in [("全部", c)] + list(c.groupby("券種", sort=False)):
        r = g["推定"] / g["本物"]
        out.append({"券種": kind, "点数": len(g), "推定÷本物（中央値）": float(r.median()),
                    "本物だと基準に届かない割合": float((g["本物"] * g["金額"] / 100 < target).mean()),
                    "本物だと基準×0.8に届かない割合": float((g["本物"] * g["金額"] / 100 < 0.8 * target).mean())})
    return pd.DataFrame(out)


def prediction_calibration(rows: pd.DataFrame, target: float = bets.TARGET) -> pd.DataFrame:
    """予測の達成確率の区間ごとに、予測の平均と実際の達成率（条件ごと）。"""
    t = rows.assign(区間=pd.cut(rows["予測の達成確率"], PRED_BINS, right=False).astype(str),
                    達成=rows["払戻"] >= target)
    out = t.groupby(["条件", "区間"], sort=False).agg(レース数=("達成", "size"), 予測の平均=("予測の達成確率", "mean"),
                                                   実際の達成率=("達成", "mean")).reset_index()
    return out


def _cmd_real_odds(args) -> int:
    from . import realodds
    clock = _Clock()
    path = experiment.v8_dir() / "odds_targets.csv"
    if not path.exists():
        raise SystemExit("先に `py -m src.v8 odds-targets` を実行してください（取り直したレースの一覧）")
    targets = pd.read_csv(path, dtype={"レースID": str, "fold": str}, encoding="utf-8-sig")
    flat_ids = targets.loc[targets["区分"] == "平場サンプル", "レースID"].tolist()   # 重賞は読まない
    if args.sample and len(flat_ids) > args.sample:
        flat_ids = list(np.random.default_rng(0).choice(flat_ids, size=args.sample, replace=False))
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    merged, _, _ = tilt_tables(args, ds, base)
    name, col = _best_tilt_source(args) if args.source else ("市場＋当日の傾向（3b）", "tilt3b")
    clock.step(f"確定オッズを読み込み中（平場サンプル {len(flat_ids):,} レース。馬連・馬単・3連複・3連単）...")
    real = realodds.load_final_odds(flat_ids, base=args.data)
    print(f"         確定オッズのあるレース: {len(real):,}（4券種そろい: "
          f"{sum(all(k in v for k in realodds.KIND_TYPES) for v in real.values()):,}）", flush=True)
    if not real:
        print("確定オッズがありません（`py -m src.jvlink fetch-odds --targets ...` で取り直す）")
        return 1
    clock.step("市場の γ・δ を推定中 ...")
    disc_q = fold_discounts(merged, "市場確率", base.table)
    n_cond = 1 + len(REAL_MARGINS)
    clock.step(f"券種混合（{name}・{FAVORITE_LIMIT}点まで）を {n_cond} 条件で計算中 ...")
    rows, ratio, chosen, check = evaluate_real_odds(merged, ds.payout_df, col, disc_q, real)
    clock.step("完了\n")
    TRIALS.extend([f"券種混合｜{name}｜本物の確定オッズで金額を決める｜余裕×{m:.1f}｜{FAVORITE_LIMIT}点まで"
                   for m in REAL_MARGINS])
    print(f"確率: {name}。確定オッズ×100 と実際の払戻の一致: {check['一致']:,} / {check['的中組']:,} 組"
          "（差が10円未満。同着のレースは除く。ほぼ全部一致するはず）")
    summ = plan_summary(rows, by=("条件",))
    print(f"\n=== 1・3 平場サンプル：券種混合 × {name} × {FAVORITE_LIMIT}点まで。金額を推定の配当で決める / 本物の確定オッズで決める ===")
    print("    本物のオッズで決めると、当たれば基準に届くことがほぼ確実（確定オッズ＝払戻のため）。"
          "残るずれは、勝ち馬・並びの予測の確率のずれ")
    _show(summ, pct=("率", "限", "確率（", "予測 −"), money=("投資",))
    paired = paired_vs(rows, ESTIMATED_COND, by="条件")
    print(f"\n=== 2 対応のある比較（同じレース。差 = その条件 − {ESTIMATED_COND}） ===")
    _show(paired, pct=("差",))
    cal = prediction_calibration(rows)
    print("\n=== 3 予測の達成確率の区間ごとの、予測と実際 ===")
    _show(cal, pct=("平均", "達成率"))
    print("\n=== 4 推定した配当 ÷ 本物の確定オッズ（全部の組。1 より大きい = 推定が高すぎる）===")
    print("    確率で重みづけ = 予測の確率（P）が高い組ほど重く（買い目になりやすい組でのずれ）")
    _show(ratio, pct=("以内", "割以上"))
    cs = chosen_summary(chosen)
    print(f"\n=== 4 補足：{ESTIMATED_COND} で選んだ組を本物の確定オッズで見ると ===")
    _show(cs, pct=("割合",))
    _save(args, summ, "real_odds_summary")
    _save(args, paired, "real_odds_paired")
    _save(args, cal, "real_odds_calibration")
    _save(args, ratio, "real_odds_ratio")
    _save(args, cs, "real_odds_chosen")
    _trials_note()
    return 0


# ---------------------------------------------------------------------------
# 本物のオッズ：取り直すレースの一覧と、速報オッズと確定オッズのずれ
# ---------------------------------------------------------------------------
def odds_targets(table: pd.DataFrame, n_flat: int = 5_000, seed: int = 0,
                 flat_folds=None) -> pd.DataFrame:
    """過去の確定オッズを取り直すレース：検証期間の重賞すべて ＋ 平場の検証用サンプル。

    平場は 2b 以降で比べている fold（2 以降）から n_flat レース（乱数の種を固定。何度作っても同じ一覧）。
    """
    races = table.drop_duplicates("レースID")[["レースID", "レース日付", "fold", "重賞"]]
    graded = races.loc[races["重賞"]].assign(区分="重賞")
    folds = list(dict.fromkeys(table["fold"]))
    flat_folds = flat_folds or folds[1:]
    flat = races.loc[~races["重賞"] & races["fold"].isin(flat_folds)].sort_values("レースID")
    if n_flat and len(flat) > n_flat:
        pick = np.sort(np.random.default_rng(seed).choice(len(flat), size=n_flat, replace=False))
        flat = flat.iloc[pick]
    out = pd.concat([graded, flat.assign(区分="平場サンプル")], ignore_index=True)
    out["日付"] = pd.to_datetime(out["レース日付"]).dt.strftime("%Y-%m-%d")
    return out[["レースID", "日付", "区分", "fold"]].sort_values("レースID").reset_index(drop=True)


def _cmd_odds_targets(args) -> int:
    clock = _Clock()
    ds = _load(args, clock)
    base = _oof(args, ds, "base")
    t = odds_targets(base.table, n_flat=args.n_flat)
    path = experiment.v8_dir() / "odds_targets.csv"
    t.to_csv(path, index=False, encoding="utf-8-sig")
    print(t.groupby("区分").agg(レース数=("レースID", "size"), 最初=("日付", "min"), 最後=("日付", "max")).to_string())
    print(f"\n保存: {path}")
    print("取り直し: py -m src.jvlink fetch-odds --from 20110101000000 --option 4 --targets " + str(path))
    return 0


def odds_drift(snapshots: pd.DataFrame, final: pd.DataFrame, post: pd.Series) -> pd.DataFrame:
    """速報オッズの各時点と確定オッズを、同じ組で比べる（1レース・1券種・1時点1行）。

    post: レースID → 発走日時。時点は「発表月日時分」（中間オッズ）か、無ければ取得時刻。
    """
    k = ["レースID", "券種", "組番"]
    m = snapshots.merge(final[k + ["オッズ"]].rename(columns={"オッズ": "確定"}), on=k, how="inner")
    m = m.loc[m["オッズ"].notna() & m["確定"].notna()]
    if m.empty:
        return pd.DataFrame()
    year = m["レースID"].str[:4]
    ann = pd.to_datetime(year + m["発表"].astype(str), format="%Y%m%d%H%M", errors="coerce")
    got = pd.to_datetime(m["取得時刻"], errors="coerce")
    m["時点"] = ann.fillna(got)
    m["発走まで（分）"] = (m["レースID"].map(post) - m["時点"]).dt.total_seconds() / 60
    m["log比"] = np.log(m["オッズ"] / m["確定"])
    rows = []
    for (rid, kind, when), g in m.groupby(["レースID", "券種", "時点"], sort=False):
        r = g["log比"]
        rows.append({"レースID": rid, "券種": kind, "時点": when, "発走まで（分）": g["発走まで（分）"].iloc[0],
                     "データ区分": g["データ区分"].iloc[0], "組の数": len(g),
                     "|log比|の中央値": float(r.abs().median()), "±10%以内": float((r.abs() <= np.log(1.1)).mean()),
                     "±25%以内": float((r.abs() <= np.log(1.25)).mean()),
                     "速報÷確定の中央値": float(np.exp(r.median()))})
    return pd.DataFrame(rows).sort_values(["レースID", "券種", "時点"]).reset_index(drop=True)


def _cmd_odds_drift(args) -> int:
    from . import odds, results
    base = odds.odds_dir(args.data)
    d = base / "rt" / args.date
    snaps = pd.concat([odds.load_odds(rt, directory=d, final_only=False, latest=False) for rt in odds.ODDS_TYPES],
                      ignore_index=True)
    if snaps.empty:
        print(f"速報オッズがありません: {d}（`py -m src.jvlink rt-odds --date {args.date}` で取る）")
        return 1
    rids = set(snaps["レースID"])
    final = pd.concat([odds.load_odds(rt, base=args.data, race_ids=rids) for rt in odds.ODDS_TYPES],
                      ignore_index=True)
    if final.empty:
        print("確定オッズがまだありません（月曜の fetch のあと `py -m src.jvlink fetch-odds` を実行）")
        return 1
    raw = jvmap.load_raw(args.data, record_types=("RA",))
    post = results.race_schedule(raw).drop_duplicates("レースID").set_index("レースID")["発走日時"]
    t = odds_drift(snaps, final, post)
    print(f"=== 速報オッズと確定オッズのずれ（{args.date}）。log比 = log(速報 ÷ 確定) ===")
    _show(t, pct=("以内",))
    if not t.empty:
        t["発走まで"] = pd.cut(t["発走まで（分）"], [-1e9, 5, 20, 60, 1e9],
                            labels=["5分前より後", "5〜20分前", "20〜60分前", "60分より前"])
        agg = t.groupby(["券種", "発走まで"], observed=True)[["|log比|の中央値", "±10%以内", "±25%以内"]].mean()
        print("\n--- 券種 × 発走までの時間（平均） ---")
        with pd.option_context("display.width", 200):
            print(agg.to_string())
    _save(args, t.drop(columns=["発走まで"], errors="ignore"), f"odds_drift_{args.date}")
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
            q.add_argument("--sample", type=int, default=5_000,
                           help="平場から何レースを使うか（0 で全部。確率6通り×買い方6通りを計算するので重い）")
        q.set_defaults(func=func)
    sub.add_parser("trackbias", help="3 当日の馬場の傾向（平場の後半）").set_defaults(func=_cmd_trackbias)
    q = sub.add_parser("tilt", help="3b・3c 市場の確率に当日・開催の傾向で傾きをつける（平場）")
    q.add_argument("--sample", type=int, default=5_000, help="S4c を計算する平場のレース数（0 で全部）")
    q.set_defaults(func=_cmd_tilt)
    q = sub.add_parser("odds-targets", help="過去の確定オッズを取り直すレースの一覧（重賞＋平場の検証用）")
    q.add_argument("--n-flat", type=int, default=5_000, help="平場のサンプルのレース数")
    q.set_defaults(func=_cmd_odds_targets)
    q = sub.add_parser("odds-drift", help="速報オッズと確定オッズのずれ（その日の rt-odds と fetch-odds の後）")
    q.add_argument("--date", required=True, help="YYYYMMDD")
    q.set_defaults(func=_cmd_odds_drift)
    q = sub.add_parser("stakes", help="券種混合の金額の余裕（×1.0/1.2/1.4）とスマッピーの点数の上限（10/15/20）")
    q.add_argument("--sample", type=int, default=5_000, help="平場のレース数（0 で全部）")
    q.add_argument("--source", default=None, help="確率（既定は tilt の S4c で一番良かったもの。例 tilt3bc）")
    q.set_defaults(func=_cmd_stakes)
    q = sub.add_parser("rating", help="v8c レーティングを市場の確率への傾きに（平場。今の本命の買い方で比べる）")
    q.add_argument("--sample", type=int, default=5_000, help="達成率を計算する平場のレース数（0 で全部）")
    q.set_defaults(func=_cmd_rating)
    q = sub.add_parser("final-compare", help="市場だけ / 3b / 3c / 3b+3c / R / 3b+R / A / 3b+A を今の本命の買い方で（平場の全レース）")
    q.add_argument("--sample", type=int, default=0, help="平場のレース数（既定 0 = 全部、約4万2千。数十分）")
    q.add_argument("--no-rating", dest="with_rating", action="store_false",
                   help="レーティングの版（R・3b+R・A・3b+A）を入れない（4通りになり、時間は約半分）")
    q.set_defaults(func=_cmd_final_compare)
    q = sub.add_parser("growth-check", help="v8d 市場＋3b＋伸び（AtCoder 型）と 市場＋3b を平場の全レースで（採用の判定つき）")
    q.add_argument("--sample", type=int, default=0, help="平場のレース数（既定 0 = 全部、約4万2千）")
    q.set_defaults(func=_cmd_growth_check)
    q = sub.add_parser("real-odds", help="v8d 本物の確定オッズで券種混合の金額を決める（平場サンプル）")
    q.add_argument("--sample", type=int, default=0, help="平場サンプルのうち何レース（既定 0 = 全部の約5,000）")
    q.add_argument("--source", default=None, help="確率（既定 tilt3b ＝ 市場＋当日の傾向）")
    q.set_defaults(func=_cmd_real_odds)
    args = parser.parse_args(argv)
    TRIALS.clear()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
