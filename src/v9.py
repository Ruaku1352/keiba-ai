"""v9：重賞で1回だけ行う確認のテスト（内容と判定のルールは結果を見る前に固定したもの）。

    py -m src.v9 confirm        F・F2・B を重賞（fold 2〜5、2014年〜2026-10-25）で比べ、判定を出す

■ 比べる3つ（これ以外の案の重賞での成績は計算しない）
  F  : 券種混合（3連単・3連複・馬単・馬連）・金額の余裕 ×1.4・15点まで・予算 12,000円。
       確率 = 市場（単勝の確定オッズ）＋当日の傾向（3b。同じ日・同じ競馬場・同じ芝ダで発走40分以上前に発走したレース）。
       係数はその fold より前の fold の平場で推定（v8 の tilt と同じ）。3連単などは Henery 型（γ・δ は市場の確率について前の fold の平場で）
  F2 : 買い方・金額・点数・予算は F と同じ。確率 ∝ exp(0.5·log モデル + 0.5·log 市場)（レース内で正規化）。
       モデルは B と同じ今のモデル。3b は入れない。γ・δ はこの確率について前の fold の平場で推定
  F・F2 とも、配当の推定は市場の確率と払戻率から（v8 と同じ）
  B  : 今のモデルの上位6頭の3連単BOX（120点・12,000円）

■ 判定（主指標）
  1レースごとに「払戻の合計 ≥ 50,000円」（熱い当たり）。同じレース同士の対応のある比較で F − B と F2 − B の差の
  97.5% 区間（候補が2つなので 95% より厳しく。方法は v8 と同じ Newcombe 法10・正確な McNemar）。
  下限が 0 を超えた案を合格。両方合格なら達成率の高いほう、片方だけならその案、どちらも合格しなければ B のまま。

■ やり直しの扱い
  結果は data/v9/confirm_<日時>/ に毎回別に保存し、前の結果は消さない。2回目以降は --rerun-reason（直したことの説明）が要る。
  data/v9/confirm_runs.csv に、実行ごとの日時・判定・理由を残す。
"""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime

import numpy as np
import pandas as pd

from . import bets, market, predict
from . import trifecta as tri
from . import v8

TARGET = bets.TARGET
MARGIN = 1.4
MAX_POINTS = 15
BOX_SIZE = 6
UNTIL = "2026-10-25"
CONFIRM_Z = statistics.NormalDist().inv_cdf(1 - 0.025 / 2)      # 97.5% 区間
NAMES = {"F": "F（券種混合・市場＋3b）", "F2": "F2（券種混合・モデル0.5＋市場0.5）", "B": "B（モデルの上位6頭BOX）"}
RULE = ("F − B と F2 − B の達成率の差の 97.5% 区間の下限が 0 を超えた案を合格。両方合格なら達成率の高いほう、"
        "片方だけならその案を採用、どちらも合格しなければ B のまま")


def v9_dir():
    d = predict.data_root() / "v9"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_f2(table: pd.DataFrame) -> pd.DataFrame:
    """F2 の確率：exp(0.5·log モデル + 0.5·log 市場) をレース内で正規化。"""
    t = table.copy()
    u = 0.5 * np.log(t["prob"].astype("float64").clip(lower=1e-12)) + \
        0.5 * np.log(t["市場確率"].astype("float64").clip(lower=1e-12))
    e = np.exp(u - u.groupby(t["レースID"]).transform("max"))
    t["F2確率"] = e / e.groupby(t["レースID"]).transform("sum")
    t.loc[t["prob"].isna() | t["市場確率"].isna(), "F2確率"] = np.nan
    return t


def graded_rows(table: pd.DataFrame, folds, until: str) -> pd.DataFrame:
    t = table.loc[table["重賞"] & table["fold"].isin(set(folds))]
    return t.loc[pd.to_datetime(t["レース日付"]) <= pd.Timestamp(until)]


def evaluate(t: pd.DataFrame, payout: pd.DataFrame, disc_f: dict, disc_f2: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """重賞を1レースずつ、F・F2・B で評価する（1レース・1案1行）。どれかが作れないレースは全部の案から外す（同じレースで比べる）。

    戻り値: (結果, 外したレースとその理由)
    """
    t, starts = market.race_slices(t)
    pay = payout.set_index("レースID")
    rows, skipped = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        r = t.iloc[a:b]
        rid = r["レースID"].iloc[0]
        fold = r["fold"].iloc[0]
        q = r["市場確率"].to_numpy(dtype="float64")
        pf = r["tilt3b"].to_numpy(dtype="float64")
        p2 = r["F2確率"].to_numpy(dtype="float64")
        pm = r["prob"].to_numpy(dtype="float64")
        posts = pd.to_numeric(r["馬番"], errors="coerce").to_numpy()
        why = None
        if rid not in pay.index:
            why = "払戻が無い"
        elif b - a < 4:
            why = "出走4頭未満"
        elif not (np.isfinite(q).all() and (q > 0).all()):
            why = "単勝オッズが欠けている"
        elif not (np.isfinite(pf).all() and np.isfinite(p2).all() and np.isfinite(pm).all() and np.isfinite(posts).all()):
            why = "確率か馬番が欠けている"
        elif fold not in disc_f or fold not in disc_f2:
            why = "その fold の γ・δ が無い"
        wins = bets.winning_combos(pay.loc[rid]) if why is None else None
        if why is None and not wins["3連単"]:
            why = "3連単の払戻が無い"
        if why:
            skipped.append({"レースID": rid, "理由": why})
            continue
        posts = posts.astype(int)
        g, d, _ = disc_f[fold]
        Q = market.combo_probabilities(market.trifecta_tensor(q, g, d))       # 配当の推定は F・F2 とも市場から
        plan_f = bets.mixed(market.trifecta_tensor(pf, g, d), Q, pf, posts, margin=MARGIN, max_points=MAX_POINTS)
        g2, d2, _ = disc_f2[fold]
        plan_f2 = bets.mixed(market.trifecta_tensor(p2, g2, d2), Q, p2, posts, margin=MARGIN, max_points=MAX_POINTS)
        plan_b = bets.box(pm, posts, size=BOX_SIZE)
        base = {"レースID": rid, "レース日付": r["レース日付"].iloc[0], "fold": fold,
                "グレード": str(r["リステッド・重賞競走"].iloc[0]), "頭数": b - a}
        for key, plan in (("F", plan_f), ("F2", plan_f2), ("B", plan_b)):
            rows.append({**base, "案": NAMES[key], "投資": plan.invest, "払戻": bets.realized(plan, wins),
                         "点数": len(plan.bets),
                         "予測の達成確率": plan.predicted if key != "B" else np.nan,
                         "買い目": "; ".join(f"{x.kind} {'-'.join(map(str, x.combo))} {x.stake}円" for x in plan.bets)})
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["レースID", "理由"])


def summary(rows: pd.DataFrame, by=("案",)) -> pd.DataFrame:
    out = []
    for key, g in rows.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        hot = int((g["払戻"] >= TARGET).sum())
        lo, hi = tri.wilson_interval(hot, n)
        pred = g["予測の達成確率"].mean()
        out.append({**dict(zip(by, key)), "レース数": n, "熱い当たり": hot, "達成率": hot / n, "下限": lo, "上限": hi,
                    "的中率": float((g["払戻"] > 0).mean()),
                    "回収率": float(g["払戻"].sum() / g["投資"].sum()) if g["投資"].sum() else np.nan,
                    "平均点数": float(g["点数"].mean()), "平均投資": float(g["投資"].mean()),
                    "予測の達成確率（平均）": pred, "予測 − 実際": pred - hot / n if pd.notna(pred) else np.nan})
    return pd.DataFrame(out)


def paired(rows: pd.DataFrame, x: str, y: str, z: float = CONFIRM_Z) -> dict:
    hot = rows.assign(熱い=rows["払戻"] >= TARGET).pivot_table(index="レースID", columns="案", values="熱い",
                                                               aggfunc="first")
    both = hot[[x, y]].dropna()
    t = tri.paired_proportion_test(both[x].astype(bool).to_numpy(), both[y].astype(bool).to_numpy(), z)
    return {"比較": f"{x} − {y}", "レース数": t["n"], "前者だけ熱い当たり": t["xだけ"], "後者だけ熱い当たり": t["yだけ"],
            "両方": t["両方"], "差": t["差"], "差の下限": t["差下限"], "差の上限": t["差上限"], "p値": t["p値"]}


def judge(rows: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """判定のルール。戻り値: (採用する案のキー F / F2 / B, F − B・F2 − B の表)。"""
    tab = pd.DataFrame([paired(rows, NAMES["F"], NAMES["B"]), paired(rows, NAMES["F2"], NAMES["B"])])
    tab["合格"] = tab["差の下限"] > 0
    passed = [k for k, ok in zip(("F", "F2"), tab["合格"]) if ok]
    if not passed:
        return "B", tab
    if len(passed) == 1:
        return passed[0], tab
    rate = summary(rows).set_index("案")["達成率"]
    return max(passed, key=lambda k: rate[NAMES[k]]), tab


def hot_payout_distribution(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in rows.groupby("案", sort=False):
        h = g.loc[g["払戻"] >= TARGET, "払戻"]
        out.append({"案": name, "件数": len(h), "中央値": h.median() if len(h) else np.nan,
                    "第1四分位": h.quantile(0.25) if len(h) else np.nan,
                    "第3四分位": h.quantile(0.75) if len(h) else np.nan, "最大": h.max() if len(h) else np.nan})
    return pd.DataFrame(out)


def _previous_runs():
    path = v9_dir() / "confirm_runs.csv"
    return (pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()), path


def _cmd_confirm(args) -> int:
    prev, runs_path = _previous_runs()
    if not prev.empty and not args.rerun_reason:
        print("確認のテストはすでに実行されています（data/v9/confirm_runs.csv）:")
        print(prev.to_string(index=False))
        print("\n実装のバグを直してやり直すときだけ、--rerun-reason \"直したこと\" を付けて実行してください"
              "（F・F2・B の定義と判定のルールは変えない。前の結果は消さずに残す）")
        return 1
    clock = v8._Clock()
    ds = v8._load(args, clock)
    base = v8._oof(args, ds, "base")
    merged, _, _ = v8.tilt_tables(args, ds, base)
    folds = [f.name for f in v8._folds(args)][1:]                           # fold 2〜5
    clock.step("γ・δ を推定中（F：市場の確率 / F2：モデル0.5＋市場0.5。どちらも前の fold の平場で）...")
    disc_f = v8.fold_discounts(merged, "市場確率", base.table)
    b2 = add_f2(base.table)
    disc_f2 = v8.fold_discounts(b2.loc[b2["fold"].isin(set(merged["fold"]))], "F2確率", b2)
    k = ["レースID", "血統登録番号"]
    t = graded_rows(merged, folds, args.until).merge(b2[k + ["F2確率"]], on=k, how="left")
    clock.step(f"重賞 {t['レースID'].nunique():,} レースで F・F2・B を計算中 ...")
    rows, skipped = evaluate(t, ds.payout_df, disc_f, disc_f2)
    # 3b の確率を作れない重賞（1着が同着・単勝オッズが欠けている）も、外したレースとして数える
    every = graded_rows(base.table, folds, args.until)["レースID"].unique()
    lost = sorted(set(every) - set(t["レースID"]))
    skipped = pd.concat([skipped, pd.DataFrame({"レースID": lost, "理由": "1着が同着か単勝オッズが欠けている"})],
                        ignore_index=True)
    clock.step("完了\n")
    if rows.empty:
        print("評価できる重賞がありません")
        return 1
    adopt, tab = judge(rows)
    verdict = {"F": f"F を採用（{NAMES['F']}）", "F2": f"F2 を採用（{NAMES['F2']}）",
               "B": "B のまま（どちらも合格しなかった）"}[adopt]
    span = f"{pd.to_datetime(rows['レース日付']).min():%Y-%m-%d} 〜 {pd.to_datetime(rows['レース日付']).max():%Y-%m-%d}"
    print("=" * 110)
    print(f"■ 判定：{verdict}")
    for _, r in tab.iterrows():
        print(f"  {r['比較']}: 差 {r['差']:+.2%}  97.5% 区間 [{r['差の下限']:+.2%}, {r['差の上限']:+.2%}]  p値 {r['p値']:.4f}"
              f"  → {'合格' if r['合格'] else '不合格'}")
    print(f"  重賞 {rows['レースID'].nunique():,} レース（{span}、fold {', '.join(folds)}）。外したレース {len(skipped):,}")
    print(f"  ルール（結果を見る前に決めたもの）：{RULE}")
    print("=" * 110)
    print("\n=== 判定に使った比較（同じレース。97.5% 区間） ===")
    v8._show(tab, pct=("差",))
    ref = pd.DataFrame([paired(rows, NAMES["F"], NAMES["F2"])])
    print("\n=== 参考：F − F2（判定には使わない。97.5% 区間） ===")
    v8._show(ref, pct=("差",))
    summ = summary(rows)
    print("\n=== 参考：達成率（Wilson 95% 区間）・的中率・回収率・点数・投資・予測の達成確率 ===")
    v8._show(summ, pct=("率", "限", "確率（", "予測 −"), money=("投資",))
    grade = summary(rows, by=("グレード", "案"))
    print("\n=== 参考：グレード別（数が少ないので参考のみ） ===")
    v8._show(grade[["グレード", "案", "レース数", "熱い当たり", "達成率", "下限", "上限"]], pct=("率", "限"))
    by_fold = summary(rows, by=("fold", "案"))
    print("\n=== 参考：fold ごと ===")
    v8._show(by_fold[["fold", "案", "レース数", "熱い当たり", "達成率", "下限", "上限"]], pct=("率", "限"))
    dist = hot_payout_distribution(rows)
    print("\n=== 参考：熱い当たりになったレースの払戻の分布（円） ===")
    v8._show(dist, money=("中央値", "四分位", "最大"))
    if len(skipped):
        print("\n=== 外したレース（理由別） ===")
        print(skipped.groupby("理由").size().to_string())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = v9_dir() / f"confirm_{stamp}"
    n = 2
    while out.exists():                                     # 前の結果は上書きしない
        out = v9_dir() / f"confirm_{stamp}_{n}"
        n += 1
    out.mkdir(parents=True)
    for name, df in [("judgement", tab.assign(判定=verdict, ルール=RULE)), ("f_minus_f2", ref), ("summary", summ),
                     ("by_grade", grade), ("by_fold", by_fold), ("hot_payouts", dist), ("races", rows),
                     ("skipped", skipped)]:
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    log = pd.concat([prev, pd.DataFrame([{"実行日時": stamp, "判定": verdict, "重賞のレース数": rows["レースID"].nunique(),
                                          "F−B 下限": tab["差の下限"].iloc[0], "F2−B 下限": tab["差の下限"].iloc[1],
                                          "やり直しの理由": args.rerun_reason or "", "保存先": str(out)}])],
                    ignore_index=True)
    log.to_csv(runs_path, index=False, encoding="utf-8-sig")
    print(f"\n  保存: {out}（実行の記録: {runs_path}）")
    if not prev.empty:
        print(f"\n[記録] やり直し（理由: {args.rerun_reason}）。前回までの判定:")
        print(prev[["実行日時", "判定", "やり直しの理由"]].to_string(index=False))
    print(f"\n■ 判定（再掲）：{verdict}")
    return 0


# ---------------------------------------------------------------------------
# v11 2：点数の少ない買い目の「勝者の呪い」の確認（記述だけ。判定には使わない。案の数にも数えない）
# ---------------------------------------------------------------------------
POINT_BUCKETS = [(1, 1, "1点"), (2, 3, "2〜3点"), (4, 6, "4〜6点"), (7, 99, "7点以上")]
CONC_BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0001]


def _bucket(n: int) -> str:
    return next(label for lo, hi, label in POINT_BUCKETS if lo <= n <= hi)


def curse_rows(t: pd.DataFrame, payout: pd.DataFrame, disc_f: dict, kind: str, final_odds: dict | None = None):
    """F の買い目を作り直し、レースごと（点数・予測・実際・集中度）と買い目ごと（推定配当・当たる確率・確定オッズ）の行。"""
    from . import realodds
    t, starts = market.race_slices(t)
    pay = payout.set_index("レースID")
    races, bet_rows = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        r = t.iloc[a:b]
        rid = r["レースID"].iloc[0]
        fold = r["fold"].iloc[0]
        q = r["市場確率"].to_numpy(dtype="float64")
        pf = r["tilt3b"].to_numpy(dtype="float64")
        posts = pd.to_numeric(r["馬番"], errors="coerce").to_numpy()
        if rid not in pay.index or b - a < 4 or fold not in disc_f or not (np.isfinite(q).all() and (q > 0).all()
                                                                            and np.isfinite(pf).all() and np.isfinite(posts).all()):
            continue
        wins = bets.winning_combos(pay.loc[rid])
        if not wins["3連単"]:
            continue
        posts = posts.astype(int)
        g, d, _ = disc_f[fold]
        Q = market.combo_probabilities(market.trifecta_tensor(q, g, d))
        P = market.trifecta_tensor(pf, g, d)
        Pc = market.combo_probabilities(P)
        plan = bets.mixed(P, Q, pf, posts, margin=MARGIN, max_points=MAX_POINTS)
        if not plan.bets:
            continue
        index = {int(x): i for i, x in enumerate(posts)}
        T = None
        if final_odds is not None and str(rid) in final_odds:
            T = realodds.payout_tables(final_odds[str(rid)], posts)
        pb = []
        for x in plan.bets:
            ix = [index[h] for h in x.combo]
            look = (lambda M: M[tuple(ix)]) if x.kind in ("3連単", "3連複") else (lambda M: M[ix[0], ix[1]])
            p_b = float(look(Pc[x.kind]))
            est = market.estimated_payout(float(look(Q[x.kind])), x.kind)
            key = tuple(sorted(x.combo)) if x.kind in ("3連複", "馬連") else tuple(x.combo)
            real = float(look(T[x.kind])) if T is not None else np.nan
            pb.append(p_b)
            bet_rows.append({"区分": kind, "レースID": rid, "券種": x.kind, "金額": x.stake, "当たる確率（予測）": p_b,
                             "推定配当": est, "確定オッズの配当": real if np.isfinite(real) else np.nan,
                             "確定÷推定": real / est if np.isfinite(real) and est > 0 else np.nan,
                             "当たった": key in wins.get(x.kind, {})})
        realized = bets.realized(plan, wins)
        races.append({"区分": kind, "レースID": rid, "点数": len(plan.bets), "点数の区分": _bucket(len(plan.bets)),
                      "予測の達成確率": plan.predicted, "熱い": realized >= TARGET,
                      "集中度": max(pb) / sum(pb) if sum(pb) > 0 else np.nan})
    return pd.DataFrame(races), pd.DataFrame(bet_rows)


def _rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        n, hot = len(g), int(g["熱い"].sum())
        lo, hi = tri.wilson_interval(hot, n)
        pred = float(g["予測の達成確率"].mean())
        out.append({**dict(zip(by, key)), "レース数": n, "予測の達成確率（平均）": pred, "実際の達成率": hot / n,
                    "下限": lo, "上限": hi, "実際 − 予測": hot / n - pred})
    return pd.DataFrame(out)


def _cmd_curse(args) -> int:
    from . import realodds
    clock = v8._Clock()
    ds = v8._load(args, clock)
    base = v8._oof(args, ds, "base")
    merged, _, _ = v8.tilt_tables(args, ds, base)
    folds = [f.name for f in v8._folds(args)][1:]
    disc_f = v8.fold_discounts(merged, "市場確率", base.table)
    graded = graded_rows(merged, folds, args.until)
    flat = merged.loc[~merged["重賞"] & merged["fold"].isin(set(folds))]
    path = v8.experiment.v8_dir() / "odds_targets.csv"
    targets = pd.read_csv(path, dtype={"レースID": str}, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    if not args.flat_all:
        ids = set(targets.loc[targets["区分"] == "平場サンプル", "レースID"]) if not targets.empty else set()
        if not ids:
            rng = np.random.default_rng(0)
            u = flat["レースID"].unique()
            ids = set(rng.choice(u, size=min(5000, len(u)), replace=False))
        flat = flat.loc[flat["レースID"].astype(str).isin(ids)]
    odds_ids = set(targets["レースID"]) if not targets.empty else set()
    clock.step(f"確定オッズを読み込み中（取り直した {len(odds_ids):,} レース）...")
    final_odds = realodds.load_final_odds(odds_ids, base=args.data) if odds_ids else {}
    clock.step(f"F の買い目を作り直し中（重賞 {graded['レースID'].nunique():,}・平場 {flat['レースID'].nunique():,} レース）...")
    r1, b1 = curse_rows(graded, ds.payout_df, disc_f, "重賞", final_odds)
    r2, b2 = curse_rows(flat, ds.payout_df, disc_f, "平場", final_odds)
    clock.step("完了\n")
    races = pd.concat([r1, r2], ignore_index=True)
    bets_df = pd.concat([b1, b2], ignore_index=True)
    print("※ 記述だけ（判定には使わない。案の数にも数えない）。F の定義は変えない")
    print("\n=== 1 最終の点数ごとの、予測の達成確率と実際（Wilson 95% 区間） ===")
    races["点数の区分"] = pd.Categorical(races["点数の区分"], [x[2] for x in POINT_BUCKETS], ordered=True)
    t1 = _rate_table(races.sort_values("点数の区分"), ["区分", "点数の区分"])
    v8._show(t1, pct=("率", "限", "確率（", "−"))
    print("\n=== 2 買い目ごとの 確定オッズの配当 ÷ 推定配当（取り直した確定オッズがあるレースだけ）。比の5区間ごとに、"
          "予測の当たる確率の合計と実際の的中数 ===")
    bb = bets_df.loc[bets_df["確定÷推定"].notna()].copy()
    t2 = pd.DataFrame()
    if len(bb) >= 5:
        bb["比の区間"] = pd.qcut(bb["確定÷推定"].rank(method="first"), 5, labels=[f"第{i}五分位" for i in range(1, 6)])
        rows = []
        for (k, band), g in bb.groupby(["区分", "比の区間"], observed=True):
            rows.append({"区分": k, "比の区間": band, "買い目の数": len(g),
                         "確定÷推定（範囲）": f"{g['確定÷推定'].min():.2f}〜{g['確定÷推定'].max():.2f}",
                         "確定÷推定（中央値）": float(g["確定÷推定"].median()),
                         "予測の的中数（当たる確率の合計）": float(g["当たる確率（予測）"].sum()),
                         "実際の的中数": int(g["当たった"].sum()),
                         "実際 ÷ 予測": float(g["当たった"].sum() / g["当たる確率（予測）"].sum())})
        t2 = pd.DataFrame(rows)
        v8._show(t2)
        print("    確定÷推定 が小さい（推定配当を高く見すぎた）組ほど、実際の的中が予測より少なければ「勝者の呪い」に近い形")
    else:
        print("（確定オッズのある買い目がありません。v8 の odds-targets と fetch-odds --targets のあとに）")
    print("\n=== 3 集中度（予測の当たる確率が一番大きい買い目 ÷ 買い目全体の当たる確率の合計）ごとの、予測と実際 ===")
    races["集中度の区間"] = pd.cut(races["集中度"], CONC_BINS, right=False,
                               labels=["〜0.3", "0.3〜0.5", "0.5〜0.7", "0.7〜0.9", "0.9〜"])
    t3 = _rate_table(races.sort_values("集中度の区間"), ["区分", "集中度の区間"])
    v8._show(t3, pct=("率", "限", "確率（", "−"))
    out = v9_dir() / "curse"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [("by_points", t1), ("by_ratio", t2), ("by_concentration", t3), ("races", races), ("bets", bets_df)]:
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n  保存: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m src.v9")
    parser.add_argument("--data", default=None, help="JV-Data CSV の場所（既定: data/jvlink）")
    parser.add_argument("--folds", type=int, default=None, help="先頭から何 fold だけ（動作確認用）")
    parser.add_argument("--refresh", action="store_true", help="保存済みの fold の予測を使わず作り直す")
    parser.add_argument("--rounds", type=int, default=predict.DEFAULT_ROUNDS, help="ラウンド数（既定 454）")
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("confirm", help="重賞で1回だけの確認のテスト（F・F2・B）")
    q.add_argument("--until", default=UNTIL, help=f"対象の最後の日（既定 {UNTIL}）")
    q.add_argument("--rerun-reason", default=None, help="やり直すときだけ：直した実装のバグの説明")
    q.set_defaults(func=_cmd_confirm)
    q = sub.add_parser("curse", help="v11 点数の少ない買い目の「勝者の呪い」の確認（記述だけ。重賞の F と平場の F）")
    q.add_argument("--until", default=UNTIL, help=f"重賞の対象の最後の日（既定 {UNTIL}。v9 と同じ）")
    q.add_argument("--flat-all", action="store_true", help="平場の全レース（既定は取り直したオッズのある平場サンプル 5,000）")
    q.set_defaults(func=_cmd_curse)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
