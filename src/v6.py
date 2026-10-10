"""v6 の実行入口：実戦用の予測（Windows で1コマンドずつ実行する）。

    py -m src.v6 train                        2003年〜最新の確定レースで学習して保存する
    py -m src.v6 train --until 2026-09-30     学習の終了日を指定する（練習モード用）
    py -m src.v6 predict                      今日以降の出馬表の重賞を予測する
    py -m src.v6 predict --date 2026-10-11    その日の重賞だけ予測する
    py -m src.v6 replay --date 2026-10-04     過去の日の重賞を「結果未確定の出馬表」として予測し、答え合わせ
    py -m src.v6 models                       保存したモデルの一覧

    py -m src.v6 diagnose weight              馬体重の代用で AUC・達成率がどれだけ落ちるか（5 fold）
    py -m src.v6 diagnose payout-trend        年ごとの3連単払戻の中央値・5万円以上の割合（重賞 vs 平場）
    py -m src.v6 diagnose check-3f            ラップから計算した前3F・後3Fと公式値の一致（距離ごと）
    py -m src.v6 diagnose style-zero          今回レース脚質判定の「0」の内訳（年ごと）

買い方は変えない：重賞のみ（G1/G2/G3/G）・予測上位6頭の3連単BOX（120点・12,000円）。
データの取得は `py -m src.jvlink fetch --option 1`（出馬表もこれで届く）。
モデル・予測は data\\models\\・data\\predictions\\ に保存する（git には入らない）。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, jvmap, predict


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.t0 = time.perf_counter()

    def step(self, what: str) -> None:
        print(f"[{time.perf_counter() - self.t0:7.0f}秒] {what}", flush=True)


def _load(args, clock: _Clock, end: str | None = None):
    clock.step("生データ（RA/SE/HR）を読み込み中 ...")
    raw = jvmap.load_raw(args.data)
    clock.step("確定したレースを変換中 ...")
    ds = jvmap.build_dataset(raw=raw, end=end)
    rr = ds.race_result
    print(f"         確定レース: {rr['レース日付'].min():%Y-%m-%d} 〜 {rr['レース日付'].max():%Y-%m-%d}"
          f"（{rr['レースID'].nunique():,} レース）")
    return raw, ds


def _jv_state(args) -> dict:
    path = Path(args.data or config.JV_DATA_DIR) / "state.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _format_stamp(stamp: str | None) -> str:
    """"20261010112818" -> "2026-10-10 11:28"。"""
    if not stamp or len(stamp) < 12:
        return stamp or "不明"
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}"


def _graded(df: pd.DataFrame) -> pd.Series:
    return df["リステッド・重賞競走"].isin(config.GRADED_VALUES)


def _model_info(model: predict.SavedModel) -> str:
    m = model.meta
    return (f"モデル: {m['name']}（学習 {m['train_first_date']}〜{m['train_last_date']}、"
            f"{m['rounds']}ラウンド、{len(m['feature_cols'])}特徴量）")


def _check_features(model: predict.SavedModel, feats: pd.DataFrame) -> None:
    missing = [c for c in model.feature_cols if c not in feats.columns]
    if missing:
        raise SystemExit(f"モデルの特徴量が予測の表にありません: {missing}"
                         "（コードを更新したら train し直してください）")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _cmd_train(args) -> int:
    clock = _Clock()
    _, ds = _load(args, clock, end=args.until)
    clock.step(f"特徴量を作成中（as_of={predict.AS_OF}：前日までに確定したレースだけで集計）...")
    frame = predict.make_training_frame(ds.race_result, ds.lap_df)
    clock.step(f"学習中（{args.rounds} ラウンド、乱数シード固定）...")
    booster, cols, info = predict.train_model(frame, rounds=args.rounds, until=args.until)
    state = _jv_state(args)
    data_info = {
        "data_last_race_date": f"{ds.race_result['レース日付'].max():%Y-%m-%d}",
        "data_races": int(ds.race_result["レースID"].nunique()),
        "jvlink_last_file_timestamp": {k: v.get("last_file_timestamp") for k, v in state.items()},
        "jvlink_fetched_at": {k: v.get("fetched_at") for k, v in state.items()},
    }
    path = predict.save_model(booster, cols, info, args.rounds, args.until, data_info, name=args.name)
    clock.step("完了")
    print(f"\n保存しました: {path}")
    print(f"  学習期間     : {info['train_first_date']} 〜 {info['train_last_date']}")
    print(f"  学習データ   : {info['train_races']:,} レース / {info['train_rows']:,} 頭")
    print(f"  ラウンド数   : {args.rounds}")
    print(f"  特徴量       : {len(cols)} 個")
    print(f"  データの時点 : 最新の確定レース {data_info['data_last_race_date']}"
          f" / 取得ファイル {_format_stamp((state.get('RACE') or {}).get('last_file_timestamp'))}")
    print(f"  git commit   : {predict._git_commit()}")
    return 0


def _cmd_models(args) -> int:
    root = predict.model_dir()
    if not root.exists():
        print("保存したモデルはありません")
        return 0
    latest = (root / "latest.txt").read_text(encoding="utf-8").strip() if (root / "latest.txt").exists() else ""
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        mark = " ← latest" if d.name == latest else ""
        print(f"  {d.name:<28} 学習 {m['train_first_date']}〜{m['train_last_date']} "
              f"{m['rounds']}R 作成 {m['created_at']}{mark}")
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def _pending_results_warnings(raw: dict, last_confirmed: pd.Timestamp,
                              target_dates: list[pd.Timestamp]) -> list[str]:
    """予測対象日より前で、まだ結果が届いていない開催日（その日の成績は特徴量に入らない）。"""
    ra = jvmap.central_only(jvmap.apply_data_kubun(raw["RA"], jvmap.RACE_KEY,
                                                   jvmap.RACE_KUBUN_PRIORITY))
    ra = ra.loc[~ra["head.DataKubun"].isin(jvmap.USABLE_RACE_KUBUN | {"9", "0"})]
    dates = jvmap.make_date(ra)
    out = []
    for day in target_dates:
        pending = dates.loc[(dates > last_confirmed) & (dates < day)]
        for d, n in pending.value_counts().sort_index().items():
            out.append(f"{d:%Y-%m-%d} の {n} レースは結果がまだ届いていません。"
                       f"{day:%Y-%m-%d} の予測にはその日の成績が入っていません"
                       "（結果が届いたら取り直して再実行すれば反映されます）")
    return out


def _cmd_predict(args) -> int:
    clock = _Clock()
    raw, ds = _load(args, clock)
    from_date = args.date or args.from_date or f"{datetime.now():%Y-%m-%d}"
    to_date = args.date or args.to_date
    entries = jvmap.prepare_entries(raw, from_date=from_date, to_date=to_date)
    if entries.empty:
        print(f"\n{from_date} 以降の出馬表（データ区分 1・2）に重賞がありません。")
        print("  `py -m src.jvlink fetch --option 1` で今週のデータを取得したか確認してください。")
        return 1
    model = predict.load_model(args.model)
    print("         " + _model_info(model))

    target_dates = sorted(pd.to_datetime(entries["レース日付"]).unique())
    last_confirmed = pd.Timestamp(ds.race_result["レース日付"].max())
    warnings = _pending_results_warnings(raw, last_confirmed, target_dates)
    for h in predict.check_duplicate_horses(entries):
        warnings.append(f"同じ馬が複数の予測対象に入っています: {h}")

    clock.step(f"予測の経路で特徴量を作成中（{len(target_dates)}日分。1日ごとに全履歴から作るので数分かかる）...")
    feats = predict.build_prediction_features(ds.race_result, entries, ds.lap_df, going=args.going)
    _check_features(model, feats)
    scored = predict.score(model, feats)
    stamp = _format_stamp((_jv_state(args).get("RACE") or {}).get("last_file_timestamp"))
    scored["データ時点"] = stamp
    clock.step("完了\n")

    for w in warnings:
        print("【注意】" + w)
    if warnings:
        print()
    for _, race in scored.groupby("レースID", sort=False):
        print(predict.render_race(race, model))
        print()

    table = predict.output_table(scored, model)
    table["データ時点"] = stamp
    path = predict.save_predictions(table, f"predict_{from_date.replace('-', '')}")
    print(f"CSV を保存しました: {path}")
    print("  先頭7列（レースID, 日付, レース名, 馬番, 馬名, 予測確率, 予測順位）はそのまま「評価入力」シートに貼れます")
    return 0


# ---------------------------------------------------------------------------
# replay（練習モード）
# ---------------------------------------------------------------------------
def _cmd_replay(args) -> int:
    clock = _Clock()
    _, ds = _load(args, clock)
    day = pd.Timestamp(args.date)
    rr = ds.race_result
    dates = pd.to_datetime(rr["レース日付"])
    targets = rr.loc[(dates == day) & _graded(rr)]
    if targets.empty:
        near = sorted(dates.loc[_graded(rr) & (dates <= day)].unique())[-5:]
        print(f"\n{day:%Y-%m-%d} に確定した重賞がありません。直近の重賞の日: "
              + ", ".join(f"{pd.Timestamp(d):%Y-%m-%d}" for d in near))
        return 1
    model = predict.load_model(args.model)
    print("         " + _model_info(model))

    clock.step("予測の経路で特徴量を作成中（結果の列を消し、馬体重は前走の値で代用）...")
    history = rr.loc[dates < day]
    feats = predict.build_prediction_features(history, targets, ds.lap_df,
                                              keep_actual_going=not args.assume_going)
    _check_features(model, feats)

    diff = None
    if not args.skip_check:
        clock.step("学習の経路でも同じレースの特徴量を作成中（一致の確認用）...")
        train = predict.make_training_frame(rr, ds.lap_df, until=f"{day:%Y-%m-%d}")
        train_rows = train.loc[(train["レース日付"] == day) & _graded(train)]
        exclude = predict.WEIGHT_COLUMNS + ((predict.GOING_COLUMN,) if args.assume_going else ())
        diff = predict.compare_paths(train_rows, feats, model.feature_cols, exclude=())
        diff["除外"] = diff["列"].isin(exclude)

    scored = predict.score(model, feats)
    key = ["レースID", "血統登録番号"]
    actual = targets[key + ["着順", "人気"]].rename(columns={"着順": "実際の着順", "人気": "実際の人気"})
    scored = scored.merge(actual, on=key, how="left")
    clock.step("完了\n")

    if model.until >= day:
        print(f"【注意】このモデルの学習終了日（{model.until:%Y-%m-%d}）が {day:%Y-%m-%d} 以降です。"
              "モデルがこのレースの結果を見ているので、的中したかどうかは参考になりません。")
        print(f"  見ていない状態で試すには: py -m src.v6 train --until {day - pd.Timedelta(days=1):%Y-%m-%d}\n")

    hits = hot = n = 0
    for rid, race in scored.groupby("レースID", sort=False):
        res = predict.actual_result(targets.loc[targets["レースID"] == rid], ds.payout_df, race)
        print(predict.render_race(race, model, actual=res))
        print()
        n += 1
        hits += res["的中"]
        hot += res["熱い"]
    print(f"まとめ: {day:%Y-%m-%d} の重賞 {n} レース / 的中 {hits} / 熱い基準（50,000円）超え {hot}")

    if diff is not None:
        checked = diff.loc[~diff["除外"]]
        print(f"\n学習の経路と予測の経路の一致（{len(checked)} 列 × {int(diff['比較した行'].max())} 頭）:")
        print(f"  一致しなかったセル（馬体重・増減を除く）: {int(checked['不一致'].sum())}"
              "  ← 0 になるはず")
        if diff.attrs.get("missing_rows"):
            print(f"  片方にしか無い行: {diff.attrs['missing_rows']}")
        bad = checked.loc[checked["不一致"] > 0]
        if not bad.empty:
            print(bad.to_string(index=False))
        for _, r in diff.loc[diff["除外"]].iterrows():
            print(f"  （意図的に変えている列）{r['列']}: {r['不一致']} セルが違う")

    table = predict.output_table(scored, model)
    path = predict.save_predictions(table, f"replay_{day:%Y%m%d}")
    print(f"\nCSV を保存しました: {path}")
    return 0


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
def _results_dir(args) -> Path:
    out = Path(args.data or config.JV_DATA_DIR) / "v6_results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _print_table(table: pd.DataFrame, pct_cols=(), money_cols=(), index=False) -> None:
    t = table.copy()
    for c in t.columns:
        if c in pct_cols or any(k in str(c) for k in pct_cols):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        elif c in money_cols or any(k in str(c) for k in money_cols):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    with pd.option_context("display.max_rows", 300, "display.max_columns", 50, "display.width", 250):
        print(t.to_string(index=index))


def _cmd_diag_weight(args) -> int:
    from . import diagnose, validate

    clock = _Clock()
    _, ds = _load(args, clock)
    folds = validate.V5_FOLDS[: args.folds] if args.folds else validate.V5_FOLDS
    out = diagnose.weight_impact_cv(ds, folds=folds, as_of=args.as_of)
    clock.step("完了\n")
    table = out["table"]
    print(f"=== 馬体重：実際の値 vs 代用（前走の馬体重・増減0）  as_of={args.as_of} ===")
    _print_table(table[["fold", "検証", "馬体重", "AUC", "重賞R数", "的中率", "達成率", "熱い当たり"]],
                 pct_cols=("的中率", "達成率"))
    print("\nプール:")
    _print_table(out["pooled"], pct_cols=("達成率", "下限", "上限"))
    if "差のp値" in out["pooled"].attrs:
        print(f"  達成率の差（代用−実際）: {out['pooled'].attrs['差（代用−実際）']:+.2%}"
              f"  p={out['pooled'].attrs['差のp値']:.3f}")

    print("\n=== モデル側の数字（実際の馬体重の予測。人気は確定時の単勝人気で、特徴量には使っていない） ===")
    real = table.loc[table["馬体重"] == "実際の馬体重"]
    _print_table(real[["fold", "検証", "上位6頭の単勝人気の平均", "全頭の単勝人気の平均", "的中率",
                       "的中レースの払戻の中央値", "的中のうち熱い割合", "重賞全体の払戻の中央値"]],
                 pct_cols=("的中率", "熱い割合"), money_cols=("中央値",))
    save = _results_dir(args) / f"weight_impact_{args.as_of}.csv"
    table.to_csv(save, index=False, encoding="utf-8-sig")
    print(f"\n保存しました: {save}")
    return 0


def _cmd_diag_payout(args) -> int:
    from . import diagnose

    clock = _Clock()
    _, ds = _load(args, clock)
    table = diagnose.payout_trend(ds, since_year=args.since_year)
    clock.step("完了\n")
    print("=== 年ごとの3連単払戻（モデルを使わない数字。100円あたり） ===")
    t = table.reset_index()
    _print_table(t, pct_cols=("割合",), money_cols=("中央値",))
    save = _results_dir(args) / "payout_trend.csv"
    table.to_csv(save, encoding="utf-8-sig")
    print(f"\n保存しました: {save}")
    return 0


def _cmd_diag_3f(args) -> int:
    from . import diagnose

    clock = _Clock()
    _, ds = _load(args, clock)
    table = diagnose.three_f_agreement(ds, since=args.since)
    clock.step("完了\n")
    print(f"=== 前3F・後3F：ラップからの計算値 vs 公式値（{args.since} 以降の平地、差 = 計算 − 公式、秒） ===")
    _print_table(table, pct_cols=("一致率",))
    save = _results_dir(args) / "three_f_agreement.csv"
    table.to_csv(save, index=False, encoding="utf-8-sig")
    print(f"\n保存しました: {save}")
    return 0


def _cmd_diag_style(args) -> int:
    from . import diagnose

    clock = _Clock()
    clock.step("生データ（SE）を読み込み中 ...")
    raw = jvmap.load_raw(args.data, record_types=("SE",))
    table = diagnose.style_zero_table(raw)
    clock.step("完了\n")
    print("=== SE 今回レース脚質判定 = 0（初期値）の内訳（年ごと） ===")
    _print_table(table.reset_index(), pct_cols=("割合",))
    total = table.sum(numeric_only=True)
    print(f"\n  全期間: 生レコード {int(total['生レコード']):,} / 0 は {int(total['件数_0']):,} 件")
    for c in ["重複（古い版）", "中央以外", "確定前の区分", "取消・除外・中止など", "最終データ"]:
        if c in table.columns:
            print(f"    {c:<14} {int(table[c].sum()):>10,}")
    save = _results_dir(args) / "style_zero.csv"
    table.to_csv(save, encoding="utf-8-sig")
    print(f"\n保存しました: {save}")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m src.v6")
    parser.add_argument("--data", default=None, help="JV-Data CSV の場所（既定: data/jvlink）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="学習して data/models に保存する")
    p.add_argument("--until", default=None, help="この日までのレースで学習する（例 2026-09-30）")
    p.add_argument("--rounds", type=int, default=predict.DEFAULT_ROUNDS,
                   help=f"ラウンド数（既定 {predict.DEFAULT_ROUNDS}）")
    p.add_argument("--name", default=None, help="保存名（既定 v6_untilYYYYMMDD）")
    p.set_defaults(func=_cmd_train)

    sub.add_parser("models", help="保存したモデルの一覧").set_defaults(func=_cmd_models)

    p = sub.add_parser("predict", help="出馬表（データ区分 1・2）の重賞を予測する")
    p.add_argument("--date", default=None, help="この日の重賞だけ予測する")
    p.add_argument("--from-date", default=None, help="この日以降（既定: 今日）")
    p.add_argument("--to-date", default=None, help="この日まで")
    p.add_argument("--going", default=None, choices=["良", "稍重", "重", "不良"],
                   help="馬場状態を指定する（既定: 未発表なら良と仮定）")
    p.add_argument("--model", default=None, help="使うモデル名（既定: 最後に学習したもの）")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("replay", help="過去の日の重賞を出馬表として予測し、答え合わせする")
    p.add_argument("--date", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--assume-going", action="store_true",
                   help="馬場状態も未発表として「良」と仮定する（既定は実際の馬場状態を使う）")
    p.add_argument("--skip-check", action="store_true",
                   help="学習の経路との一致確認を省く（数分短くなる）")
    p.set_defaults(func=_cmd_replay)

    p = sub.add_parser("diagnose", help="確認・診断")
    dsub = p.add_subparsers(dest="what", required=True)
    q = dsub.add_parser("weight", help="馬体重の代用の影響（validate と同じ fold）")
    q.add_argument("--folds", type=int, default=None, help="先頭から何foldだけ回すか")
    q.add_argument("--as-of", default=predict.AS_OF, choices=["day", "race"],
                   help="day = v6 のモデルと同じ特徴量（既定） / race = validate と同じ特徴量")
    q.set_defaults(func=_cmd_diag_weight)
    q = dsub.add_parser("payout-trend", help="年ごとの3連単払戻（重賞 vs 平場）")
    q.add_argument("--since-year", type=int, default=2003)
    q.set_defaults(func=_cmd_diag_payout)
    q = dsub.add_parser("check-3f", help="ラップから計算した3Fと公式値の一致")
    q.add_argument("--since", default="2003-01-01")
    q.set_defaults(func=_cmd_diag_3f)
    dsub.add_parser("style-zero", help="今回レース脚質判定の0の内訳").set_defaults(func=_cmd_diag_style)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
