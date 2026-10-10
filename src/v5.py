"""v5 の実行入口（Windows で1コマンドずつ確認するため）。

    py -m src.v5 build                 保存済み CSV を変換して中身を確認する
    py -m src.v5 grades                年 × 格付けのレース数（Kaggle 側との突き合わせ用）
    py -m src.v5 coverage              主要な列の年ごとの欠損率（何年から使えるか）
    py -m src.v5 --start-date none build   1986年より前のレースも含めて変換する
    py -m src.v5 validate              v4 と同じ時系列CVを回し、v4 と比較する
    py -m src.v5 validate --ablation   公式脚質特徴量の有無で AUC も比べる（時間は約2倍）

取得（JV-Link を叩く部分）は `py -m src.jvlink` 側。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, jvmap


def _date_arg(value: str | None) -> str | None:
    """"none" / "all" なら絞らない。"""
    if value is None or value.lower() in ("none", "all", ""):
        return None
    return value


def _build(args) -> jvmap.JVDataset:
    ds = jvmap.build_dataset(args.data, start=_date_arg(args.start_date),
                             end=_date_arg(args.end_date), jump_as_flat=args.jump_as_flat,
                             exclude_irregular_payout=args.exclude_irregular)
    return ds


def _cmd_coverage(args) -> int:
    ds = _build(args)
    table = jvmap.coverage_by_year(ds)
    print("年ごとの欠損率（%）。前3F・後3F は平地のレース単位、その他は馬単位。")
    print("3F_ラップ補完 = 公式値が空でラップから計算したレースの割合（%）")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(table.to_string())
    save = Path(args.data or config.JV_DATA_DIR) / "v5_results"
    save.mkdir(parents=True, exist_ok=True)
    table.to_csv(save / "coverage_by_year.csv", encoding="utf-8-sig")
    print(f"\n保存しました: {save / 'coverage_by_year.csv'}")
    return 0


def _cmd_build(args) -> int:
    ds = _build(args)
    print(ds.summary())
    rr = ds.race_result
    print("\n  列:", list(rr.columns))
    print("\n  先頭3行:")
    with pd.option_context("display.max_columns", 12, "display.width", 200):
        print(rr.head(3))
    return 0


def _cmd_grades(args) -> int:
    ds = _build(args)
    with pd.option_context("display.max_rows", 100):
        print(jvmap.grade_count_by_year(ds.race_result))
    return 0


def _cmd_validate(args) -> int:
    from . import validate

    ds = _build(args)
    folds = validate.V5_FOLDS[: args.folds] if args.folds else validate.V5_FOLDS
    out = validate.run_v5(ds, folds=folds, since=args.since, ablation=args.ablation)

    # 結果を CSV にも残す（data/ 配下なので git には入らない）
    save = Path(args.data or config.JV_DATA_DIR) / "v5_results"
    save.mkdir(parents=True, exist_ok=True)
    out["fold_summary"].to_csv(save / "fold_summary.csv", index=False, encoding="utf-8-sig")
    out["grade_table"].to_csv(save / "grade_table.csv", index=False, encoding="utf-8-sig")
    out["comparison"].to_csv(save / "comparison_v4.csv", index=False, encoding="utf-8-sig")
    out["style_importance"].to_csv(save / "style_importance.csv", index=False, encoding="utf-8-sig")
    pd.Series(out["recent"]).to_csv(save / "recent_check.csv", encoding="utf-8-sig")
    if "ablation" in out:
        out["ablation"].to_csv(save / "ablation.csv", index=False, encoding="utf-8-sig")
    print(f"\n結果を保存しました: {save}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m src.v5")
    parser.add_argument("--data", default=None, help="JV-Data CSV の場所（既定: data/jvlink）")
    parser.add_argument("--start-date", default=jvmap.DEFAULT_START,
                        help=f"この日以降のレースだけ使う（既定 {jvmap.DEFAULT_START}。none で絞らない）")
    parser.add_argument("--end-date", default=None, help="この日までのレースだけ使う（既定なし）")
    parser.add_argument("--jump-as-flat", action="store_true",
                        help="障害重賞(J.G1〜3)を平地と同じ G1〜3 として数える")
    parser.add_argument("--exclude-irregular", action="store_true",
                        help="不成立・特払・返還のあったレースを検証から外す（既定は外さない＝v4と同じ）")
    parser.add_argument("--keep-irregular", action="store_true",
                        help="（既定の動作と同じ。互換のために残してある）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build").set_defaults(func=_cmd_build)
    sub.add_parser("grades").set_defaults(func=_cmd_grades)
    sub.add_parser("coverage", help="主要な列の年ごとの欠損率").set_defaults(func=_cmd_coverage)

    p = sub.add_parser("validate")
    p.add_argument("--since", default="2021-08-01")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--folds", type=int, default=None, help="先頭から何foldだけ回すか（動作確認用）")
    p.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
