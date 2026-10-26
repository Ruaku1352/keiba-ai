"""v8 の検証の土台：5 fold の「検証期間の予測」（out-of-fold）を作ってためておく。

■ 条件（本番と同じ）
  - 特徴量は as_of="day"（前日までに確定したレースだけ）
  - 検証期間の馬体重は代用（前走の馬体重・増減0）。学習は実際の馬体重（validate と同じ）
  - ラウンド数は本番と同じ 454 で固定（validate は検証期間で early stopping していたが、
    v8 では検証期間を一切見ずに学習する。案どうしの比べ方を公平にするため）
  - fold は validate と同じ（学習 2003〜、検証 2011〜2026 の5区間）

■ 重賞を見ないためのガード
  v8 の新しい案（1.5 / 2 / 3）は、平場だけで比べる（重賞は次のフェーズで1回だけ確かめる）。
  新しい案の成績を計算する関数は exploration_only() を通し、重賞の行があればエラーにする。
  1（今のモデル・人気・ランダムの比較）と、キャリブレーション（確率の当たり具合）は例外で、全レースで出してよい。

■ 保存
  data/v8/oof_<名前>.pkl（git には入らない）。--refresh で作り直す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features, predict, preprocess

VARIANTS = {
    "base": {"extra": {}, "odds": False, "説明": "今のモデル（本番と同じ特徴量）"},
    "levels": {"extra": {"levels": {}}, "odds": False, "説明": "今の特徴量 + 格・着差・推移・タイム指数（1.5）"},
    "base_odds": {"extra": {}, "odds": True, "説明": "今の特徴量 + 市場確率（単勝オッズ）"},
    "levels_odds": {"extra": {"levels": {}}, "odds": True, "説明": "1.5 の特徴量 + 市場確率"},
    "trackbias40": {"extra": {"trackbias": {"cutoff_min": 20, "delay_min": 20}}, "odds": False,
                    "説明": "今の特徴量 + 当日の馬場の傾向（発走40分前までに発走したレース）"},
    "trackbias80": {"extra": {"trackbias": {"cutoff_min": 60, "delay_min": 20}}, "odds": False,
                    "説明": "今の特徴量 + 当日の馬場の傾向（発走80分前まで＝打ち切り60分前）"},
}

KEEP = ["レースID", "血統登録番号", "レース日付", "競馬場コード", "芝・ダート区分", "距離(m)",
        "リステッド・重賞競走", "着順", "人気", "単勝オッズ", "馬番", "馬名表示", "発走時刻"]


def v8_dir() -> Path:
    out = predict.data_root() / "v8"
    out.mkdir(parents=True, exist_ok=True)
    return out


class GradedResultsError(RuntimeError):
    """新しい案の重賞での成績を出そうとした（v8 の枠組みでは禁止）。"""


def is_graded(df: pd.DataFrame) -> pd.Series:
    return df["リステッド・重賞競走"].isin(config.GRADED_VALUES)


def exploration_only(df: pd.DataFrame, what: str = "") -> pd.DataFrame:
    """平場だけを返す。重賞が混ざっていたらエラー（呼ぶ側で平場に絞ってから渡すこと）。"""
    if is_graded(df).any():
        raise GradedResultsError(f"新しい案（{what}）の成績を重賞で出そうとしました。平場に絞ってください")
    return df


def market_probability(df: pd.DataFrame) -> pd.Series:
    """単勝オッズから市場の勝率（1/オッズ をレース内で合計1に正規化。控除の分を除く）。"""
    inv = 1.0 / pd.to_numeric(df["単勝オッズ"], errors="coerce").where(lambda s: s > 0)
    return inv / inv.groupby(df["レースID"]).transform("sum")


def build_frame(dataset, variant: str) -> pd.DataFrame:
    spec = VARIANTS[variant]
    df = preprocess.basic_clean(dataset.race_result)
    df = features.add_all_features(df, lap_df=dataset.lap_df, as_of=predict.AS_OF, extra=spec["extra"])
    if spec["odds"]:
        # 市場の確率（確定オッズ。本番では締切前のオッズになる）。レース前に分かる値として特徴量にする
        df["市場確率_log"] = np.log(market_probability(df).clip(lower=1e-4)).astype("float32")
    df = preprocess.downcast(df)
    return preprocess.to_category(df)


@dataclass
class OOF:
    variant: str
    table: pd.DataFrame          # 1頭1行。予測スコア・レース内の確率・fold
    importance: pd.DataFrame     # fold ごとの gain 重要度
    feature_cols: list[str]


def _train(train: pd.DataFrame, cols: list[str], rounds: int):
    import lightgbm as lgb
    data = lgb.Dataset(train[cols], label=train[config.TARGET])
    return lgb.train(predict.train_params(), data, num_boost_round=rounds)


def run_oof(dataset, variant: str, folds=None, rounds: int = predict.DEFAULT_ROUNDS,
            verbose: bool = True) -> OOF:
    """fold ごとに学習し、検証期間を予測する（検証の馬体重は代用）。"""
    from . import validate

    folds = folds or validate.V5_FOLDS
    if verbose:
        print(f"[v8] 特徴量を作成：{variant}（{VARIANTS[variant]['説明']}）...", flush=True)
    frame = build_frame(dataset, variant)
    frame.attrs = {}          # 途中の表の attrs を持ち越さない（pd.concat が attrs を比べるため）
    prev = predict.previous_weight(frame)
    cols = features.feature_columns(frame)
    parts, imps = [], []
    for fold in folds:
        train = validate._slice_years(frame, fold.train_years)
        test = validate._slice_years(frame, fold.test_years)
        if train.empty or test.empty:
            continue
        if verbose:
            print(f"  fold{fold.name} 学習 {fold.train_years[0]}-{fold.train_years[1]}"
                  f" ({len(train):,}行) → 検証 {fold.test_years[0]}-{fold.test_years[1]} ({len(test):,}行)",
                  flush=True)
        booster = _train(train, cols, rounds)
        test_sub = test.copy()
        test_sub["馬体重"] = prev.loc[test.index].astype("float32")
        test_sub["場体重増減"] = np.float32(0.0)
        out = test[[c for c in KEEP if c in test.columns]].copy()
        out["score"] = booster.predict(test_sub[cols])
        out["fold"] = fold.name
        parts.append(out)
        imps.append(pd.DataFrame({"fold": fold.name, "特徴量": cols,
                                  "gain": booster.feature_importance(importance_type="gain")}))
    table = pd.concat(parts, ignore_index=True)
    table["重賞"] = is_graded(table)
    table["prob"] = table["score"] / table.groupby("レースID")["score"].transform("sum")
    table["win"] = (pd.to_numeric(table["着順"], errors="coerce") == 1).astype("int8")
    table["市場確率"] = market_probability(table)
    return OOF(variant, table, pd.concat(imps, ignore_index=True), cols)


def load_or_run(dataset, variant: str, folds=None, refresh: bool = False,
                rounds: int = predict.DEFAULT_ROUNDS, verbose: bool = True) -> OOF:
    """data/v8/ にあれば読み、無ければ作って保存する。fold の数・ラウンド数が違えば作り直す。"""
    path = v8_dir() / f"oof_{variant}.pkl"
    n_folds = len(folds) if folds else 5
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("folds") == n_folds and meta.get("rounds") == rounds:
            obj = pd.read_pickle(path)
            if verbose:
                print(f"[v8] 保存済みの予測を使います: {path.name}（作り直すには --refresh）")
            return OOF(variant, obj["table"], obj["importance"], obj["feature_cols"])
    oof = run_oof(dataset, variant, folds=folds, rounds=rounds, verbose=verbose)
    pd.to_pickle({"table": oof.table, "importance": oof.importance, "feature_cols": oof.feature_cols}, path)
    meta_path.write_text(json.dumps({"folds": n_folds, "rounds": rounds, "variant": variant,
                                     "git_commit": predict._git_commit()}, ensure_ascii=False),
                         encoding="utf-8")
    return oof
