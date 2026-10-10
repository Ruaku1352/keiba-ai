"""race_result.csv の読み込みと前処理。

進捗まとめ（前処理セクション）でやっていた内容を関数にまとめ直したもの。
メモリ節約のため、読み込み時の dtype 指定とダウンキャストを入れている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# カテゴリとして扱いたい論理カラム（LightGBM の category 型にする）
CATEGORICAL_LOGICAL = ["jockey", "trainer", "course", "surface", "turn", "going"]


def load_race_result(path: str | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    """CSV を読み込む。usecols を渡すと必要な列だけ読んでメモリを節約できる。"""
    path = path or config.RACE_RESULT_CSV
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """着順・タイム欠損の除去、型変換、目的変数の作成までを行う。"""
    cols = config.resolve_columns(df)
    config.require(cols, "race_id", "date", "horse", "rank")

    df = df.copy()

    # --- 着順・タイムが欠損している行（中止・失格など）を落とす ------------
    subset = [cols["rank"]]
    if "time" in cols:
        subset.append(cols["time"])
    df = df.dropna(subset=subset)

    # --- 着順を数値に。数値化できない行（"取消" 等）は落とす ---------------
    rank = pd.to_numeric(df[cols["rank"]], errors="coerce")
    df = df.loc[rank.notna()].copy()
    df[cols["rank"]] = rank.loc[rank.notna()].astype("int16")

    # --- 日付を datetime に ------------------------------------------------
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"]])

    # --- 体重増減の欠損は 0 埋め ------------------------------------------
    if "horse_weight_diff" in cols:
        df[cols["horse_weight_diff"]] = pd.to_numeric(
            df[cols["horse_weight_diff"]], errors="coerce"
        ).fillna(0)

    # --- 数値であってほしい列を数値化 --------------------------------------
    for logical in ["post", "bracket", "age", "weight_carried", "horse_weight",
                    "distance", "win_odds", "popularity", "last3f"]:
        if logical in cols:
            df[cols[logical]] = pd.to_numeric(df[cols[logical]], errors="coerce")

    # --- 目的変数：1着なら 1 ----------------------------------------------
    df[config.TARGET] = (df[cols["rank"]] == 1).astype("int8")

    # --- 並び順を固定する（以降の累積計算はこの順序に依存する）-------------
    # 日付 -> レースID -> 馬番 の順に並べておけば、
    # groupby().cumsum() が「時系列で過去→未来」の順に走ることが保証される。
    sort_keys = [cols["date"], cols["race_id"]]
    if "post" in cols:
        sort_keys.append(cols["post"])
    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    return df


def clean_entries(df: pd.DataFrame) -> pd.DataFrame:
    """予測対象（結果がまだ無い行）を、basic_clean と同じ型に整える。

    basic_clean との違いは「着順・タイムが無い行を落とさない」ことだけ。
    着順と目的変数は欠損のまま（NaN）にする。並び順も basic_clean と同じ。
    """
    cols = config.resolve_columns(df)
    config.require(cols, "race_id", "date", "horse", "rank")
    df = df.copy()
    df[cols["rank"]] = pd.to_numeric(df[cols["rank"]], errors="coerce").astype("float64")
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    if "horse_weight_diff" in cols:
        df[cols["horse_weight_diff"]] = pd.to_numeric(
            df[cols["horse_weight_diff"]], errors="coerce").fillna(0)
    for logical in ["post", "bracket", "age", "weight_carried", "horse_weight",
                    "distance", "win_odds", "popularity", "last3f"]:
        if logical in cols:
            df[cols[logical]] = pd.to_numeric(df[cols[logical]], errors="coerce")
    df[config.TARGET] = np.nan
    sort_keys = [cols["date"], cols["race_id"]] + ([cols["post"]] if "post" in cols else [])
    return df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


def to_category(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列を category dtype に変換（LightGBM 用 & メモリ削減）。"""
    cols = config.resolve_columns(df)
    for logical in CATEGORICAL_LOGICAL:
        name = cols.get(logical)
        if name is not None and not isinstance(df[name].dtype, pd.CategoricalDtype):
            df[name] = df[name].astype("category")
    return df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64/int64 を小さい型に落としてメモリを削減する。"""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def time_series_split(
    df: pd.DataFrame, quantile: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """レース日付で時系列分割する（Data Leakage 対策）。

    戻り値: (train_df, test_df, 分割日)
    """
    quantile = config.SPLIT_QUANTILE if quantile is None else quantile
    cols = config.resolve_columns(df)
    date_col = cols["date"]

    split_date = df[date_col].quantile(quantile)
    train_df = df[df[date_col] <= split_date]
    test_df = df[df[date_col] > split_date]
    return train_df, test_df, split_date


def memory_usage_mb(df: pd.DataFrame) -> float:
    """DataFrame のメモリ使用量（MB）。Colab で確認用。"""
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def filter_years(df: pd.DataFrame, start_year: int | None = None,
                 end_year: int | None = None) -> pd.DataFrame:
    """年で絞り込む（Colab が重いときの対策）。

    注意: 過去実績特徴量は「絞り込む前」に計算した方が精度が出る。
    （絞ってから計算すると、序盤の馬の通算成績がリセットされてしまうため）
    """
    cols = config.resolve_columns(df)
    year = df[cols["date"]].dt.year
    mask = np.ones(len(df), dtype=bool)
    if start_year is not None:
        mask &= (year >= start_year).to_numpy()
    if end_year is not None:
        mask &= (year <= end_year).to_numpy()
    return df.loc[mask].copy()
