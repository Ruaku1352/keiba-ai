"""カラム名・パス・パラメータの設定をここに集約する。

Kaggle の JRA データセットはカラム名の表記ゆれがあるため、
「候補リスト」を持たせて実データから自動で解決できるようにしている。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# パス（Colab を想定。ローカルで動かすときは書き換える）
# ---------------------------------------------------------------------------
DATA_DIR = "/content/drive/MyDrive"
RACE_RESULT_CSV = f"{DATA_DIR}/race_result.csv"
ODDS_CSV = f"{DATA_DIR}/19860105-20210731_odds.csv"
LAPTIME_CSV = f"{DATA_DIR}/19860105-20210731_laptime.csv"
CORNER_CSV = f"{DATA_DIR}/20020615-20210731_corner_passing_order.csv"


# ---------------------------------------------------------------------------
# JRA-VAN（v5以降。Windows ローカルで使う）
# ---------------------------------------------------------------------------
# SDK の展開先。**SDK のファイルはリポジトリに入れない**（公開リポジトリのため、
# コミットすると JRA システムサービスの著作物の再配布になる）。
# 環境変数 JVSDK_DIR があればそれを優先し、なければ開発者の想定パスを使う。
JVSDK_DIR = os.environ.get(
    "JVSDK_DIR", r"C:\Users\tensu\keiba\JRA-VAN Data Lab. SDK Ver5.0.0_64bit"
)
# SDK 内の Python 版構造体の相対位置
JVSDK_STRUCT_RELPATH = Path("JV-Data構造体") / "Python版" / "JVData_Struct.py"

# 取得した JV-Data の保存先。**再配布禁止なので .gitignore 済み**
JV_DATA_DIR = os.environ.get("JV_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data" / "jvlink"))


# ---------------------------------------------------------------------------
# 論理名 -> 実カラム名の候補（先に見つかったものを採用する）
# ---------------------------------------------------------------------------
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "race_id": ["レースID", "レースキー", "race_id"],
    "date": ["レース日付", "年月日", "日付"],
    "horse": ["馬名", "血統登録番号", "馬ID"],
    # 末尾の「〜コード」は JRA-VAN 用。Kaggle には無い列なので、
    # 先頭の候補が見つかる Kaggle データの挙動は変わらない。
    "jockey": ["騎手", "騎手名", "騎手コード"],
    "trainer": ["調教師", "調教師名", "調教師コード"],
    "rank": ["着順"],
    "post": ["馬番"],
    "bracket": ["枠番"],
    "age": ["馬齢", "年齢"],
    "weight_carried": ["斤量", "負担重量"],
    "horse_weight": ["馬体重"],
    "horse_weight_diff": ["場体重増減", "馬体重増減"],
    "distance": ["距離(m)", "距離"],
    "course": ["競馬場名", "競馬場"],
    "surface": ["芝・ダート区分", "芝ダ障害コード"],
    "turn": ["右左回り・直線区分", "右左回り"],
    "going": ["馬場状態1", "馬場状態"],
    "win_odds": ["単勝オッズ", "オッズ", "確定単勝オッズ"],
    "popularity": ["人気", "確定単勝人気順位"],
    "time": ["タイム", "走破タイム"],
    "last3f": ["後３Ｆタイム", "上がり3ハロン", "後3Fタイム"],
    "grade": ["リステッド・重賞競走", "重賞コード", "グレード"],
    "race_name": ["競走名", "レース名", "競走名略称"],
}

# 重賞とみなす格付け（NaN は平場）
GRADED_VALUES = ("G1", "G2", "G3", "G")

# 「熱い」の定義：予算 -> 目標払戻（開発者の設定した基準）
HOT_CRITERIA = {
    1000: 10_000,
    3000: 20_000,
    5000: 30_000,
    10000: 50_000,
}

# 目的変数の名前
TARGET = "1着フラグ"


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """DataFrame の実カラムから「論理名 -> 実カラム名」の辞書を作る。

    見つからない論理名は辞書に入らない。呼び出し側は `.get()` で扱うこと。
    """
    mapping: dict[str, str] = {}
    for logical, candidates in COLUMN_CANDIDATES.items():
        for name in candidates:
            if name in df.columns:
                mapping[logical] = name
                break
    return mapping


def require(cols: dict[str, str], *logical_names: str) -> list[str]:
    """必須カラムが解決できているか確認して実カラム名を返す。"""
    missing = [n for n in logical_names if n not in cols]
    if missing:
        raise KeyError(
            f"必要なカラムが見つかりません: {missing} / "
            f"config.COLUMN_CANDIDATES に実際のカラム名を追加してください"
        )
    return [cols[n] for n in logical_names]


# ---------------------------------------------------------------------------
# 学習設定
# ---------------------------------------------------------------------------
# 時系列分割の位置（0.8 = 古い方 80% を学習、新しい方 20% を検証）
SPLIT_QUANTILE = 0.8

# LightGBM のベースパラメータ
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}

# 直近N走系の窓幅
RECENT_WINDOWS = (3, 5)
JOCKEY_RECENT_WINDOW = 100
