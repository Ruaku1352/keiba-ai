"""競馬予想AI: 前処理・特徴量・評価・買い目抽出のパッケージ。

jvlink / jvmap / v5 はここで import しない。`py -m src.jvlink` のように
モジュールを直接実行するとき、パッケージの初期化で先に読み込まれていると
Python が RuntimeWarning（モジュールが二重に読み込まれる恐れ）を出すため。
使うときは `from src import jvlink` のように個別に import する。
"""

from . import (betting, config, corner, evaluate, features, leakfree, pace,  # noqa: F401
               paddock, payout, preprocess, records, selection, trifecta, validate)

__all__ = [
    "config", "leakfree", "preprocess", "features", "corner", "pace",
    "evaluate", "betting", "payout", "trifecta", "selection", "validate",
    "paddock", "records",
]
