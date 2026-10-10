"""依頼B：ラップタイムからペース特徴量を作る。

■ 前提（実データで確認済みの落とし穴）
  laptime.csv の「前半3ハロン」「上がり3ハロン」列は**壊れている**。
  値が `ラップタイム1` と同じになっていて、3本の合計になっていない。
  なので、この2列は使わず**ラップタイムから自分で計算する**。

      前半3F = ラップ1 + ラップ2 + ラップ3      （= ペース3 と一致する）
      上がり3F = 有効なラップの末尾3本の合計

  ラップが入っているのは全体の約96%（障害競走などは欠損）。

■ ペース指標
      ペース指標 = 前半3F - 上がり3F

  プラスが大きいほど「前半が速く後半が失速した」＝ハイペース。
  マイナスなら前半が遅くて後半に脚を使った＝スローペース。

  ただし生の秒数は距離・馬場で大きく変わるので、そのままでは比較できない。
  そこで **(芝ダート × 距離帯) ごとに標準化した z スコア** に直す。
  この標準化にも「過去のレースだけ」を使う（未来の平均を使えばリークになる）。

■ リーク対策
  そのレース自身のペースは、レース前には分からない。
  だから特徴量にできるのは
    (1) 過去レースのペースから作った「その馬のペース適性」
    (2) 出走馬の脚質構成から推定した「今日の想定ペース」
  の2つだけ。実際のペースは絶対に特徴量へ入れない。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config, leakfree

_LAP_PATTERN = re.compile(r"^ラップタイム(\d+)$")
_PACE_PATTERN = re.compile(r"^ペース(\d+)$")

# 過去サンプルが何件たまったら統計量を信用するか。
# 少なすぎる母数で z 化すると値が暴れるので下限を設けている。
# （テストなど小さいデータで動かすときはここを下げる）
PACE_Z_MIN_COUNT = 30       # 条件別の平均・標準偏差
EXPECTED_PACE_MIN_COUNT = 50  # 前傾度ビンごとの想定ペース


def _matching_columns(columns, pattern: re.Pattern) -> list[str]:
    """`ラップタイム1..18` のような連番列を番号順に並べて返す。"""
    found = []
    for c in columns:
        m = pattern.match(str(c))
        if m:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found)]


def load_laptime(path: str | None = None) -> pd.DataFrame:
    """laptime.csv を読んで、レース単位のペース指標を計算する。

    戻り値の列: レースID, 前半3F, 上がり3F, ラップ本数, 合計タイム, ペース指標
    """
    path = path or config.LAPTIME_CSV
    raw = pd.read_csv(path, low_memory=False)
    return compute_race_pace(raw)


def compute_race_pace(raw: pd.DataFrame) -> pd.DataFrame:
    """ラップ列から前半3F・上がり3Fを計算する（壊れた既存列は無視）。"""
    id_col = "レースID" if "レースID" in raw.columns else raw.columns[0]
    lap_cols = _matching_columns(raw.columns, _LAP_PATTERN)
    if not lap_cols:
        # ラップ列がない場合はペース（累積タイム）から差分で復元する
        pace_cols = _matching_columns(raw.columns, _PACE_PATTERN)
        if not pace_cols:
            raise KeyError("ラップタイム列もペース列も見つかりません")
        cum = raw[pace_cols].to_numpy(dtype="float64")
        laps = np.diff(np.column_stack([np.zeros(len(cum)), cum]), axis=1)
        laps[np.isnan(cum)] = np.nan
    else:
        laps = raw[lap_cols].to_numpy(dtype="float64")

    valid = ~np.isnan(laps) & (laps > 0)
    n_valid = valid.sum(axis=1)

    # ラップは先頭から連続して入っている前提。念のため NaN を 0 にして合計する
    filled = np.where(valid, laps, 0.0)
    total = filled.sum(axis=1)

    # --- 前半3F = 先頭3本 ---------------------------------------------------
    first3 = np.where(n_valid >= 3, filled[:, :3].sum(axis=1), np.nan)

    # --- 上がり3F = 末尾3本 -------------------------------------------------
    # 有効ラップ数 n のとき、使うのは index n-3, n-2, n-1。
    # np.take_along_axis で行ごとに違う位置を一度に取り出す。
    idx = np.clip(np.column_stack([n_valid - 3, n_valid - 2, n_valid - 1]), 0, filled.shape[1] - 1)
    last3 = np.take_along_axis(filled, idx, axis=1).sum(axis=1)
    last3 = np.where(n_valid >= 3, last3, np.nan)

    out = pd.DataFrame({
        "レースID": leakfree.normalize_id(raw[id_col]),
        "前半3F": first3.astype("float32"),
        "上がり3F": last3.astype("float32"),
        "ラップ本数": n_valid.astype("int16"),
        "合計タイム": np.where(n_valid > 0, total, np.nan).astype("float32"),
    })
    # ペース指標：プラスならハイペース（前半が速く後半が遅い）
    out["ペース指標"] = (out["前半3F"] - out["上がり3F"]).astype("float32")
    return out


def attach_race_pace(df: pd.DataFrame, pace_df: pd.DataFrame) -> pd.DataFrame:
    """race_result にレースの実測ペースを貼り付ける。

    ⚠ 貼り付けた `_ペース指標` はそのレースの結果なので、特徴量にはしない。
       add_pace_features() が過去集計にだけ使う。
    """
    cols = config.resolve_columns(df)
    key = pd.DataFrame({"レースID": leakfree.normalize_id(df[cols["race_id"]])})
    merged = key.merge(
        pace_df[["レースID", "ペース指標", "前半3F", "上がり3F"]], on="レースID", how="left"
    )

    df = df.copy()
    for col, name in [("ペース指標", "_ペース指標"), ("前半3F", "_前半3F"), ("上がり3F", "_上がり3F")]:
        df[name] = pd.Series(merged[col].to_numpy(), index=df.index, dtype="float32")
    return df


def _race_frame(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """1行1レースに畳んだフレームを作る。

    なぜ畳むのか（重要）:
      ペースはレース単位の値なので、馬ごとの行のまま累積すると
      **同じレースの他の馬の行**が「過去」として混ざる。
      それはそのレース自身のペースを見ているのと同じでリークになる。
      レース単位に畳んでから累積すれば、過去は必ず別レースになる。

    df は [レース日付, レースID, 馬番] 順にソート済みなので、
    drop_duplicates(keep="first") でそのまま時系列順のレース一覧になる。
    """
    cols = config.resolve_columns(df)
    race = pd.DataFrame({"レースID": df[cols["race_id"]].to_numpy()})
    for c in value_cols:
        race[c] = df[c].to_numpy()
    return race.drop_duplicates(subset="レースID", keep="first").reset_index(drop=True)


def _standardize_by_condition(df: pd.DataFrame, value_col: str,
                              sign: float = 1.0, day_col: str | None = None) -> pd.Series:
    """(芝ダート × 距離帯) ごとに、**過去のレースだけ**を使って z 化する。

    全期間の平均・標準偏差で割ると未来の情報が混ざるため、
    leakfree の過去平均・過去標準偏差を使う。
    計算はレース単位（_race_frame）で行い、最後に馬の行へ配り直す。

    sign=-1 を渡すと符号を反転してから z 化する（上がり3Fは小さいほど良いため）。
    day_col を渡すと、同じ日のレースを「過去」に含めない（as_of="day"）。
    """
    cols = config.resolve_columns(df)

    aux: list[str] = [value_col] + ([day_col] if day_col else [])
    df = df.copy()
    if "surface" in cols:
        df["_surf_code_pace"] = leakfree.label_encode(df[cols["surface"]]).astype("int16")
        aux.append("_surf_code_pace")
    if "distance" in cols:
        df["_dist_bin_pace"] = (
            pd.to_numeric(df[cols["distance"]], errors="coerce") // 400
        ).astype("float32")
        aux.append("_dist_bin_pace")

    race = _race_frame(df, aux)
    race[value_col] = race[value_col].astype("float32") * sign
    keys = [c for c in aux if c not in (value_col, day_col)] or ["レースID"]

    if day_col:
        mean = leakfree.past_mean_ignore_nan_excluding(race, keys, race[value_col], day_col,
                                                       min_count=PACE_Z_MIN_COUNT)
        std = leakfree.past_std_ignore_nan_excluding(race, keys, race[value_col], day_col,
                                                     min_count=PACE_Z_MIN_COUNT)
    else:
        mean = leakfree.past_mean_ignore_nan(race, keys, race[value_col],
                                            min_count=PACE_Z_MIN_COUNT)
        std = leakfree.past_std_ignore_nan(race, keys, race[value_col],
                                          min_count=PACE_Z_MIN_COUNT)
    race["_z"] = ((race[value_col] - mean) / std.replace(0, np.nan)).astype("float32")

    # 馬の行へ配り直す
    mapped = pd.DataFrame({"レースID": df[cols["race_id"]].to_numpy()}).merge(
        race[["レースID", "_z"]], on="レースID", how="left"
    )
    return pd.Series(mapped["_z"].to_numpy(), index=df.index, dtype="float32")


def add_pace_features(df: pd.DataFrame, high_pace_threshold: float = 0.0,
                      as_of: str = "race") -> pd.DataFrame:
    """馬ごとのペース適性と、今日の想定ペースを追加する。

    attach_race_pace() を先に呼んでおくこと。
    脚質特徴量（corner.py）まで済んでいると想定ペースも作れる。
    as_of="day" なら、レース単位の統計（z 化の平均・標準偏差、想定ペース）に
    同じ日のレースを含めない。馬ごとの集計は1日1走なのでどちらでも同じ。
    """
    day_col = "_集計日" if as_of == "day" else None
    if "_ペース指標" not in df.columns:
        raise KeyError("先に attach_race_pace() を呼んでください")

    cols = config.resolve_columns(df)
    horse, rank = cols["horse"], cols["rank"]
    df = df.copy()

    # そのレースのペースを条件別に z 化（この列も「結果」なので特徴量にはしない）
    df["_ペース指標z"] = _standardize_by_condition(df, "_ペース指標", day_col=day_col)
    z = df["_ペース指標z"]

    show = (df[rank] <= 3).astype("float32")          # 複勝圏
    is_high = (z > high_pace_threshold).astype("float32")   # ハイペースだったレース
    is_slow = (z <= high_pace_threshold).astype("float32")  # スローペースだったレース

    # --- 馬のペース適性（過去のみ）------------------------------------------
    df["前走ペース指標"] = df.groupby(horse, observed=True, sort=False)["_ペース指標z"].shift(1)
    df["経験ペース平均"] = leakfree.past_mean_ignore_nan(df, [horse], z)
    df["好走時ペース平均"] = leakfree.past_conditional_mean(df, [horse], z, show)

    # ハイペース/スローペースそれぞれでの過去複勝率
    df["ハイペース時複勝率"] = leakfree.past_conditional_mean(df, [horse], show, is_high, min_count=2)
    df["スローペース時複勝率"] = leakfree.past_conditional_mean(df, [horse], show, is_slow, min_count=2)
    df["ペース巧拙差"] = (df["ハイペース時複勝率"] - df["スローペース時複勝率"]).astype("float32")

    # 上がり3Fの速さ（レース平均との差）。末脚の絶対的な速さを見る
    if "_上がり3F" in df.columns:
        # 上がり3Fは小さいほど速い＝良いので、符号を反転してから z 化する
        last3f_z = _standardize_by_condition(df, "_上がり3F", sign=-1.0, day_col=day_col)
        df["_上がり3Fz"] = last3f_z
        df["過去上がり3F偏差"] = leakfree.past_mean_ignore_nan(df, [horse], last3f_z)
        df["前走上がり3F偏差"] = df.groupby(horse, observed=True, sort=False)["_上がり3Fz"].shift(1)

    # --- 今日の想定ペース ----------------------------------------------------
    df = add_expected_pace(df, day_col=day_col)

    return df


def add_expected_pace(df: pd.DataFrame, day_col: str | None = None) -> pd.DataFrame:
    """出走馬の脚質構成から「今日の想定ペース」を推定する。

    仕組み:
      1. 前傾度（逃げ・先行馬の密度）を 0.05 刻みのビンに分ける
      2. **過去のレース**で、そのビンが実際に何 z のペースになったかの平均を取る
      3. その平均を今日の想定ペースとして使う

    ヒューリスティックな係数を手で決めるのではなく、
    「過去に同じような構成のレースがどうなったか」から学ぶ形にしている。
    過去平均しか見ないので、当然リークしない。
    """
    if "前傾度" not in df.columns or "_ペース指標z" not in df.columns:
        return df

    cols = config.resolve_columns(df)
    df["_前傾度bin"] = (df["前傾度"].fillna(-1) * 20).round().astype("float32")

    # 前傾度もペースもレース単位の値。馬の行のまま累積すると
    # 同じレースの他の馬の行が「過去」に混ざり、当日のペースを見てしまう。
    # 必ずレース単位に畳んでから過去平均を取る。
    race = _race_frame(df, ["_前傾度bin", "_ペース指標z"] + ([day_col] if day_col else []))
    if day_col:
        race["想定ペース"] = leakfree.past_mean_ignore_nan_excluding(
            race, ["_前傾度bin"], race["_ペース指標z"], day_col,
            min_count=EXPECTED_PACE_MIN_COUNT)
    else:
        race["想定ペース"] = leakfree.past_mean_ignore_nan(
            race, ["_前傾度bin"], race["_ペース指標z"], min_count=EXPECTED_PACE_MIN_COUNT
        )
    mapped = pd.DataFrame({"レースID": df[cols["race_id"]].to_numpy()}).merge(
        race[["レースID", "想定ペース"]], on="レースID", how="left"
    )
    df["想定ペース"] = pd.Series(mapped["想定ペース"].to_numpy(), index=df.index, dtype="float32")

    # 得意ペースと今日の想定ペースのズレ。0 に近いほど条件が向いている
    if "好走時ペース平均" in df.columns:
        df["想定ペース適性差"] = (df["想定ペース"] - df["好走時ペース平均"]).abs().astype("float32")
    if "前走ペース指標" in df.columns:
        df["前走比ペース変化"] = (df["想定ペース"] - df["前走ペース指標"]).astype("float32")

    # ハイペース想定 × ハイペース巧者 の相互作用
    if "ペース巧拙差" in df.columns:
        df["想定ペース×巧拙"] = (df["想定ペース"] * df["ペース巧拙差"]).astype("float32")

    return df


def pace_feature_columns() -> list[str]:
    """依頼Bで追加した特徴量の列名。"""
    return [
        "前走ペース指標", "経験ペース平均", "好走時ペース平均",
        "ハイペース時複勝率", "スローペース時複勝率", "ペース巧拙差",
        "過去上がり3F偏差", "前走上がり3F偏差",
        "想定ペース", "想定ペース適性差", "前走比ペース変化", "想定ペース×巧拙",
    ]
