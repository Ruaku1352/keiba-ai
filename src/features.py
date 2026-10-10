"""特徴量エンジニアリング（依頼1）。

■ 設計の大原則：Data Leakage を絶対に起こさない
   すべての「過去実績系」特徴量は、**その行自身を含まない**過去だけから作る。
   実装は次の 2 つの道具で統一している。

     過去の件数  : groupby.cumcount()                → 自分より前の行数
     過去の合計  : groupby.cumsum() - 自分の値        → 自分より前の合計

   cumsum() は「自分を含む累積和」なので、自分の値を引けば
   「自分より前だけの累積和」になる。これが漏れ防止の中核。

   直近N走は「累積和の差分」で取る:
       直近N走の合計 = (自分より前の累積和) - (N行前の時点での累積和)
   groupby.rolling は 100 万行超だと非常に遅いので使わない。

■ 前提
   preprocess.basic_clean() で
   [レース日付, レースID, 馬番] の順にソート済みであること。
   この順序が保証されていれば cumsum は必ず「過去 → 未来」の向きに走る。

■ 同日レースの扱い（as_of）
   as_of="race"（既定。v1〜v5 の validate と同じ）
       騎手・調教師の集計は「同じ開催日の、より前のレース」を過去として含む。
   as_of="day"（v6 の学習と予測で使う）
       騎手・調教師の集計、レース単位のペースの統計を「前日までに確定したレース」だけで作る。
       実戦では前日の夜に予測するので、当日の他のレースの結果はまだ無い。
       学習のときも同じ条件にしておかないと、学習と予測で特徴量の意味がずれる。
       馬の集計は1日1走なので、どちらでも同じ値になる。
   集計の「日」は `_集計日` 列で決める（無ければレース日付）。予測のときは、未確定の
   レースをすべて同じ日（最も早い予測対象日）にまとめて、互いに混ざらないようにする（predict.py）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, leakfree


# ---------------------------------------------------------------------------
# 低レベルのヘルパー（実体は leakfree.py。ここでは短い別名を張るだけ）
# ---------------------------------------------------------------------------
_past_count = leakfree.past_count
_past_sum = leakfree.past_sum
_past_rate = leakfree.past_rate
_past_window_mean = leakfree.past_window_mean
_label_encode = leakfree.label_encode


# ---------------------------------------------------------------------------
# 馬の実績系
# ---------------------------------------------------------------------------
def add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの過去実績特徴量を追加する。"""
    cols = config.resolve_columns(df)
    horse, rank, date = cols["horse"], cols["rank"], cols["date"]

    win = (df[rank] == 1).astype("float32")      # 1着
    quinella = (df[rank] <= 2).astype("float32")  # 連対（2着以内）
    show = (df[rank] <= 3).astype("float32")      # 複勝圏（3着以内）

    keys = [horse]

    # --- 通算成績（その時点まで）------------------------------------------
    df["通算出走回数"] = _past_count(df, keys).astype("int32")
    df["通算勝率"] = _past_rate(df, keys, win).astype("float32")
    df["通算連対率"] = _past_rate(df, keys, quinella).astype("float32")
    df["通算複勝率"] = _past_rate(df, keys, show).astype("float32")

    # --- 直近N走の平均着順 --------------------------------------------------
    rank_f = df[rank].astype("float32")
    for n in config.RECENT_WINDOWS:
        df[f"過去{n}走平均着順"] = _past_window_mean(df, keys, rank_f, n)

    # --- 前走系 -------------------------------------------------------------
    g_horse = df.groupby(horse, observed=True, sort=False)
    df["前走着順"] = g_horse[rank].shift(1).astype("float32")
    df["前走からの日数"] = (df[date] - g_horse[date].shift(1)).dt.days.astype("float32")

    if "distance" in cols:
        dist = cols["distance"]
        prev_dist = g_horse[dist].shift(1)
        df["前走との距離差"] = (df[dist] - prev_dist).astype("float32")

    if "last3f" in cols:
        df["前走上がり3F"] = g_horse[cols["last3f"]].shift(1).astype("float32")

    # --- 条件別の過去勝率 ---------------------------------------------------
    # 距離帯：±200m のスライド窓は groupby にできないので、200m 刻みのビンで代用。
    # （1600m の馬は 1500-1699m の馬とだけ比較される。厳密な ±200m ではない）
    if "distance" in cols:
        df["距離帯"] = (df[cols["distance"]] // 200).astype("float32")
        df["同距離帯_過去勝率"] = _past_rate(df, [horse, "距離帯"], win).astype("float32")
        df["同距離帯_過去出走数"] = _past_count(df, [horse, "距離帯"]).astype("int32")

    if "surface" in cols:
        code = _label_encode(df[cols["surface"]])
        df["_surface_code"] = code
        df["同芝ダ_過去勝率"] = _past_rate(df, [horse, "_surface_code"], win).astype("float32")
        df["同芝ダ_過去出走数"] = _past_count(df, [horse, "_surface_code"]).astype("int32")

    if "going" in cols:
        code = _label_encode(df[cols["going"]])
        df["_going_code"] = code
        df["同馬場状態_過去勝率"] = _past_rate(df, [horse, "_going_code"], win).astype("float32")
        df["同馬場状態_過去出走数"] = _past_count(df, [horse, "_going_code"]).astype("int32")

    if "course" in cols:
        code = _label_encode(df[cols["course"]])
        df["_course_code"] = code
        df["同競馬場_過去勝率"] = _past_rate(df, [horse, "_course_code"], win).astype("float32")

    return df


# ---------------------------------------------------------------------------
# 騎手・調教師系
# ---------------------------------------------------------------------------
AS_OF_CHOICES = ("race", "day")
DAY_COL = "_集計日"


def _exclusion_col(df: pd.DataFrame, as_of: str) -> str:
    """同じ行を「過去」に含めない単位の列名。race ならレースID、day なら _集計日。"""
    if as_of not in AS_OF_CHOICES:
        raise ValueError(f"as_of は {AS_OF_CHOICES} のどれか: {as_of!r}")
    if as_of == "day":
        return DAY_COL
    return config.resolve_columns(df)["race_id"]


def add_jockey_features(df: pd.DataFrame, as_of: str = "race") -> pd.DataFrame:
    """騎手系の特徴量を追加する。

    as_of="race": 同じレースの行だけを除く（騎手は1レース1頭なので、実質は自分だけ）。
                  騎手×競馬場・騎手×芝ダ・直近100走は v1 からの実装のまま。
    as_of="day" : 同じ日の行をすべて除く（前日までの成績）。
    """
    cols = config.resolve_columns(df)
    if "jockey" not in cols:
        return df
    jockey, rank = cols["jockey"], cols["rank"]

    win = (df[rank] == 1).astype("float32")
    df["_jockey_code"] = _label_encode(df[jockey])
    excl = _exclusion_col(df, as_of)
    window = config.JOCKEY_RECENT_WINDOW

    df["騎手通算勝率"] = leakfree.past_rate_excluding_race(
        df, ["_jockey_code"], win, excl).astype("float32")
    df["騎手通算騎乗数"] = leakfree.past_count_excluding_race(
        df, ["_jockey_code"], excl).astype("int32")

    if as_of == "day":
        df[f"騎手直近{window}走勝率"] = leakfree.past_window_mean_excluding(
            df, ["_jockey_code"], win, window, excl)
    else:
        df[f"騎手直近{window}走勝率"] = _past_window_mean(df, ["_jockey_code"], win, window)

    for code_col, name in [("_course_code", "騎手×競馬場_勝率"), ("_surface_code", "騎手×芝ダ_勝率")]:
        if code_col not in df.columns:
            continue
        keys = ["_jockey_code", code_col]
        if as_of == "day":
            df[name] = leakfree.past_rate_excluding_race(df, keys, win, excl, min_count=10)
        else:
            df[name] = _past_rate(df, keys, win, min_count=10)
        df[name] = df[name].astype("float32")

    return df


def add_trainer_features(df: pd.DataFrame, as_of: str = "race") -> pd.DataFrame:
    """調教師系の特徴量を追加する。

    調教師は1レースに複数頭を出すので、少なくとも同じレースの同厩馬は「過去」に
    含めない（v5で修正。v1〜v4では漏れていた）。as_of="day" なら同じ日の行をすべて除く。
    """
    cols = config.resolve_columns(df)
    if "trainer" not in cols:
        return df
    trainer, rank = cols["trainer"], cols["rank"]

    win = (df[rank] == 1).astype("float32")
    df["_trainer_code"] = _label_encode(df[trainer])
    excl = _exclusion_col(df, as_of)

    df["調教師通算勝率"] = leakfree.past_rate_excluding_race(
        df, ["_trainer_code"], win, excl).astype("float32")
    df["調教師通算出走数"] = leakfree.past_count_excluding_race(
        df, ["_trainer_code"], excl).astype("int32")
    return df


# ---------------------------------------------------------------------------
# 馬 × 騎手
# ---------------------------------------------------------------------------
def add_horse_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """コンビ回数・コンビ勝率・乗り替わりフラグ。"""
    cols = config.resolve_columns(df)
    if "jockey" not in cols:
        return df
    horse, jockey, rank = cols["horse"], cols["jockey"], cols["rank"]

    win = (df[rank] == 1).astype("float32")
    if "_jockey_code" not in df.columns:
        df["_jockey_code"] = _label_encode(df[jockey])

    keys = [horse, "_jockey_code"]
    df["コンビ回数"] = _past_count(df, keys).astype("int32")
    df["コンビ勝率"] = _past_rate(df, keys, win).astype("float32")

    # 乗り替わり：前走と騎手が違えば 1。初出走（前走なし）は NaN。
    prev_jockey = df.groupby(horse, observed=True, sort=False)["_jockey_code"].shift(1)
    df["乗り替わりフラグ"] = np.where(
        prev_jockey.isna(), np.nan, (df["_jockey_code"] != prev_jockey).astype("float32")
    ).astype("float32")

    return df


# ---------------------------------------------------------------------------
# レース単位の特徴量（当日情報のみを使うので漏れではない）
# ---------------------------------------------------------------------------
def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """出走頭数など、レース内で完結する特徴量。"""
    cols = config.resolve_columns(df)
    race_id = cols["race_id"]
    g_race = df.groupby(race_id, observed=True, sort=False)

    df["出走頭数"] = g_race[cols["post"] if "post" in cols else race_id].transform("size").astype("int16")

    # レース内での相対値（当日出走馬同士の比較。結果は使っていないので安全）
    if "通算勝率" in df.columns:
        df["通算勝率_レース内順位"] = g_race["通算勝率"].rank(
            ascending=False, method="min"
        ).astype("float32")
    if "騎手通算勝率" in df.columns:
        df["騎手勝率_レース内順位"] = g_race["騎手通算勝率"].rank(
            ascending=False, method="min"
        ).astype("float32")
    if "weight_carried" in cols:
        df["斤量_レース内平均差"] = (
            df[cols["weight_carried"]] - g_race[cols["weight_carried"]].transform("mean")
        ).astype("float32")

    return df


# ---------------------------------------------------------------------------
# まとめて実行
# ---------------------------------------------------------------------------
def add_all_features(df: pd.DataFrame, strict_daily_lag: bool = False,
                     drop_helper_cols: bool = True,
                     corner_df: pd.DataFrame | None = None,
                     lap_df: pd.DataFrame | None = None,
                     as_of: str = "race") -> pd.DataFrame:
    """すべての特徴量を追加する（この関数だけ呼べばよい）。

    Parameters
    ----------
    as_of :
        "race"（既定・validate と同じ）か "day"（前日までに確定したレースだけで集計）。
        モジュール冒頭の説明を参照。
    strict_daily_lag :
        True なら as_of="day" と同じ（v1 からの互換のために残してある）。
    drop_helper_cols :
        True なら内部用の作業列（`_` で始まる列）を最後に削除する。
    corner_df :
        corner.load_corner() の戻り値。渡すと脚質・展開特徴量（依頼A）が付く。
    lap_df :
        pace.load_laptime() の戻り値。渡すとペース特徴量（依頼B）が付く。
        脚質特徴量が先に付いていると「想定ペース」も作れる。
    """
    from . import corner as corner_mod  # 循環 import を避けるため関数内で読む
    from . import pace as pace_mod

    if strict_daily_lag:
        as_of = "day"
    df = df.copy()
    if as_of == "day" and DAY_COL not in df.columns:
        df[DAY_COL] = df[config.resolve_columns(df)["date"]]

    df = add_horse_features(df)
    df = add_jockey_features(df, as_of=as_of)
    df = add_trainer_features(df, as_of=as_of)
    df = add_horse_jockey_features(df)
    df = add_race_features(df)

    # 依頼A：脚質・展開（ペースの想定に使うので先に計算する）
    if corner_df is not None:
        # Kaggle：通過順の文字列をパースした表を結合する
        df = corner_mod.attach_corner_position(df, corner_df)
        df = corner_mod.add_running_style_features(df)
    elif "_4コーナー順位" in df.columns:
        # JRA-VAN：SE に数値で入っている4コーナー順位をそのまま使う（jvmap が作る列）
        df = corner_mod.attach_corner_from_columns(df)
        df = corner_mod.add_running_style_features(df)

    # JRA-VAN の公式脚質判定（過去走だけで集計）
    if "_脚質判定" in df.columns:
        df = corner_mod.add_official_style_features(df)

    # 依頼B：ペース
    if lap_df is not None:
        df = pace_mod.attach_race_pace(df, lap_df)
        df = pace_mod.add_pace_features(df, as_of=as_of)

    if drop_helper_cols:
        # `_` で始まる列は「そのレースの結果」や中間計算なので、
        # 特徴量に紛れ込まないようここで一括削除する。
        helper = [c for c in df.columns if c.startswith("_")]
        df = df.drop(columns=helper)

    return df


def feature_columns(df: pd.DataFrame, include_odds: bool = False) -> list[str]:
    """学習に使う特徴量の列名リストを返す。

    include_odds=False（既定）なら単勝オッズ・人気を除外する。
    市場（オッズ）と独立した予測を作るのが目的なので、通常は除外して学習し、
    オッズは「期待値の計算時」にだけ使う。
    """
    cols = config.resolve_columns(df)

    base_logical = ["bracket", "post", "age", "weight_carried", "horse_weight",
                    "horse_weight_diff", "distance", "course", "surface", "turn", "going"]
    if include_odds:
        base_logical += ["win_odds", "popularity"]

    engineered = [
        "通算出走回数", "通算勝率", "通算連対率", "通算複勝率",
        "前走着順", "前走からの日数", "前走との距離差", "前走上がり3F",
        "同距離帯_過去勝率", "同距離帯_過去出走数",
        "同芝ダ_過去勝率", "同芝ダ_過去出走数",
        "同馬場状態_過去勝率", "同馬場状態_過去出走数",
        "同競馬場_過去勝率",
        "騎手通算勝率", "騎手通算騎乗数", f"騎手直近{config.JOCKEY_RECENT_WINDOW}走勝率",
        "騎手×競馬場_勝率", "騎手×芝ダ_勝率",
        "調教師通算勝率", "調教師通算出走数",
        "コンビ回数", "コンビ勝率", "乗り替わりフラグ",
        "出走頭数", "通算勝率_レース内順位", "騎手勝率_レース内順位", "斤量_レース内平均差",
    ]
    engineered += [f"過去{n}走平均着順" for n in config.RECENT_WINDOWS]

    # 依頼A・Bの特徴量（データを渡していれば列が存在する）
    from . import corner as corner_mod
    from . import pace as pace_mod
    engineered += corner_mod.running_style_feature_columns()
    engineered += corner_mod.official_style_feature_columns()
    engineered += pace_mod.pace_feature_columns()

    result = [cols[k] for k in base_logical if k in cols]
    result += [c for c in engineered if c in df.columns]
    # 重複を除きつつ順序を維持
    seen: set[str] = set()
    return [c for c in result if not (c in seen or seen.add(c))]
