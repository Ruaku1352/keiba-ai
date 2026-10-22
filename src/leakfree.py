"""リークを起こさない「過去だけの集計」を作る道具箱。

このプロジェクトの全ての過去実績系特徴量は、ここにある関数だけで作る。
道具を1箇所に集めておけば「どこかで未来を見ていないか」を目視で追える。

■ 中核となる考え方

    groupby.cumcount()                  → 自分より前の行数
    groupby.shift(1) してから cumsum()   → 自分より前の合計

  「自分より前の値だけ」を足し上げる。v7 までは「自分を含む累積和 − 自分の値」で
  求めていたが、浮動小数では (前の合計 + 自分) − 自分 が前の合計とぴったり同じにならず、
  丸めの向きが**自分の値（＝そのレースの結果）**で変わっていた。値そのものの差は 1e-8 程度だが、
  脚質カテゴリのように境界（0.10 など）で切る特徴量では、ちょうど境界にいる馬のカテゴリが
  自分の結果で変わってしまう（学習の経路と予測の経路で特徴量がずれた原因。v7 で修正）。
  今は引き算をせず、前の行だけを足すので、結果の列を消しても同じ値になる。

■ 前提
  DataFrame が [レース日付, レースID, 馬番] 順にソート済みであること
  （preprocess.basic_clean が保証する）。この順序があるので、
  cumsum は必ず「過去 → 未来」の向きに走る。

■ 欠損の扱い
  past_mean_ignore_nan / past_std_ignore_nan は
  「値がある行だけ」を母数にする。コーナー通過順のように
  一部の年しかデータがない特徴量では、これを使わないと
  欠損を0とみなして平均が歪む。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _grouper(df: pd.DataFrame, keys: list[str]) -> list[pd.Series]:
    """groupby に渡すキー列のリスト（Series で渡すと index が揃う）。"""
    return [df[k] for k in keys]


def past_count(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """自分より前に、そのキーで何行あったか。"""
    return df.groupby(keys, observed=True, sort=False).cumcount()


def _exclusive_cumsum(value: pd.Series, grouper) -> pd.Series:
    """グループごとの「自分より前の行だけ」の累積和。NaN は 0。引き算をしないので、
    自分の値が何であっても結果はビット単位で同じ（自分の値に依存しない）。"""
    tmp = value.fillna(0)
    prev = tmp.groupby(grouper, observed=True, sort=False).shift(1, fill_value=0)
    return prev.groupby(grouper, observed=True, sort=False).cumsum()


def past_sum(df: pd.DataFrame, keys: list[str], value: pd.Series) -> pd.Series:
    """自分より前の合計（自分自身は含まない）。NaN は 0 として扱う。"""
    return _exclusive_cumsum(value, _grouper(df, keys))


def past_rate(df: pd.DataFrame, keys: list[str], value: pd.Series,
              min_count: int = 1) -> pd.Series:
    """自分より前の平均（勝率など）。母数が min_count 未満なら NaN。

    value が 0/1 のフラグであることを前提にした「率」用。
    欠損がある連続値には past_mean_ignore_nan を使うこと。
    """
    cnt = past_count(df, keys)
    tot = past_sum(df, keys, value)
    rate = tot / cnt.replace(0, np.nan)
    return rate.where(cnt >= min_count)


def past_window_mean(df: pd.DataFrame, keys: list[str], value: pd.Series,
                     window: int) -> pd.Series:
    """直近 window 行の平均（自分自身は含まない）。

    累積和の差分で計算するので O(n)。
        直近N走の合計 = (自分より前の累積和) - (N行前時点の「その行より前の」累積和)
    どちらの累積和も自分より前の行だけで決まるので、自分の値には依存しない。
    groupby.rolling は162万行だと現実的な時間で終わらないため使わない。
    """
    g = _grouper(df, keys)
    past_cum = _exclusive_cumsum(value, g)
    lagged = past_cum.groupby(g, observed=True, sort=False).shift(window).fillna(0)

    cnt = past_count(df, keys)
    denom = np.minimum(cnt, window)  # 出走数が window に満たない馬に対応
    return ((past_cum - lagged) / denom.replace(0, np.nan)).astype("float32")


def past_mean_ignore_nan(df: pd.DataFrame, keys: list[str], value: pd.Series,
                         min_count: int = 1) -> pd.Series:
    """自分より前の平均。**値が入っている行だけ**を母数にする。

    仕組み: 合計は NaN を 0 とみなして足し、母数は「値があった回数」を数える。
    こうすると欠損レース（コーナー情報のない年など）が平均を薄めない。
    """
    valid = value.notna().astype("float32")
    total = past_sum(df, keys, value)
    n = past_sum(df, keys, valid)
    mean = total / n.replace(0, np.nan)
    return mean.where(n >= min_count).astype("float32")


def past_std_ignore_nan(df: pd.DataFrame, keys: list[str], value: pd.Series,
                        min_count: int = 2) -> pd.Series:
    """自分より前の標準偏差（母集団標準偏差）。値がある行だけが母数。

    Var(X) = E[X^2] - E[X]^2 を使うので、2回の累積和だけで求まる。
    浮動小数の丸めで極小の負値が出ることがあるため 0 で下限を切っている。
    """
    valid = value.notna().astype("float32")
    n = past_sum(df, keys, valid)
    s1 = past_sum(df, keys, value)
    s2 = past_sum(df, keys, value.astype("float64") ** 2)

    denom = n.replace(0, np.nan)
    var = s2 / denom - (s1 / denom) ** 2
    var = var.clip(lower=0)
    return np.sqrt(var).where(n >= min_count).astype("float32")


def past_conditional_mean(df: pd.DataFrame, keys: list[str], value: pd.Series,
                          condition: pd.Series, min_count: int = 1) -> pd.Series:
    """条件を満たした過去レースだけでの平均。

    例）「複勝圏に入ったレースの平均ペース指標」＝ハイペース巧者かどうか。
    合計も母数も condition を掛けてから累積するだけ。
    """
    cond = condition.fillna(0).astype("float32")
    valid = (value.notna().astype("float32") * cond)
    total = past_sum(df, keys, value.fillna(0) * cond)
    n = past_sum(df, keys, valid)
    mean = total / n.replace(0, np.nan)
    return mean.where(n >= min_count).astype("float32")


def past_count_excluding_race(df: pd.DataFrame, keys: list[str], race_col: str) -> pd.Series:
    """自分より前の行数。ただし**同じレースの行は自分より前でも数えない**。

    なぜ必要か:
      調教師は1レースに複数頭を出すことがある。cumcount() のままだと、
      同じレースで先に並んでいる同厩馬の行が「過去」として数えられてしまう。
      それはそのレースの結果を見ているのと同じでリークになる
      （ペースで _race_frame を使ったのと同じ種類の問題）。

    仕組み:
      (キー単位の累積) - (キー×レース単位の累積) = 前のレースまでの分
      行が「日付→レースID」順に並んでいれば、同じレースの行はキーの中で連続するので、
      引き算で「今のレースの分」だけがきれいに消える。
    """
    whole = df.groupby(keys, observed=True, sort=False).cumcount()
    same_race = df.groupby(keys + [race_col], observed=True, sort=False).cumcount()
    return whole - same_race


def past_sum_excluding_race(df: pd.DataFrame, keys: list[str], value: pd.Series,
                            race_col: str) -> pd.Series:
    """自分より前の合計。同じレースの行は含めない（past_count_excluding_race と同じ考え方）。

    「そのキーで、同じレース（race_col）の最初の行より前の合計」を、同じレースの全行に配る。
    行が「日付→レース」順に並んでいれば同じレースの行はキーの中で連続するので、
    最初の行の「自分より前の合計」= 前のレースまでの合計。引き算をしないので、
    同じレースの行（自分を含む）の値には依存しない。
    """
    excl = _exclusive_cumsum(value, _grouper(df, keys))
    return excl.groupby(_grouper(df, keys + [race_col]), observed=True, sort=False).transform("first")


def past_rate_excluding_race(df: pd.DataFrame, keys: list[str], value: pd.Series,
                             race_col: str, min_count: int = 1) -> pd.Series:
    """同じレースを除いた過去の率。1レースに同じキーが複数行ありうるキー（調教師など）用。"""
    cnt = past_count_excluding_race(df, keys, race_col)
    tot = past_sum_excluding_race(df, keys, value, race_col)
    rate = tot / cnt.replace(0, np.nan)
    return rate.where(cnt >= min_count)


def past_mean_ignore_nan_excluding(df: pd.DataFrame, keys: list[str], value: pd.Series,
                                   group_col: str, min_count: int = 1) -> pd.Series:
    """past_mean_ignore_nan の「同じ group_col の行を除く」版。

    group_col に日付を渡せば「前日までの平均」になる（as_of="day" で使う）。
    """
    valid = value.notna().astype("float32")
    total = past_sum_excluding_race(df, keys, value, group_col)
    n = past_sum_excluding_race(df, keys, valid, group_col)
    mean = total / n.replace(0, np.nan)
    return mean.where(n >= min_count).astype("float32")


def past_std_ignore_nan_excluding(df: pd.DataFrame, keys: list[str], value: pd.Series,
                                  group_col: str, min_count: int = 2) -> pd.Series:
    """past_std_ignore_nan の「同じ group_col の行を除く」版。"""
    valid = value.notna().astype("float32")
    n = past_sum_excluding_race(df, keys, valid, group_col)
    s1 = past_sum_excluding_race(df, keys, value, group_col)
    s2 = past_sum_excluding_race(df, keys, value.astype("float64") ** 2, group_col)
    denom = n.replace(0, np.nan)
    var = (s2 / denom - (s1 / denom) ** 2).clip(lower=0)
    return np.sqrt(var).where(n >= min_count).astype("float32")


def past_window_mean_excluding(df: pd.DataFrame, keys: list[str], value: pd.Series,
                               window: int, group_col: str) -> pd.Series:
    """「同じ group_col の行より前」の直近 window 行の平均。

    例）騎手の直近100走勝率を「前日まで」で出す。同じ日に何鞍も乗るので、
        自分より前の100行ではなく「その日の最初の騎乗より前の100行」を数える。

    仕組み（キーごとに並べた配列の上で考える）:
      P[j] = そのキーの先頭から j の手前までの合計（自分を含まない累積和）
      f    = 自分と同じ日の最初の行の位置
      窓   = [max(f - window, キーの先頭), f)
      合計 = P[f] - P[窓の始まり]、件数 = f - 窓の始まり
    行が「日付→レース」順に並んでいれば、同じ日の行はキーの中で連続するので f が決まる。
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype="float32", index=df.index)
    key_id = df.groupby(keys, observed=True, sort=False).ngroup().to_numpy()
    grp_id = df.groupby(keys + [group_col], observed=True, sort=False).ngroup().to_numpy()
    v = value.fillna(0).to_numpy(dtype="float64")

    order = np.argsort(key_id, kind="stable")          # キーごとに、元の（時系列の）順で並べる
    k, g, vs = key_id[order], grp_id[order], v[order]
    idx = np.arange(n)

    new_key = np.r_[True, k[1:] != k[:-1]]
    key_start = np.maximum.accumulate(np.where(new_key, idx, 0))
    new_grp = new_key | np.r_[True, g[1:] != g[:-1]]
    first_in_grp = np.maximum.accumulate(np.where(new_grp, idx, 0))

    # P[j] = そのキーの先頭から j の手前までの合計。キーごとに前の行だけを足す（引き算をしない）
    vs_s, k_s = pd.Series(vs), pd.Series(k)
    prefix = vs_s.groupby(k_s, sort=False).shift(1, fill_value=0.0) \
        .groupby(k_s, sort=False).cumsum().to_numpy()

    lo = np.maximum(first_in_grp - window, key_start)
    total = prefix[first_in_grp] - prefix[lo]
    count = (first_in_grp - lo).astype("float64")
    mean_sorted = np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)

    out = np.empty(n)
    out[order] = mean_sorted
    return pd.Series(out, index=df.index, dtype="float32")


def label_encode(s: pd.Series) -> pd.Series:
    """groupby キー用に文字列/カテゴリを整数コード化する（速度・メモリ対策）。"""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes
    return s.astype("category").cat.codes


def normalize_id(s: pd.Series) -> pd.Series:
    """レースIDを文字列に正規化する。

    CSV ごとに int で読まれたり str で読まれたりするため、
    結合前に必ずこれを通して型を揃える。float で読まれた場合の
    "198601010101.0" も int を経由して潰す。
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int64").astype(str)
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
