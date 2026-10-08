"""依頼A：コーナー通過順から脚質・展開の特徴量を作る。

corner_passing_order.csv は JRA 標準表記の文字列が入っている。

    (*13,6)4,10-2,9,12,3(7,11)1=8,5

  数字     : 馬番
  (a,b)    : 括弧内は横並び（同じ位置の集団）
  ,        : わずかな差
  -        : やや離れた差（約1馬身以上）
  =        : 大きく離れた差（約10馬身以上）
  *        : ハナ（先頭）
  連続空白  : 大きく離れた馬

■ パース方針
  記号の「差の大きさ」は捨てて、**左から順に順位を振る**だけにした。
  括弧内は同着扱い（競技順位方式：(a,b) なら両方1位、次は3位）。
  差の大小まで数値化しても、下流の「相対位置の平均」ではほぼ効かないため、
  まず単純で壊れにくい実装を選んでいる。

■ リーク対策
  そのレース自身の通過順は**絶対に特徴量にしない**。
  馬ごとの脚質は leakfree の過去集計だけから作る。
  一方、レース内の展開特徴量（逃げ馬頭数など）は
  「過去実績から作った脚質スコア」を集計したものなので、
  当日の結果を一切使っていない = 安全。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config, leakfree

# 「(」「)」「数字」だけを順に拾うトークナイザ
_TOKEN = re.compile(r"[()]|\d+")

CORNER_COLUMNS = ["1コーナー", "2コーナー", "3コーナー", "4コーナー"]

# 脚質カテゴリの境界（4コーナーの相対位置。0=先頭, 1=最後方）
# 逃げ: 先頭付近 / 先行: 前1/3 / 差し: 中団 / 追込: 後方
KYAKUSHITSU_BINS = (0.0, 0.10, 0.35, 0.70, 1.01)
KYAKUSHITSU_LABELS = ("逃げ", "先行", "差し", "追込")


def parse_passing_order(text: object) -> dict[int, int]:
    """通過順の文字列を {馬番: 順位} の辞書に変換する。

    >>> parse_passing_order("(*13,6)4,10-2,9")
    {13: 1, 6: 1, 4: 3, 10: 4, 2: 5, 9: 6}

    括弧内は同順位。次の順位は「それまでの頭数 + 1」になる（競技順位方式）。
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    result: dict[int, int] = {}
    placed = 0          # ここまでに順位を振った頭数
    group: list[int] = []  # 括弧の中に溜めている馬番
    in_group = False

    def flush(members: list[int]) -> None:
        """溜めた馬番に同じ順位を与える。"""
        nonlocal placed
        if not members:
            return
        rank = placed + 1
        for horse_no in members:
            result.setdefault(horse_no, rank)
        placed += len(members)

    for token in _TOKEN.findall(text):
        if token == "(":
            in_group = True
            group = []
        elif token == ")":
            flush(group)
            in_group = False
            group = []
        else:
            number = int(token)
            if in_group:
                group.append(number)
            else:
                flush([number])

    flush(group)  # 閉じ括弧が欠けている壊れた行への保険
    return result


def load_corner(path: str | None = None, corners: list[str] | None = None) -> pd.DataFrame:
    """corner_passing_order.csv を読んで縦持ち（1行=1馬1コーナー）に変換する。

    戻り値の列: レースID, corner, 馬番, 通過順位

    corners を指定しない場合は 4コーナーのみ。
    1・2コーナーは短距離戦で欠損が多いため、既定では使わない。
    """
    path = path or config.CORNER_CSV
    corners = corners or ["4コーナー"]

    raw = pd.read_csv(path, low_memory=False)
    id_col = "レースID" if "レースID" in raw.columns else raw.columns[0]

    records = []
    for corner in corners:
        if corner not in raw.columns:
            continue
        sub = raw.loc[raw[corner].notna(), [id_col, corner]]
        for race_id, text in zip(sub[id_col].to_numpy(), sub[corner].to_numpy()):
            for horse_no, rank in parse_passing_order(text).items():
                records.append((race_id, corner, horse_no, rank))

    out = pd.DataFrame(records, columns=["レースID", "corner", "馬番", "通過順位"])
    if not out.empty:
        out["レースID"] = leakfree.normalize_id(out["レースID"])
        out["馬番"] = out["馬番"].astype("int16")
        out["通過順位"] = out["通過順位"].astype("int16")
    return out


def attach_corner_position(df: pd.DataFrame, corner_df: pd.DataFrame,
                           corner: str = "4コーナー") -> pd.DataFrame:
    """race_result に「そのレースでの4角相対位置」を貼り付ける。

    相対位置 = (通過順位 - 1) / (出走頭数 - 1)
        0.0 = 先頭、1.0 = 最後方。頭数の違いを吸収するための正規化。

    ⚠ この列は**そのレースの結果**なので、そのまま特徴量にしてはいけない。
       add_running_style_features() が過去集計にだけ使う。
    """
    cols = config.resolve_columns(df)
    race_id, post = cols["race_id"], cols["post"]

    sub = corner_df.loc[corner_df["corner"] == corner, ["レースID", "馬番", "通過順位"]]

    key = pd.DataFrame({
        "レースID": leakfree.normalize_id(df[race_id]),
        "馬番": pd.to_numeric(df[post], errors="coerce").astype("Int16"),
    })
    merged = key.merge(sub, on=["レースID", "馬番"], how="left")

    df = df.copy()
    rank_in_corner = pd.Series(merged["通過順位"].to_numpy(), index=df.index, dtype="float32")

    # 出走頭数は race_result 側から数える（corner 側は欠損があるため）
    n_runners = df.groupby(race_id, observed=True, sort=False)[post].transform("size")
    denom = (n_runners - 1).replace(0, np.nan)
    df["_4角相対位置"] = ((rank_in_corner - 1) / denom).astype("float32")
    df["_4角通過順位"] = rank_in_corner
    return df


def attach_corner_from_columns(df: pd.DataFrame, rank_col: str = "_4コーナー順位",
                               field_size_col: str = "公式出走頭数") -> pd.DataFrame:
    """JRA-VAN 用：SE の4コーナー順位（数値）から4角相対位置を作る。

    Kaggle 版は通過順の文字列をパースしていたが、JRA-VAN は馬ごとの順位が
    数値で直接入っているのでパースは不要。ここから先（相対位置 → 脚質スコア）は
    attach_corner_position() と同じ列を作るので、下流のロジックはそのまま使える。

    分母の出走頭数は RA の公式値を優先する。race_result 側の行数で数えると、
    競走中止で除いた馬のぶん頭数が減り、相対位置が1を超えることがあるため。
    """
    cols = config.resolve_columns(df)
    race_id, post = cols["race_id"], cols["post"]

    df = df.copy()
    rank_in_corner = pd.to_numeric(df[rank_col], errors="coerce").astype("float32")
    rank_in_corner = rank_in_corner.where(rank_in_corner > 0)  # 00 は「通過していない」

    if field_size_col in df.columns:
        n_runners = pd.to_numeric(df[field_size_col], errors="coerce")
    else:
        n_runners = df.groupby(race_id, observed=True, sort=False)[post].transform("size")
    denom = (n_runners - 1).replace(0, np.nan)
    df["_4角相対位置"] = ((rank_in_corner - 1) / denom).clip(0, 1).astype("float32")
    df["_4角通過順位"] = rank_in_corner
    return df


def add_official_style_features(df: pd.DataFrame, col: str = "_脚質判定") -> pd.DataFrame:
    """JRA-VAN の公式脚質判定（1:逃 2:先 3:差 4:追）を、過去走だけで集計する。

    ⚠ 公式脚質判定は「今回レース」の判定なので、**そのレース自身の値はリーク**。
       自前の脚質スコアと同じく、leakfree の過去集計だけから特徴量を作る。

    自前の脚質スコア（4角相対位置の過去平均）と並べて学習し、
    どちらが効くかは gain 重要度で比べる（validate.style_importance_table）。
    """
    cols = config.resolve_columns(df)
    horse, race_id = cols["horse"], cols["race_id"]
    df = df.copy()

    kubun = pd.to_numeric(df[col], errors="coerce")
    kubun = kubun.where(kubun.between(1, 4)).astype("float32")  # 0 は初期値

    df["公式脚質_過去平均"] = leakfree.past_mean_ignore_nan(df, [horse], kubun)
    # 逃げた率・追い込んだ率（値がある過去走のうち）
    df["公式脚質_逃げ率"] = leakfree.past_mean_ignore_nan(
        df, [horse], (kubun == 1).astype("float32").where(kubun.notna()))
    df["公式脚質_追込率"] = leakfree.past_mean_ignore_nan(
        df, [horse], (kubun == 4).astype("float32").where(kubun.notna()))
    df["公式脚質_前走"] = df.groupby(horse, observed=True, sort=False)[col].shift(1)
    df["公式脚質_前走"] = pd.to_numeric(df["公式脚質_前走"], errors="coerce").where(
        lambda s: s.between(1, 4)).astype("float32")

    # レース内での相対値（自前の「脚質スコア_レース内平均差」に対応する比較対象）
    df["公式脚質_レース内平均差"] = (
        df["公式脚質_過去平均"]
        - df.groupby(race_id, observed=True, sort=False)["公式脚質_過去平均"].transform("mean")
    ).astype("float32")
    return df


def official_style_feature_columns() -> list[str]:
    """公式脚質判定から作った特徴量の列名（JRA-VAN データのときだけ存在する）。"""
    return ["公式脚質_過去平均", "公式脚質_逃げ率", "公式脚質_追込率",
            "公式脚質_前走", "公式脚質_レース内平均差"]


def add_running_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの脚質（過去実績から）とレース内の展開特徴量を追加する。

    attach_corner_position() を先に呼んでおくこと。
    """
    cols = config.resolve_columns(df)
    horse, rank = cols["horse"], cols["rank"]

    if "_4角相対位置" not in df.columns:
        raise KeyError("先に attach_corner_position() を呼んでください")

    df = df.copy()
    pos = df["_4角相対位置"]

    # --- 馬の脚質（過去だけ。値がある過去レースのみを母数にする）-----------
    df["脚質スコア"] = leakfree.past_mean_ignore_nan(df, [horse], pos)
    df["脚質安定度"] = leakfree.past_std_ignore_nan(df, [horse], pos)
    df["脚質サンプル数"] = leakfree.past_sum(df, [horse], pos.notna().astype("float32")).astype("int16")

    # 直近3走の脚質（近走で戦法を変えている馬を捉える）
    df["脚質スコア直近3走"] = _recent_mean_ignore_nan(df, [horse], pos, 3)

    # 前走の位置取り
    df["前走4角相対位置"] = df.groupby(horse, observed=True, sort=False)["_4角相対位置"].shift(1)

    # 好走時（3着以内）の位置取り。「前で残すタイプか、差して来るタイプか」
    show = (df[rank] <= 3).astype("float32")
    df["好走時脚質スコア"] = leakfree.past_conditional_mean(df, [horse], pos, show)

    # --- 脚質カテゴリ（離散化）--------------------------------------------
    df["脚質カテゴリ"] = pd.cut(
        df["脚質スコア"], bins=list(KYAKUSHITSU_BINS),
        labels=list(KYAKUSHITSU_LABELS), right=False,
    )

    df = add_pace_scenario_features(df)
    return df


def _recent_mean_ignore_nan(df: pd.DataFrame, keys: list[str], value: pd.Series,
                            window: int) -> pd.Series:
    """直近 window 走のうち「値がある回」だけの平均。

    past_window_mean は欠損を0扱いするので、コーナー情報のように
    欠損レースがある列にはこちらを使う。合計・母数それぞれを
    窓で取って割るだけ。
    """
    total = leakfree.past_window_mean(df, keys, value.fillna(0), window)
    valid = leakfree.past_window_mean(df, keys, value.notna().astype("float32"), window)
    return (total / valid.replace(0, np.nan)).astype("float32")


def add_pace_scenario_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内の展開特徴量。

    材料は「過去実績から作った脚質スコア」だけなので、
    当日の結果は一切使っていない（＝リークしない）。
    """
    cols = config.resolve_columns(df)
    race_id = cols["race_id"]
    g = df.groupby(race_id, observed=True, sort=False)

    cat = df["脚質カテゴリ"].astype(object)
    is_nige = (cat == "逃げ").astype("float32")
    is_senko = (cat == "先行").astype("float32")

    df["レース内逃げ馬頭数"] = is_nige.groupby(df[race_id], sort=False).transform("sum").astype("float32")
    df["レース内先行馬頭数"] = is_senko.groupby(df[race_id], sort=False).transform("sum").astype("float32")

    # 自分がレース内で何番目に前を取りたい馬か（1 = 最も前）
    df["脚質スコア_レース内順位"] = g["脚質スコア"].rank(method="min").astype("float32")
    df["脚質スコア_レース内平均差"] = (
        df["脚質スコア"] - g["脚質スコア"].transform("mean")
    ).astype("float32")

    # 単騎逃げ：レース内の逃げ馬が自分1頭だけなら 1
    df["単騎逃げフラグ"] = ((df["レース内逃げ馬頭数"] == 1) & (is_nige == 1)).astype("int8")

    # 前に行きたい馬の密度。高いほどハイペースになりやすい
    n_runners = g[cols["post"]].transform("size").astype("float32")
    df["前傾度"] = (
        (df["レース内逃げ馬頭数"] + 0.5 * df["レース内先行馬頭数"]) / n_runners
    ).astype("float32")

    # --- 枠 × 脚質、距離 × 脚質の相互作用 ---------------------------------
    # 決定木は掛け算を自力で作れないので、明示的に列として与える。
    if "bracket" in cols:
        bracket = pd.to_numeric(df[cols["bracket"]], errors="coerce").astype("float32")
        max_bracket = g[cols["bracket"]].transform("max").astype("float32")
        # 枠の相対位置（0=最内, 1=最外）。頭数が少ないと枠数も減るため正規化する
        waku = ((bracket - 1) / (max_bracket - 1).replace(0, np.nan)).astype("float32")
        df["枠相対位置"] = waku
        df["枠×脚質"] = (waku * df["脚質スコア"]).astype("float32")
        # 内枠で前に行ける馬 = 有利。値が大きいほどその条件に合致
        df["内枠先行度"] = ((1 - waku) * (1 - df["脚質スコア"])).astype("float32")

    if "distance" in cols:
        dist = pd.to_numeric(df[cols["distance"]], errors="coerce").astype("float32")
        df["距離×脚質"] = (dist / 1000.0 * df["脚質スコア"]).astype("float32")

    return df


def running_style_feature_columns() -> list[str]:
    """依頼Aで追加した特徴量の列名。"""
    return [
        "脚質スコア", "脚質カテゴリ", "脚質安定度", "脚質サンプル数", "脚質スコア直近3走",
        "前走4角相対位置", "好走時脚質スコア",
        "レース内逃げ馬頭数", "レース内先行馬頭数",
        "脚質スコア_レース内順位", "脚質スコア_レース内平均差",
        "単騎逃げフラグ", "前傾度",
        "枠相対位置", "枠×脚質", "内枠先行度", "距離×脚質",
    ]
