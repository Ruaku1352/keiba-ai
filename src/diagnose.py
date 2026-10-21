"""v6 の確認・診断（依頼 1(2) の馬体重代用の影響と、3 の小さな確認）。

    weight_impact_cv     validate と同じ fold で「実際の馬体重」と「代用した馬体重」の
                         AUC・重賞の達成率を比べる。同時に、モデル側の数字
                         （上位6頭の単勝人気の平均・的中レースの払戻の中央値）も出す
    payout_trend         モデルを使わない数字：年ごとの3連単払戻の中央値と 50,000円以上の割合
    three_f_agreement    ラップから計算した前3F・後3Fと公式値の一致（距離ごと）
    style_zero_table     SE の今回レース脚質判定「0」（初期値）が、どこから来ているか
    hot_payout_distribution  CV で熱い当たりになったレースの3連単払戻の分布
    arima_payouts        有馬記念の3連単（組番・払戻・人気）の一覧

どれも探索ではなく確認。買い方（重賞・上位6頭BOX）は変えない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features, jvmap, predict, preprocess, trifecta

HOT_PAYOUT = 50_000   # 上位6頭BOX（12,000円）の熱い基準


# ---------------------------------------------------------------------------
# 1(2) 馬体重の代用で予測がどれだけ落ちるか ＋ 3(4) モデル側の数字
# ---------------------------------------------------------------------------
def substituted_weight_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """特徴量の表の馬体重を「前走の馬体重」、増減を 0 にした版を作る。

    馬体重から作っている特徴量は「馬体重」「場体重増減」の2列だけ
    （test_v6_predict で、予測の経路の代用と学習の経路が他の列で一致することを確認している）。
    なので、学習の経路の表の2列を置き換えれば、全レースを予測の経路に通したのと同じになる。
    frame は [日付, レースID, 馬番] 順（make_training_frame / basic_clean の順）であること。
    """
    out = frame.copy()
    prev = predict.previous_weight(out)
    out["馬体重"] = prev.astype("float32")
    out["場体重増減"] = np.float32(0.0)
    return out


def _auc(y, p) -> float:
    """ROC AUC（Mann-Whitney の U から。同順位は平均順位）。scikit-learn に依存しないため。"""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(p, dtype="float64")).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _graded_box_stats(test: pd.DataFrame, pred: np.ndarray, payout_df: pd.DataFrame,
                      hot_rows: list | None = None) -> dict:
    """重賞・上位6頭BOX の成績と、モデルが選んだ6頭の人気。

    hot_rows にリストを渡すと、熱い当たりになったレースの行（DataFrame）を追加する。
    """
    race = trifecta.build_race_table(test, pred, payout_df)
    graded = race.loc[race["重賞"]]
    res = trifecta.simulate(graded, trifecta.box(predict.BOX_SIZE))
    if hot_rows is not None:
        hot_rows.append(hot_hit_races(graded, test, res.get("熱い基準", HOT_PAYOUT)))

    # 上位6頭に入った馬の単勝人気（確定レースの人気。特徴量には使っていない）
    t = test[["レースID", "人気", "リステッド・重賞競走"]].copy()
    t["pred"] = pred
    t = t.loc[t["リステッド・重賞競走"].isin(config.GRADED_VALUES)
              & t["レースID"].isin(set(graded.loc[graded["3連単払戻"].notna(), "レースID"]))]
    t["予測順位"] = t.groupby("レースID")["pred"].rank(ascending=False, method="first")
    top = t.loc[t["予測順位"] <= predict.BOX_SIZE]

    g = graded.loc[graded["3連単払戻"].notna()]
    hit = (g["1着予測順位"] <= 6) & (g["2着予測順位"] <= 6) & (g["3着予測順位"] <= 6)
    hit_pay = g.loc[hit, "3連単払戻"]
    return {
        "重賞R数": res.get("レース数", 0),
        "的中率": res.get("的中率"),
        "達成率": res.get("達成率"),
        "達成率下限": res.get("達成率下限"),
        "達成率上限": res.get("達成率上限"),
        "熱い当たり": res.get("熱い当たり", 0),
        "上位6頭の単勝人気の平均": float(top["人気"].mean()) if len(top) else np.nan,
        "全頭の単勝人気の平均": float(t["人気"].mean()) if len(t) else np.nan,
        "的中レースの払戻の中央値": float(hit_pay.median()) if len(hit_pay) else np.nan,
        "的中のうち熱い割合": float((hit_pay >= HOT_PAYOUT).mean()) if len(hit_pay) else np.nan,
        "重賞全体の払戻の中央値": float(g["3連単払戻"].median()) if len(g) else np.nan,
    }


def format_kumi(kumi) -> str:
    """"050312" -> "5-3-12"（3連単の組番。1着-2着-3着の馬番）。"""
    if isinstance(kumi, str) and len(kumi) == 6 and kumi.isdigit():
        return f"{int(kumi[:2])}-{int(kumi[2:4])}-{int(kumi[4:6])}"
    return ""


def hot_hit_races(graded: pd.DataFrame, test: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """重賞のレース表のうち、上位6頭BOX が熱い当たり（的中かつ払戻 ≥ 基準）になったレース。

    的中の判定は trifecta.simulate と同じ（出走頭数が6頭未満なら買える頭数まで）。
    """
    g = graded.loc[graded["3連単払戻"].notna()]
    k = np.minimum(predict.BOX_SIZE, g["出走頭数"])
    hit = (g["1着予測順位"] <= k) & (g["2着予測順位"] <= k) & (g["3着予測順位"] <= k)
    hot = g.loc[hit & (g["3連単払戻"] >= threshold)]
    names = test.drop_duplicates("レースID").set_index("レースID")
    cols = ["レースID", "日付", "格付け", "出走頭数", "3連単払戻"] + \
        [c for c in ("3連単組番", "3連単人気") if c in hot.columns]
    out = hot[cols].copy()
    if "3連単組番" in out.columns:
        out["3連単組番"] = out["3連単組番"].map(format_kumi)
    if "3連単人気" in out.columns:
        out["3連単人気"] = pd.to_numeric(out["3連単人気"], errors="coerce").round().astype("Int64")
    for src, dst in [("競走名", "レース名"), ("競馬場名", "競馬場")]:
        if src in names.columns:
            out[dst] = out["レースID"].map(names[src]).astype(object)
    return out


def hot_payout_distribution(hits: pd.DataFrame, by: str = "fold") -> pd.DataFrame:
    """熱い当たりの3連単払戻の分布（件数・最小・四分位・中央値・最大・平均）。by ごと＋全体。"""
    def stats(p: pd.Series) -> dict:
        p = p.dropna()
        if p.empty:
            return {"件数": 0}
        return {"件数": len(p), "最小": p.min(), "25%": p.quantile(0.25), "中央値": p.median(),
                "75%": p.quantile(0.75), "最大": p.max(), "平均": p.mean()}

    rows = [{by: name, **stats(g["3連単払戻"])} for name, g in hits.groupby(by, sort=False)]
    rows.append({by: "全体", **stats(hits["3連単払戻"])})
    return pd.DataFrame(rows)


def weight_impact_cv(dataset, folds=None, as_of: str = predict.AS_OF, model_fn=None,
                     verbose: bool = True) -> dict:
    """validate の fold で、実際の馬体重 vs 代用した馬体重を比べる。

    モデルは fold ごとに1つだけ学習する（validate と同じく、学習データの馬体重は実際の値）。
    同じモデルで、検証期間を「実際の馬体重」「代用した馬体重」の2通りで予測する。
    実戦の「発表前の予測」は後者にあたる。

    as_of : "day"（既定・v6 のモデルと同じ）か "race"（validate と同じ特徴量）
    model_fn(train, test, cols) -> booster : テスト用に差し替えられる
    """
    from . import validate

    folds = folds or validate.V5_FOLDS
    if verbose:
        print(f"[diagnose] 特徴量を作成（as_of={as_of}、全期間で1回だけ）...")
    df = preprocess.basic_clean(dataset.race_result)
    df = features.add_all_features(df, lap_df=dataset.lap_df, as_of=as_of)
    df = preprocess.downcast(df)
    df = preprocess.to_category(df)
    sub = substituted_weight_frame(df)
    cols = features.feature_columns(df)
    model_fn = model_fn or _train_with_early_stopping

    rows, hot_parts = [], []
    for fold in folds:
        train = validate._slice_years(df, fold.train_years)
        test = validate._slice_years(df, fold.test_years)
        test_sub = validate._slice_years(sub, fold.test_years)
        if train.empty or test.empty:
            continue
        if verbose:
            print(f"  fold{fold.name} {fold.label}: train {len(train):,} / test {len(test):,}")
        booster = model_fn(train, test, cols)
        best = getattr(booster, "best_iteration", None) or None
        p_real = booster.predict(test[cols], num_iteration=best)
        p_sub = booster.predict(test_sub[cols], num_iteration=best)
        for label, p in [("実際の馬体重", p_real), ("代用（前走・増減0）", p_sub)]:
            hot_rows: list = []
            st = _graded_box_stats(test, p, dataset.payout_df, hot_rows)
            for h in hot_rows:
                hot_parts.append(h.assign(fold=fold.name, 馬体重=label))
            rows.append({"fold": fold.name, "検証": f"{fold.test_years[0]}-{fold.test_years[1]}",
                         "馬体重": label, "AUC": _auc(test[config.TARGET], p),
                         "best_iteration": best, **st})
    table = pd.DataFrame(rows)
    pooled = _pool_weight_table(table)
    hot = pd.concat(hot_parts, ignore_index=True) if hot_parts else pd.DataFrame(
        columns=["fold", "馬体重", "レースID", "日付", "3連単払戻"])
    return {"table": table, "pooled": pooled, "hot_hits": hot}


def _train_with_early_stopping(train, test, cols):
    """validate と同じ学習（検証期間で early stopping）。再現性の設定だけ足す。"""
    from . import train as train_mod
    return train_mod.train_lgb(train, test, cols, params=predict.REPRO_PARAMS)


def _pool_weight_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    rows = []
    for label, g in table.groupby("馬体重", sort=False):
        hits, n = int(g["熱い当たり"].sum()), int(g["重賞R数"].sum())
        lo, hi = trifecta.wilson_interval(hits, n)
        rows.append({"馬体重": label, "重賞R数": n, "熱い当たり": hits,
                     "達成率": hits / n if n else np.nan, "下限": lo, "上限": hi,
                     "AUC（fold平均）": g["AUC"].mean()})
    out = pd.DataFrame(rows)
    if len(out) == 2:
        a, b = table.loc[table["馬体重"] == out.loc[0, "馬体重"]], table.loc[table["馬体重"] == out.loc[1, "馬体重"]]
        test = trifecta.two_proportion_test(int(b["熱い当たり"].sum()), int(b["重賞R数"].sum()),
                                            int(a["熱い当たり"].sum()), int(a["重賞R数"].sum()))
        out.attrs["差（代用−実際）"] = test["差"]
        out.attrs["差のp値"] = test["p値"]
    return out


# ---------------------------------------------------------------------------
# 3(4) モデルを使わない数字：年ごとの3連単払戻
# ---------------------------------------------------------------------------
def payout_trend(dataset, since_year: int = 2003) -> pd.DataFrame:
    """年ごとの重賞・平場の3連単払戻の中央値と、50,000円以上だったレースの割合。

    重賞の分け方は validate と同じ（G1/G2/G3/G が重賞、それ以外が平場）。
    3連単の払戻があるレースだけを数える（発売前・不成立・特払は除く）。
    """
    races = dataset.race_result.drop_duplicates("レースID")[
        ["レースID", "レース日付", "リステッド・重賞競走", "芝・ダート区分"]]
    pay = dataset.payout_df[["レースID", "3連単払戻"]]
    r = races.merge(pay, on="レースID", how="inner")
    r = r.loc[r["3連単払戻"].notna()]
    r["年"] = pd.to_datetime(r["レース日付"]).dt.year
    r = r.loc[r["年"] >= since_year]
    r["区分"] = np.where(r["リステッド・重賞競走"].isin(config.GRADED_VALUES), "重賞", "平場")

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"R数": len(g), "中央値": g["3連単払戻"].median(),
                          "5万円以上の割合": (g["3連単払戻"] >= HOT_PAYOUT).mean()})

    parts = {k: r.loc[r["区分"] == k].groupby("年").apply(agg, include_groups=False)
             for k in ("重賞", "平場")}
    out = pd.concat(parts, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


# ---------------------------------------------------------------------------
# 3(3) ラップから計算した前3F・後3F と公式値の一致
# ---------------------------------------------------------------------------
def three_f_agreement(dataset, since: str = "2003-01-01") -> pd.DataFrame:
    """公式値とラップからの計算値が両方あるレースで、距離ごとに一致を確かめる。

    一致 = 差の絶対値が 0.05 秒未満（どちらも0.1秒単位なので、同じ値かどうか）。
    差 = 計算値 − 公式値。
    仕様書（RA 前3ハロン）: 200m で割り切れない距離は「端数＋400m」のタイム
    （＝ラップの先頭3本の合計）。計算もラップの先頭3本の合計なので、定義は同じはず。
    """
    need = ["_前3F公式", "_前3F計算", "_後3F公式", "_後3F計算"]
    rr = dataset.race_result
    if not all(c in rr.columns for c in need):
        raise KeyError("race_result に公式値・計算値の列がありません（build し直してください）")
    races = rr.drop_duplicates("レースID")
    races = races.loc[(pd.to_datetime(races["レース日付"]) >= pd.Timestamp(since))
                      & races["芝・ダート区分"].isin(["芝", "ダート"])]
    rows = []
    for dist, g in races.groupby("距離(m)"):
        row = {"距離": int(dist), "200mで割り切れる": int(dist) % 200 == 0, "レース数": len(g)}
        for name in ("前3F", "後3F"):
            both = g[[f"_{name}公式", f"_{name}計算"]].dropna()
            d = both[f"_{name}計算"] - both[f"_{name}公式"]
            row[f"{name}_両方あり"] = len(both)
            row[f"{name}_一致率"] = float((d.abs() < 0.05).mean()) if len(both) else np.nan
            row[f"{name}_差の平均"] = float(d.mean()) if len(both) else np.nan
            row[f"{name}_差の絶対値の平均"] = float(d.abs().mean()) if len(both) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    total = {"距離": "全体", "200mで割り切れる": "", "レース数": int(out["レース数"].sum())}
    for name in ("前3F", "後3F"):
        both = races[[f"_{name}公式", f"_{name}計算"]].dropna()
        d = both[f"_{name}計算"] - both[f"_{name}公式"]
        total.update({f"{name}_両方あり": len(both),
                      f"{name}_一致率": float((d.abs() < 0.05).mean()) if len(both) else np.nan,
                      f"{name}_差の平均": float(d.mean()) if len(both) else np.nan,
                      f"{name}_差の絶対値の平均": float(d.abs().mean()) if len(both) else np.nan})
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------
# 3(2) 今回レース脚質判定の「0」
# ---------------------------------------------------------------------------
def style_zero_table(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """SE の KyakusituKubun（今回レース脚質判定）が "0"（仕様書: 初期値）の行の内訳を年ごとに。

    仕様書では、この項目はデータ区分 6・7（確定成績）でだけ設定され、
    1〜5（出馬表・速報）と海外（B）は初期値、地方（A）は「設定する場合としない場合が混在」。
    変換（jvmap.map_horses）では "0" を欠損（NaN）にしている。

    列:
      生レコード / 0の件数 / 0の割合（生レコード全体）
      0の内訳: 重複（後で新しい版に置き換わる行）/ 中央以外 / 確定前の区分(1〜5) /
               取消・除外・中止など / 最終データ（学習に使う行）
    """
    se = raw["SE"].copy()
    se["_年"] = pd.to_numeric(se["id.Year"], errors="coerce")
    zero = se["KyakusituKubun"].astype(str).str.strip().isin({"0", ""})

    kept = jvmap.dedupe_records(se.assign(_row=np.arange(len(se))), extra_keys=("KettoNum",))
    is_kept = pd.Series(False, index=se.index)
    is_kept.iloc[kept["_row"].to_numpy()] = True
    central = se["id.JyoCD"].isin(jvmap.CENTRAL_JYO)
    usable = se["head.DataKubun"].isin(jvmap.USABLE_RACE_KUBUN)
    excluded = se["IJyoCD"].isin(jvmap.EXCLUDE_IJYO)

    reason = np.select(
        [~is_kept, ~central, ~usable, excluded],
        ["重複（古い版）", "中央以外", "確定前の区分", "取消・除外・中止など"],
        default="最終データ")
    t = pd.DataFrame({"年": se["_年"], "0": zero, "理由": reason})
    by_year = t.groupby("年").agg(生レコード=("0", "size"), 件数_0=("0", "sum"))
    by_year["0の割合"] = by_year["件数_0"] / by_year["生レコード"]
    breakdown = pd.crosstab(t.loc[t["0"], "年"], t.loc[t["0"], "理由"])
    out = by_year.join(breakdown, how="left").fillna(0)
    # 最終データ（学習に使う中央・確定・取消なし）の中での 0 の割合
    final = t.loc[t["理由"] == "最終データ"]
    out["最終データでの0の割合"] = final.groupby("年")["0"].mean()
    return out


# ---------------------------------------------------------------------------
# 有馬記念の3連単
# ---------------------------------------------------------------------------
ARIMA_NAME = "有馬記念"


def arima_payouts(dataset, since_year: int = 2004) -> pd.DataFrame:
    """有馬記念（競走名に「有馬記念」を含む G1）の3連単の一覧。

    列: 年, 日付, レースID, 頭数, 組番（1着-2着-3着の馬番）, 払戻（100円あたり）, 3連単人気,
        1〜3着の馬名と単勝人気
    3連単の人気順は HR の「3連単払戻の人気順」。同着で的中組が複数あるときは1番目の組。
    """
    rr = dataset.race_result
    races = rr.drop_duplicates("レースID")
    races = races.loc[races["競走名"].astype(str).str.contains(ARIMA_NAME)
                      & (races["リステッド・重賞競走"] == "G1")
                      & (pd.to_datetime(races["レース日付"]).dt.year >= since_year)]
    pay = dataset.payout_df.set_index("レースID")
    rows = []
    for _, r in races.sort_values("レース日付").iterrows():
        rid = r["レースID"]
        horses = rr.loc[rr["レースID"] == rid]
        row = {"年": pd.Timestamp(r["レース日付"]).year, "日付": f"{pd.Timestamp(r['レース日付']):%Y-%m-%d}",
               "レースID": rid, "頭数": len(horses)}
        p = pay.loc[rid] if rid in pay.index else None
        kumi = p.get("3連単組番") if p is not None else None
        row["組番"] = format_kumi(kumi)
        row["払戻"] = p.get("3連単払戻") if p is not None else np.nan
        row["3連単人気"] = p.get("3連単人気") if p is not None else np.nan
        for k in (1, 2, 3):
            h = horses.loc[horses["着順"] == k]
            row[f"{k}着"] = "/".join(f"{int(x.馬番)} {x.馬名表示}" for x in h.itertuples()) if len(h) else ""
            row[f"{k}着人気"] = "/".join(str(int(x)) for x in h["人気"].dropna()) if len(h) else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["3連単人気"] = pd.to_numeric(out["3連単人気"], errors="coerce").round().astype("Int64")
    return out


# ---------------------------------------------------------------------------
# v7 共通：fold ごとの学習と、レースごとの「熱い当たり」
# ---------------------------------------------------------------------------
def build_feature_frame(dataset, weekend: bool = False) -> pd.DataFrame:
    """as_of="day" の特徴量の表。weekend=True なら「その週末（連続した開催日）より前」だけで集計する。

    weekend の仕組み：集計の除外単位（_集計日）を「その開催日が属する連続開催の初日」にする。
    as_of="day" は同じ _集計日 の行を互いの過去に含めないので、土日（月曜祝日なら土日月）の
    レースどうしが混ざらなくなる。本番で「日曜の予測に土曜の結果が無い」状態と同じになる。
    （例外：馬の前走・通算成績は日付で切らないが、同じ週末に2回走る馬はほぼいない）
    過去の各レースのペースの z 化だけは、本番の予測の経路と同じく「そのレースの前日まで」で行う
    （本番では過去のレースは前日まで、で計算されるため。test_v7 で予測の経路との一致を確認）。
    """
    df = preprocess.basic_clean(dataset.race_result)
    if weekend:
        from . import pace
        df[features.DAY_COL] = weekend_block_start(df["レース日付"])
        df[pace.STANDARDIZE_DAY_COL] = df["レース日付"]   # 各レースのペースの z 化は本番と同じく前日まで
    df = features.add_all_features(df, lap_df=dataset.lap_df, as_of="day")
    df = preprocess.downcast(df)
    return preprocess.to_category(df)


def weekend_block_start(dates: pd.Series) -> pd.Series:
    """各開催日について、1日も空かずに続く開催（土日、土日月など）の初日。"""
    d = pd.to_datetime(dates)
    days = np.sort(d.unique())
    if len(days) == 0:
        return d
    gap = np.diff(days) > np.timedelta64(1, "D")
    block = np.r_[0, np.cumsum(gap)]
    first = days[np.r_[True, gap]]
    return d.map(pd.Series(first[block], index=days))


def race_hot_flags(test: pd.DataFrame, pred: np.ndarray, payout_df: pd.DataFrame) -> pd.Series:
    """重賞・上位6頭BOX が熱い当たりになったか（レースID → bool）。払戻のある重賞だけ。

    判定は trifecta.simulate と同じ（出走頭数が6未満なら買える頭数まで、基準は投資額の中央値から）。
    """
    race = trifecta.build_race_table(test, pred, payout_df)
    g = race.loc[race["重賞"] & race["3連単払戻"].notna()]
    res = trifecta.simulate(g, trifecta.box(predict.BOX_SIZE))
    threshold = res.get("熱い基準", HOT_PAYOUT)
    k = np.minimum(predict.BOX_SIZE, g["出走頭数"])
    hit = (g["1着予測順位"] <= k) & (g["2着予測順位"] <= k) & (g["3着予測順位"] <= k)
    return pd.Series((hit & (g["3連単払戻"] >= threshold)).to_numpy(), index=g["レースID"].to_numpy())


def _with_substituted_weight(test: pd.DataFrame, prev_weight: pd.Series) -> pd.DataFrame:
    out = test.copy()
    out["馬体重"] = prev_weight.loc[test.index].astype("float32")
    out["場体重増減"] = np.float32(0.0)
    return out


def _compare_arms(arms: dict[str, pd.Series], base: str) -> pd.DataFrame:
    """同じ重賞レースでの熱い当たりを、腕ごとに集計し base との対応のある差を出す。"""
    rows = []
    ref = arms[base]
    for name, flags in arms.items():
        common = ref.index.intersection(flags.index)
        hot, n = int(flags.loc[common].sum()), len(common)
        lo, hi = trifecta.wilson_interval(hot, n)
        row = {"条件": name, "重賞R数": n, "熱い当たり": hot, "達成率": hot / n if n else np.nan,
               "下限": lo, "上限": hi}
        if name != base:
            t = trifecta.paired_proportion_test(flags.loc[common].to_numpy(), ref.loc[common].to_numpy())
            row.update({f"差（−{base}）": t["差"], "差の下限": t["差下限"], "差の上限": t["差上限"],
                        "p値": t["p値"], "こちらだけ当たり": t["xだけ"], f"{base}だけ当たり": t["yだけ"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in out.columns:
        if c.endswith("だけ当たり"):
            out[c] = out[c].astype("Int64")
    return out


# ---------------------------------------------------------------------------
# v7 3(2) 同じ週末の前の日の結果が入らないことの影響
# ---------------------------------------------------------------------------
WEEKEND_ARMS = {
    "A": "学習=前日まで / 検証=前日まで（今の検証と同じ）",
    "B": "学習=前日まで / 検証=週末より前（今の本番。速報を取らない場合）",
    "C": "学習=週末より前 / 検証=週末より前（学習側を本番に合わせる案）",
}


def weekend_gap_cv(dataset, folds=None, model_fn=None, verbose: bool = True) -> dict:
    """「前日までの結果」と「その週末より前の結果だけ」で作った特徴量を比べる（5 fold）。

    3つの腕（WEEKEND_ARMS）。検証の馬体重はすべて代用（前走・増減0）＝本番の条件。
    A と B は同じモデル（前日までの特徴量で学習）で、検証の特徴量だけを変える → 本番で失う分。
    C は週末より前の特徴量で学習し直したモデル → 学習側を本番に合わせたときにどこまで戻るか。
    学習は validate と同じ（検証期間で early stopping）。
    """
    from . import validate

    folds = folds or validate.V5_FOLDS
    model_fn = model_fn or _train_with_early_stopping
    if verbose:
        print("[diagnose] 特徴量を作成：前日まで ...")
    day = build_feature_frame(dataset, weekend=False)
    if verbose:
        print("[diagnose] 特徴量を作成：週末より前 ...")
    wk = _align_rows(build_feature_frame(dataset, weekend=True), day)
    prev = predict.previous_weight(day)
    cols = features.feature_columns(day)

    flags = {k: [] for k in WEEKEND_ARMS}
    aucs = []
    for fold in folds:
        tr_d, te_d = validate._slice_years(day, fold.train_years), validate._slice_years(day, fold.test_years)
        tr_w, te_w = validate._slice_years(wk, fold.train_years), validate._slice_years(wk, fold.test_years)
        if tr_d.empty or te_d.empty:
            continue
        if verbose:
            print(f"  fold{fold.name} {fold.label}: train {len(tr_d):,} / test {len(te_d):,}")
        te_d_sub, te_w_sub = _with_substituted_weight(te_d, prev), _with_substituted_weight(te_w, prev)
        m_day = model_fn(tr_d, te_d, cols)
        m_wk = model_fn(tr_w, te_w, cols)
        preds = {
            "A": m_day.predict(te_d_sub[cols], num_iteration=getattr(m_day, "best_iteration", None) or None),
            "B": m_day.predict(te_w_sub[cols], num_iteration=getattr(m_day, "best_iteration", None) or None),
            "C": m_wk.predict(te_w_sub[cols], num_iteration=getattr(m_wk, "best_iteration", None) or None),
        }
        row = {"fold": fold.name}
        for k, p in preds.items():
            flags[k].append(race_hot_flags(te_d, p, dataset.payout_df))
            row[f"AUC_{k}"] = _auc(te_d[config.TARGET], p)
        aucs.append(row)
    arms = {k: pd.concat(v) for k, v in flags.items()}
    table = _compare_arms(arms, base="A")
    table.insert(1, "説明", table["条件"].map(WEEKEND_ARMS))
    auc = pd.DataFrame(aucs)
    # C と B の対応のある差（学習側を合わせると取り戻せるか）
    common = arms["B"].index.intersection(arms["C"].index)
    c_vs_b = trifecta.paired_proportion_test(arms["C"].loc[common].to_numpy(),
                                             arms["B"].loc[common].to_numpy())
    share = _same_weekend_share(dataset)
    return {"table": table, "auc": auc, "C_vs_B": c_vs_b, "影響を受ける割合": share}


def _align_rows(frame: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """frame の行を like と同じ順番・同じ index にそろえる（キーは レースID・血統登録番号）。"""
    key = ["レースID", "血統登録番号"]
    a = pd.MultiIndex.from_frame(frame[key].astype(str))
    b = pd.MultiIndex.from_frame(like[key].astype(str))
    if a.has_duplicates or not b.isin(a).all():
        raise ValueError("2つの特徴量の表の行が対応しません")
    out = frame.iloc[a.get_indexer(b)].copy()
    out.index = like.index
    return out


def _same_weekend_share(dataset) -> dict:
    """重賞のうち、同じ週末の前の日に開催があるレース（＝本番で前日の結果が欠けるレース）の割合。"""
    races = dataset.race_result.drop_duplicates("レースID")
    races = races.loc[races["リステッド・重賞競走"].isin(config.GRADED_VALUES)]
    all_days = pd.Series(pd.to_datetime(dataset.race_result["レース日付"]).unique())
    first = pd.Series(weekend_block_start(all_days).to_numpy(), index=all_days.to_numpy())
    d = pd.to_datetime(races["レース日付"])
    affected = d.map(first) < d
    return {"重賞レース数": int(len(races)), "週末の2日目以降": int(affected.sum()),
            "割合": float(affected.mean()) if len(races) else np.nan}


# ---------------------------------------------------------------------------
# v7 4 学習開始 1995年 vs 2003年
# ---------------------------------------------------------------------------
START_YEAR_RULE = (
    "判定ルール（結果を見る前に固定）:\n"
    "  主指標 = 重賞・上位6頭BOX の熱い当たりの有無（5 fold の検証レースをプール。馬体重は代用＝本番の条件）\n"
    "  同じ検証レースを両方のモデルで予測するので、対応のある比較（Newcombe 法10 の差の 95% 区間）を使う\n"
    "  差（1995 − 2003）の 95% 区間の下限が 0 を超えたら 1995年を採用。そうでなければ 2003年のまま\n"
    "  AUC・実際の馬体重での達成率は参考として表示するだけで、判定には使わない"
)


def start_year_cv(dataset, folds=None, years=(1995, 2003), model_fn=None, verbose: bool = True) -> dict:
    """学習開始年だけを変えて、同じ検証 fold で比べる（as_of=day）。比べるのはこの1組だけ。"""
    from . import validate

    folds = folds or validate.V5_FOLDS
    model_fn = model_fn or _train_with_early_stopping
    if verbose:
        print("[diagnose] 特徴量を作成（as_of=day、全期間で1回だけ）...")
    df = build_feature_frame(dataset)
    prev = predict.previous_weight(df)
    cols = features.feature_columns(df)

    sub_flags = {y: [] for y in years}
    real_flags = {y: [] for y in years}
    aucs = []
    for fold in folds:
        test = validate._slice_years(df, fold.test_years)
        if test.empty:
            continue
        test_sub = _with_substituted_weight(test, prev)
        row = {"fold": fold.name, "検証": f"{fold.test_years[0]}-{fold.test_years[1]}"}
        for y in years:
            train = validate._slice_years(df, (y, fold.train_years[1]))
            if verbose:
                print(f"  fold{fold.name} 学習 {y}-{fold.train_years[1]}: train {len(train):,} / test {len(test):,}")
            m = model_fn(train, test, cols)
            best = getattr(m, "best_iteration", None) or None
            p_sub = m.predict(test_sub[cols], num_iteration=best)
            p_real = m.predict(test[cols], num_iteration=best)
            sub_flags[y].append(race_hot_flags(test, p_sub, dataset.payout_df))
            real_flags[y].append(race_hot_flags(test, p_real, dataset.payout_df))
            row[f"AUC_{y}（代用）"] = _auc(test[config.TARGET], p_sub)
            row[f"best_iteration_{y}"] = best
            row[f"学習行数_{y}"] = len(train)
        aucs.append(row)

    new, old = years
    names = {new: f"学習開始{new}", old: f"学習開始{old}"}
    sub = {names[y]: pd.concat(v) for y, v in sub_flags.items()}
    real = {names[y]: pd.concat(v) for y, v in real_flags.items()}
    table = _compare_arms({names[new]: sub[names[new]], names[old]: sub[names[old]]}, base=names[old])
    table_real = _compare_arms({names[new]: real[names[new]], names[old]: real[names[old]]}, base=names[old])
    lower = float(table.loc[table["条件"] == names[new], "差の下限"].iloc[0])
    adopt = bool(lower > 0)
    verdict = (f"採用：{new}年から学習する（差の下限 {lower:+.2%} > 0）" if adopt
               else f"不採用：{old}年のまま（差の下限 {lower:+.2%} ≤ 0）")
    return {"rule": START_YEAR_RULE, "table": table, "table_real": table_real,
            "auc": pd.DataFrame(aucs), "adopt": adopt, "verdict": verdict}
