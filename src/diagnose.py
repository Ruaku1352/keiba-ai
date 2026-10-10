"""v6 の確認・診断（依頼 1(2) の馬体重代用の影響と、3 の小さな確認）。

    weight_impact_cv     validate と同じ fold で「実際の馬体重」と「代用した馬体重」の
                         AUC・重賞の達成率を比べる。同時に、モデル側の数字
                         （上位6頭の単勝人気の平均・的中レースの払戻の中央値）も出す
    payout_trend         モデルを使わない数字：年ごとの3連単払戻の中央値と 50,000円以上の割合
    three_f_agreement    ラップから計算した前3F・後3Fと公式値の一致（距離ごと）
    style_zero_table     SE の今回レース脚質判定「0」（初期値）が、どこから来ているか

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


def _graded_box_stats(test: pd.DataFrame, pred: np.ndarray, payout_df: pd.DataFrame) -> dict:
    """重賞・上位6頭BOX の成績と、モデルが選んだ6頭の人気。"""
    race = trifecta.build_race_table(test, pred, payout_df)
    graded = race.loc[race["重賞"]]
    res = trifecta.simulate(graded, trifecta.box(predict.BOX_SIZE))

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

    rows, model_rows = [], []
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
            st = _graded_box_stats(test, p, dataset.payout_df)
            rows.append({"fold": fold.name, "検証": f"{fold.test_years[0]}-{fold.test_years[1]}",
                         "馬体重": label, "AUC": _auc(test[config.TARGET], p),
                         "best_iteration": best, **st})
    table = pd.DataFrame(rows)
    pooled = _pool_weight_table(table)
    return {"table": table, "pooled": pooled}


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

    kept = jvmap.apply_data_kubun(se.assign(_row=np.arange(len(se))),
                                  jvmap.RACE_KEY + ["KettoNum"], jvmap.RACE_KUBUN_PRIORITY)
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
