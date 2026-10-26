"""v8 3b・3c：市場の確率に「傾向」で傾きをつける（平場で試す。本番の特徴量・予測には入れていない）。

最終的に使う確率は市場（＋Henery 型）なので、「市場がその傾向を織り込み切れているか」を確かめる。

    P_i ∝ exp(β·log 市場_i + Σ θ_k·項_ik)

係数（β, θ）は、評価する fold より前の fold の**平場**だけで推定する（market.benter_walk_forward と同じ）。
θ が 0 に近ければ、市場はその傾向を織り込み済み。

■ 3b 当日の傾向（trackbias と同じ。発走 40 分前までに発走した、同じ日・同じ競馬場・同じ芝ダのレース）
    当日_内枠×枠位置 = 当日の内枠傾向 × (0.5 − 馬番位置)
    当日_前残り×脚質 = 当日の前残り傾向 × (0.5 − 脚質スコア)

■ 3c 開催の進み具合（当日の結果を使わない。前日までのデータだけで計算できる）
    開催（同じ年・競馬場・回）の中で、同じ芝ダ・距離帯（〜1400 / 〜1800 / それより長い）の**前日までの**レースの
    内枠傾向・前残り傾向を「合計 ÷ (レース数 + 2)」で平均（少ないときは 0 に寄せる）。開催の初日は 0。
    開催_内枠×枠位置   = 開催の内枠傾向 × (0.5 − 馬番位置)
    開催_前残り×脚質   = 開催の前残り傾向 × (0.5 − 脚質スコア)
    開催日目×枠位置    = (開催日目 − 1) / 7 × (0.5 − 馬番位置)   … 開催が進むと内が荒れる、などの一般的な傾き
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import corner, leakfree, market, preprocess, trackbias

DAY_TERMS = ["当日_内枠×枠位置", "当日_前残り×脚質"]
MEETING_TERMS = ["開催_内枠×枠位置", "開催_前残り×脚質", "開催日目×枠位置"]
MEETING_SHRINK = 2.0


def style_frame(dataset) -> pd.DataFrame:
    """脚質スコア（過去の4角位置）と4角相対位置まで付けた表。全部の特徴量は作らない（速さのため）。"""
    df = preprocess.basic_clean(dataset.race_result)
    df = corner.attach_corner_from_columns(df)
    df = corner.add_running_style_features(df)
    return df


def _post_position(df: pd.DataFrame) -> pd.Series:
    n = pd.to_numeric(df["公式出走頭数"], errors="coerce") if "公式出走頭数" in df.columns else \
        df.groupby("レースID")["レースID"].transform("size")
    return (pd.to_numeric(df["馬番"], errors="coerce") - 1) / (n - 1).replace(0, np.nan)


def add_meeting_features(df: pd.DataFrame, shrink: float = MEETING_SHRINK) -> pd.DataFrame:
    """3c の項。df は style_frame の戻り値（日付→レースID→馬番の順）。"""
    df = df.copy()
    races = trackbias.race_bias_stats(df)
    races["開催"] = races["レースID"].astype(str).str[:8]                 # 年4 + 場2 + 回2
    races["開催日目"] = pd.to_numeric(races["レースID"].astype(str).str[8:10], errors="coerce")
    band = pd.cut(races["距離"], [0, 1400, 1800, 100000], labels=False, right=True)
    races["_mk"] = leakfree.label_encode(races["開催"] + "_" + races["芝ダ"].astype(str) + "_" + band.astype(str))
    races = races.sort_values(["日付", "レースID"], kind="mergesort").reset_index(drop=True)
    for stat, name in [("内枠傾向", "開催_内枠傾向"), ("前残り傾向", "開催_前残り傾向")]:
        v = races[stat].astype("float64")
        s = leakfree.past_sum_excluding_race(races, ["_mk"], v.fillna(0), "日付")       # 前日まで（同じ日は除く）
        c = leakfree.past_sum_excluding_race(races, ["_mk"], v.notna().astype("float64"), "日付")
        races[name] = s / (c + shrink)
    m = races.set_index("レースID")
    rid = df["レースID"].to_numpy()
    for c in ["開催_内枠傾向", "開催_前残り傾向", "開催日目"]:
        df[c] = m[c].reindex(rid).to_numpy()
    post = _post_position(df)
    style = df["脚質スコア"] if "脚質スコア" in df.columns else pd.Series(np.nan, index=df.index)
    df["開催_内枠×枠位置"] = df["開催_内枠傾向"] * (0.5 - post)
    df["開催_前残り×脚質"] = df["開催_前残り傾向"] * (0.5 - style)
    df["開催日目×枠位置"] = (df["開催日目"] - 1) / 7.0 * (0.5 - post)
    return df


def build_terms(dataset, cutoff_min: float = 20, delay_min: float = 20) -> pd.DataFrame:
    """馬ごとの 3b・3c の項（レースID・血統登録番号つき）。"""
    df = style_frame(dataset)
    df = trackbias.add_track_bias_features(df, cutoff_min=cutoff_min, delay_min=delay_min)
    df = add_meeting_features(df)
    keep = ["レースID", "血統登録番号", "当日_使えたレース数", "当日_内枠傾向", "当日_前残り傾向",
            "開催_内枠傾向", "開催_前残り傾向", "開催日目"] + DAY_TERMS + MEETING_TERMS
    return df[keep].reset_index(drop=True)


def _design(t: pd.DataFrame, base_col: str, terms: list[str]) -> np.ndarray:
    cols = [np.log(t[base_col].astype("float64").clip(lower=1e-6)).to_numpy()]
    cols += [t[c].astype("float64").fillna(0.0).to_numpy() for c in terms]
    return np.column_stack(cols)


def tilt_walk_forward(table: pd.DataFrame, terms: list[str], out_col: str,
                      base_col: str = "市場確率") -> tuple[pd.DataFrame, pd.DataFrame]:
    """fold ごとに、前の fold の平場で (β, θ) を推定し、その fold の確率 out_col を作る（fold 2 以降）。"""
    t = market._one_winner(table)
    folds = list(dict.fromkeys(t["fold"]))
    parts, coefs = [], []
    for i, f in enumerate(folds):
        if i == 0:
            continue
        fit = t.loc[t["fold"].isin(folds[:i]) & ~t["重賞"]]
        fs, starts = market.race_slices(fit)
        beta = market.fit_conditional_logit(_design(fs, base_col, terms), starts, fs["win"].to_numpy())
        coefs.append({"検証fold": f, "推定レース数": int(len(starts) - 1), "β（市場）": beta[0],
                      **{f"θ（{c}）": b for c, b in zip(terms, beta[1:])}})
        ev = t.loc[t["fold"] == f].copy()
        u = _design(ev, base_col, terms) @ beta
        u = pd.Series(u, index=ev.index)
        e = np.exp(u - u.groupby(ev["レースID"]).transform("max"))
        ev[out_col] = (e / e.groupby(ev["レースID"]).transform("sum")).to_numpy()
        parts.append(ev)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(coefs)
