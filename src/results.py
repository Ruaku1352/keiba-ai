"""v7：保存した予測と、届いた結果の照合（練習の成績表）。

■ どの予測を使うか
  data/predictions/ の予測 CSV をすべて読む。
    predict_*.csv … 本番の流れで出した予測（区分「本番予測」）
    replay_*.csv  … 練習モード（区分「練習（replay）」）
  同じレースが複数のファイルにあれば:
    本番予測 → **発走時刻より前に作った最後のファイル**を使う（作成時刻はファイル名の _YYYYmmdd_HHMMSS）。
               発走後に作ったものしか無いレースは照合しない（理由を表示する）
    練習     → 最後のファイル。ただしモデルの学習終了日がレース日以降なら「参考外」（モデルが結果を見ている）

■ 照合（1レースごと）
  - 実際の1〜3着（馬番）、3連単の払戻（同着で的中組が複数あれば、そのすべて）
  - 6頭BOX が的中したか：払戻の的中組（1着-2着-3着の馬番）が、BOX の馬番にすべて入っているか
  - 出走取消・除外（異常区分 1〜3）の馬を含む買い目は返還される。投資は実際に走った頭数 n で
    n×(n-1)×(n-2)×100円（6頭とも走れば 12,000円）
  - 熱い当たり：的中した払戻の合計 ≥ 50,000円（6頭BOX の基準。返還で投資が減っても基準は変えない）
  - 仮の収支 = 的中の払戻合計 − 投資
  予測の CSV に馬番が無い（出走馬名表の段階）ときは、馬名で結果の馬番に対応付ける。

■ 結果の取り方
  生データ（RA/SE/HR）から直接読む。build_dataset は競走中止・失格の馬を落とすが、
  照合では「走ったが着順なし」と「取消・除外（返還）」を区別したいため。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import jvmap, predict, trifecta

# validate（as_of=day・馬体重は代用＝本番と同じ条件）で見込まれる達成率。
# v6 の diagnose weight の「代用」のプール値（ユーザーが実データで実行した結果）。
EXPECTED_RATE = 0.0470
EXPECTED_CI = (0.0385, 0.0571)

INVEST = predict.BOX_SIZE * (predict.BOX_SIZE - 1) * (predict.BOX_SIZE - 2) * trifecta.BET_UNIT
HOT_THRESHOLD = trifecta.hot_threshold(INVEST)          # 50,000円
REFUND_IJYO = {"1", "2", "3"}                            # 出走取消・発走除外・競走除外（返還）
IJYO_LABELS = {"1": "取消", "2": "発走除外", "3": "競走除外", "4": "中止", "5": "失格",
               "6": "落馬再騎乗", "7": "降着"}

PRACTICE = "練習（replay）"
LIVE = "本番予測"
_STAMP = re.compile(r"_(\d{8})_(\d{6})\.csv$")


# ---------------------------------------------------------------------------
# 予測の読み込み
# ---------------------------------------------------------------------------
def load_predictions(pred_dir: str | Path | None = None) -> pd.DataFrame:
    """予測 CSV をすべて読み、ファイル名・作成時刻・区分を付けてつなぐ。"""
    pred_dir = Path(pred_dir or predict.prediction_dir())
    parts = []
    for path in sorted(pred_dir.glob("*.csv")):
        if path.name.startswith("predict_"):
            kind = LIVE
        elif path.name.startswith("replay_"):
            kind = PRACTICE
        else:
            continue
        m = _STAMP.search(path.name)
        if not m:
            continue
        df = pd.read_csv(path, dtype={"レースID": str}, encoding="utf-8-sig")
        df["予測ファイル"] = path.name
        df["作成時刻"] = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
        df["区分"] = kind
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["レースID"] = out["レースID"].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def race_schedule(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """全レース（出馬表の段階も含む）の発走日時。発走時刻が変わったら新しい版が残る。"""
    ra = jvmap.central_only(jvmap.apply_data_kubun(raw["RA"], jvmap.RACE_KEY,
                                                   jvmap.RACE_KUBUN_PRIORITY))
    races = pd.DataFrame({"レースID": jvmap.make_race_id(ra).to_numpy(),
                          "レース日付": jvmap.make_date(ra).to_numpy(),
                          "発走時刻": ra["HassoTime"].astype(str).str.strip().to_numpy(),
                          "RAデータ区分": ra["head.DataKubun"].to_numpy()})
    hhmm = races["発走時刻"].where(races["発走時刻"].str.fullmatch(r"\d{4}"), "2359")
    races["発走日時"] = races["レース日付"] + pd.to_timedelta(
        hhmm.str[:2].astype(int) * 60 + hhmm.str[2:].astype(int), unit="min")
    return races


def choose_final_predictions(preds: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """レースごとに照合に使う予測ファイルを1つ選ぶ。戻り値は (選んだ予測の行, 使えなかったレースと理由)。"""
    if preds.empty:
        return preds, pd.DataFrame(columns=["区分", "レースID", "理由"])
    post = schedule.set_index("レースID")["発走日時"]
    files = preds.drop_duplicates(["区分", "レースID", "予測ファイル"])[
        ["区分", "レースID", "予測ファイル", "作成時刻"]].copy()
    files["発走日時"] = files["レースID"].map(post)

    pick, skipped = {}, []
    for (kind, rid), g in files.groupby(["区分", "レースID"], sort=False):
        g = g.sort_values("作成時刻")
        if kind == LIVE:
            g = g.loc[g["発走日時"].isna() | (g["作成時刻"] < g["発走日時"])]
            if g.empty:
                when = files.loc[files["レースID"] == rid, "発走日時"].iloc[0]
                skipped.append({"区分": kind, "レースID": rid,
                                "理由": f"発走（{when:%Y-%m-%d %H:%M}）より前に作った予測がない"})
                continue
        pick[(kind, rid)] = g.iloc[-1]["予測ファイル"]

    keep = [pick.get((k, r)) == f for k, r, f in zip(preds["区分"], preds["レースID"], preds["予測ファイル"])]
    return preds.loc[keep].reset_index(drop=True), pd.DataFrame(skipped)


# ---------------------------------------------------------------------------
# 結果の読み込み
# ---------------------------------------------------------------------------
def race_results(raw: dict[str, pd.DataFrame], race_ids) -> pd.DataFrame:
    """指定レースの馬ごとの結果（着順確定 5/6/7 のレースだけ）。取消・中止の馬も残す。"""
    ids = set(race_ids)
    se = raw["SE"]
    se = se.loc[jvmap.make_race_id(se).isin(ids)]
    if se.empty:
        return pd.DataFrame(columns=["レースID", "馬番", "馬名表示", "着順", "異常区分"])
    se = jvmap.central_only(jvmap.apply_data_kubun(se, jvmap.RACE_KEY + ["KettoNum"],
                                                   jvmap.RACE_KUBUN_PRIORITY))
    se = se.loc[se["head.DataKubun"].isin(jvmap.USABLE_RACE_KUBUN)]
    return jvmap.map_horses(se)


@dataclass
class RaceCheck:
    row: dict
    horses: pd.DataFrame


def check_race(pred: pd.DataFrame, horses: pd.DataFrame, pay_row: pd.Series | None) -> RaceCheck:
    """1レースの照合。pred はそのレースの予測（1頭1行）、horses は結果（1頭1行）。"""
    p = pred.sort_values("予測順位")
    r0 = p.iloc[0]
    row = {"区分": r0["区分"], "レースID": r0["レースID"], "日付": r0["日付"], "レース名": r0["レース名"],
           "格付け": r0.get("格付け"), "予測ファイル": r0["予測ファイル"],
           "モデル学習終了日": r0.get("モデル学習終了日")}

    # 予測の馬に結果を付ける（馬名で対応付け。馬番は出走馬名表の段階では空のため）
    h = horses.copy()
    h["馬名"] = h["馬名表示"].astype(str).str.strip()
    m = p.merge(h[["馬名", "馬番", "着順", "異常区分", "単勝オッズ", "馬体重"]].rename(
        columns={"馬番": "実際の馬番"}), on="馬名", how="left")
    m["状態"] = m["異常区分"].map(IJYO_LABELS).fillna("")
    m.loc[m["実際の馬番"].isna(), "状態"] = "結果に無い"

    box = m.loc[m["予測順位"] <= predict.BOX_SIZE]
    ran = box.loc[box["実際の馬番"].notna() & ~box["異常区分"].isin(REFUND_IJYO)]
    n = len(ran)
    invest = n * (n - 1) * (n - 2) * trifecta.BET_UNIT if n >= 3 else 0
    box_posts = {int(x) for x in ran["実際の馬番"]}
    row["BOX"] = "・".join(str(int(x)) if pd.notna(x) else "?" for x in box["実際の馬番"])
    row["返還された馬"] = "・".join(box.loc[box["異常区分"].isin(REFUND_IJYO), "馬名"])

    top = h.loc[h["着順"].isin([1, 2, 3])].sort_values("着順")
    for k in (1, 2, 3):
        row[f"{k}着"] = "/".join(str(int(x)) for x in top.loc[top["着順"] == k, "馬番"])

    combos = []
    if pay_row is not None and isinstance(pay_row.get("3連単組番一覧"), str) and pay_row["3連単組番一覧"]:
        kumis = pay_row["3連単組番一覧"].split(";")
        pays = [float(x) for x in str(pay_row["3連単払戻一覧"]).split(";")]
        combos = list(zip(kumis, pays))
    elif pay_row is not None and pd.notna(pay_row.get("3連単払戻")):
        combos = [(str(pay_row.get("3連単組番", "")), float(pay_row["3連単払戻"]))]

    hit_pay = 0.0
    for kumi, pay in combos:
        posts = {int(kumi[i:i + 2]) for i in (0, 2, 4)} if len(kumi) == 6 else set()
        if posts and posts <= box_posts:
            hit_pay += pay
    row["3連単"] = " / ".join(f"{_fmt_kumi(k)} {p:,.0f}円" for k, p in combos) if combos else "払戻なし"
    row["3連単払戻"] = combos[0][1] if combos else np.nan
    row["的中"] = bool(hit_pay > 0)
    row["払戻"] = hit_pay
    row["熱い"] = bool(hit_pay >= HOT_THRESHOLD)
    row["投資"] = invest
    row["収支"] = hit_pay - invest
    row["照合"] = "済" if combos else "払戻データなし"
    return RaceCheck(row, m)


def _fmt_kumi(kumi: str) -> str:
    if len(kumi) == 6 and kumi.isdigit():
        return f"{int(kumi[:2])}-{int(kumi[2:4])}-{int(kumi[4:6])}"
    return kumi


def reconcile(preds: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> dict:
    """予測と結果を照合する。戻り値: races（1レース1行）, horses（記録シート用）, pending, skipped。"""
    schedule = race_schedule(raw)
    final, skipped = choose_final_predictions(preds, schedule)
    if final.empty:
        return {"races": pd.DataFrame(), "horses": pd.DataFrame(), "pending": pd.DataFrame(),
                "skipped": skipped}
    results = race_results(raw, final["レースID"].unique())
    payout = jvmap.prepare_payout(raw["HR"]).set_index("レースID")
    status = schedule.set_index("レースID")["RAデータ区分"]

    races, horses, pending = [], [], []
    for (kind, rid), p in final.groupby(["区分", "レースID"], sort=False):
        h = results.loc[results["レースID"] == rid]
        if status.get(rid) == "9":
            pending.append({"区分": kind, "レースID": rid, "日付": p["日付"].iloc[0],
                            "レース名": p["レース名"].iloc[0], "状態": "レース中止"})
            continue
        if h.empty or rid not in payout.index:
            pending.append({"区分": kind, "レースID": rid, "日付": p["日付"].iloc[0],
                            "レース名": p["レース名"].iloc[0], "状態": "結果待ち"})
            continue
        chk = check_race(p, h, payout.loc[rid])
        if kind == PRACTICE and pd.notna(chk.row.get("モデル学習終了日")) \
                and pd.Timestamp(chk.row["モデル学習終了日"]) >= pd.Timestamp(chk.row["日付"]):
            chk.row["区分"] = PRACTICE + "・参考外"
        races.append(chk.row)
        horses.append(chk.horses.assign(区分=chk.row["区分"]))
    races_df = pd.DataFrame(races)
    if not races_df.empty:
        races_df = races_df.sort_values(["日付", "レースID"]).reset_index(drop=True)
    return {"races": races_df,
            "horses": pd.concat(horses, ignore_index=True) if horses else pd.DataFrame(),
            "pending": pd.DataFrame(pending), "skipped": skipped}


def summary(races: pd.DataFrame) -> pd.DataFrame:
    """区分ごとの通算成績と、validate の期待値。"""
    rows = []
    if not races.empty:
        done = races.loc[races["照合"] == "済"]
        for kind, g in done.groupby("区分", sort=False):
            n, hot = len(g), int(g["熱い"].sum())
            lo, hi = trifecta.wilson_interval(hot, n)
            rows.append({"区分": kind, "レース数": n, "的中": int(g["的中"].sum()), "熱い当たり": hot,
                         "達成率": hot / n if n else np.nan, "下限": lo, "上限": hi,
                         "投資": int(g["投資"].sum()), "払戻": float(g["払戻"].sum()),
                         "仮の収支": float(g["収支"].sum())})
    order = {LIVE: 0, PRACTICE: 1}
    rows.sort(key=lambda r: order.get(r["区分"], 2))
    rows.append({"区分": "validate の期待値（as_of=day・馬体重代用）", "達成率": EXPECTED_RATE,
                 "下限": EXPECTED_CI[0], "上限": EXPECTED_CI[1]})
    out = pd.DataFrame(rows)
    for c in ("レース数", "的中", "熱い当たり", "投資"):
        if c in out.columns:
            out[c] = out[c].astype("Int64")
    return out


def record_rows(horses: pd.DataFrame) -> pd.DataFrame:
    """「記録」シートに書く行（1頭1行）。着順は取消・中止なら文字で。"""
    h = horses.loc[~horses["区分"].astype(str).str.contains("参考外")].copy()
    if h.empty:
        return pd.DataFrame()
    h = h.sort_values(["区分"], key=lambda s: s.eq(LIVE)).drop_duplicates(["レースID", "馬名"], keep="last")
    # 着順。取消・除外・中止・失格は、その文字を入れる
    rank = [state if state in ("取消", "発走除外", "競走除外", "中止", "失格")
            else ("" if pd.isna(r) else int(r))
            for r, state in zip(h["着順"], h["状態"])]
    return pd.DataFrame({
        "レースID": h["レースID"], "日付": h["日付"], "レース名": h["レース名"],
        "馬番": h["実際の馬番"].map(lambda x: "" if pd.isna(x) else int(x)),
        "馬名": h["馬名"], "予測確率": h["予測確率"].astype(float).round(4),
        "予測順位": h["予測順位"], "着順": rank,
        "単勝オッズ": h["単勝オッズ"], "馬体重": h["馬体重"].map(lambda x: "" if pd.isna(x) else int(x)),
    }).reset_index(drop=True)
