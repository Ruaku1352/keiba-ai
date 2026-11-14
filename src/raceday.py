"""v10：当日の仕組み（F の買い目を、毎週末の重賞すべてで自動で作る。有馬記念のリハーサル）。

    py -m src.raceday prep      --date 20261101 [--schedule]   前の晩：翌日の重賞の準備（係数・脚質）。--schedule でタスクも登録
    py -m src.raceday update    --date 20261101                当日（タスクスケジューラが10分おきに）：速報 → 当日の傾向 → 買い目 → シート
    py -m src.raceday schedule  --date 20261101 [--remove]     タスクスケジューラへの登録・削除
    py -m src.raceday replay    --date 20261025 [--at 1520]    練習（ドライラン）：保存した速報オッズ・確定データで当日の流れを再現
    py -m src.raceday score     [--sheet]                      答え合わせ：保存した最終の買い目を実際の払戻で照合（6頭BOX と並べる）
    py -m src.raceday freeze / unfreeze                        係数を固定する（12月中旬）／固定をやめる

■ F（確認のテストで採用したもの。定義は変えない）
  券種混合（3連単・3連複・馬単・馬連）× 市場（単勝オッズ）＋当日の傾向（3b）× 金額の余裕 ×1.4 × 15点まで × 12,000円。
  当日の傾向は「同じ日・同じ競馬場・同じ芝ダで、対象レースの発走の40分以上前に発走し、**その時点で結果が届いていた**レース」。
  係数（β・θ）と γ・δ は、レースの日より前の平場（2011年〜）で推定する（検証の「前の期間の平場で」と同じ考え方）。

■ 時刻のきまり
  発走 60分前の回で「仮の買い目」、20分前の回で「最終の買い目」を保存。最終のあとは、そのレースは更新しない。
  当日の結果（0B12）が取れなかったときは「市場だけ」（単勝オッズの確率そのまま）に自動で切り替え、シートにそう出す。
  失敗しても、前の晩の準備・前回の買い目（latest / provisional / final の JSON）はそのまま残る。

■ 時計の確認
  実行のたびに、インターネットの時刻（HTTP の Date）と PC の時計を比べ、5分以上ずれていたらシートとログに大きく警告し、
  発走までの時間の判定はネットの時刻で行う。速報オッズの発表時刻が PC の時計より5分以上未来のときも警告する。
"""

from __future__ import annotations

import argparse
import email.utils
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import bets, config, corner, experiment, jvmap, market, predict, preprocess, realodds, tilt, trackbias

MARGIN = 1.4
MAX_POINTS = 15
BUDGET = bets.BUDGET
TARGET = bets.TARGET
HEAT_TARGETS = (50_000, 40_000, 30_000)
PROVISIONAL_BEFORE = 60          # 分
FINAL_BEFORE = 20                # 分
INTERVAL = 10                    # 分（タスクスケジューラの間隔）
CUTOFF_MIN, DELAY_MIN = 20, 20   # 当日の傾向：発走の 20 + 20 = 40 分以上前に発走したレース（検証と同じ）
EXCEPTION_RATE = 0.06            # 予測の達成確率がこれ未満なら「例外を考えてもいい場面」
CLOCK_TOLERANCE_MIN = 5
FIT_START = "2011-01-01"
SMAPPY_ORDER = ("馬連", "馬単", "3連複", "3連単")       # 投票の画面の券種の並び（馬連 → 馬単 → 3連複 → 3連単）
SHEET_TITLE = "当日の買い目"
REPLAY_SHEET_TITLE = "当日の買い目（再現）"
SCORE_SHEET_TITLE = "当日の答え合わせ"
JST = timezone(timedelta(hours=9))
STAGES = ("latest", "provisional", "final")
STAGE_LABELS = {"latest": "途中", "provisional": "仮", "final": "最終"}


def raceday_dir() -> Path:
    d = predict.data_root() / "raceday"
    d.mkdir(parents=True, exist_ok=True)
    return d


def day_dir(date: str, replay: bool = False) -> Path:
    d = raceday_dir() / ("replay" if replay else "") / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d %H:%M:%S")
    return o


def write_json(path: Path, obj) -> None:
    """書きかけのファイルが残らないよう、別名に書いてから置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def append_csv(path: Path, rows: list[dict]) -> None:
    """行を追記する。行ごとに列がそろっていなくても、見出しの名前で列を合わせる（新しい列があれば足して書き直す）。

    v10 の最初の版は見出しを最初の1回しか書かず、列の少ない行（保存済み・発走後など）がずれて入っていた。
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if not path.exists():
        new.to_csv(path, index=False, encoding="utf-8-sig")
        return
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    extra = [c for c in new.columns if c not in header]
    if extra:
        old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        pd.concat([old, new.astype(object)], ignore_index=True)[header + extra].to_csv(path, index=False, encoding="utf-8-sig")
    else:
        new.reindex(columns=header).to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 係数（β・θ・γ・δ）
# ---------------------------------------------------------------------------
def frozen_path() -> Path:
    return raceday_dir() / "frozen_coefs.json"


def fit_coefs(ds, until: str, verbose: bool = True) -> dict:
    """until（YYYYMMDD）より前の平場（FIT_START〜）で、3b の係数と市場の γ・δ を推定する。"""
    end = pd.Timestamp(until)
    if verbose:
        print(f"[raceday] 3b の係数と γ・δ を推定中（{FIT_START} 〜 {end - pd.Timedelta(days=1):%Y-%m-%d} の平場。数分）...",
              flush=True)
    rr = ds.race_result
    terms = tilt.build_terms(ds)
    t = rr[["レースID", "血統登録番号", "レース日付", "馬番", "着順", "リステッド・重賞競走"]].copy()
    t["市場確率"] = experiment.market_probability(rr).to_numpy()
    t = t.merge(terms[["レースID", "血統登録番号"] + tilt.DAY_TERMS], on=["レースID", "血統登録番号"], how="left")
    d = pd.to_datetime(t["レース日付"])
    t = t.loc[(d >= pd.Timestamp(FIT_START)) & (d < end) & ~experiment.is_graded(t)]
    t["win"] = (pd.to_numeric(t["着順"], errors="coerce") == 1).astype(int)
    t = market._one_winner(t)
    fs, starts = market.race_slices(t)
    beta = market.fit_conditional_logit(tilt._design(fs, "市場確率", tilt.DAY_TERMS), starts, fs["win"].to_numpy())
    g, dl, n = market.fit_discount(t, "市場確率")
    return {"β": [float(b) for b in beta], "項": list(tilt.DAY_TERMS), "γ": g, "δ": dl, "推定レース数": int(len(starts) - 1),
            "期間": [FIT_START, f"{end - pd.Timedelta(days=1):%Y-%m-%d}"],
            "作成": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def load_or_fit_coefs(ds, until: str, refit: bool = False, verbose: bool = True) -> dict:
    """固定した係数があればそれを使う。無ければ until ごとに推定して保存（同じ until なら使い回す）。"""
    frozen = read_json(frozen_path())
    if frozen and not refit:
        if verbose:
            print(f"[raceday] 固定した係数を使います（{frozen_path().name}。推定期間 {frozen['期間'][0]}〜{frozen['期間'][1]}）")
        return {**frozen, "固定": True}
    path = raceday_dir() / "coefs" / f"coefs_{until}.json"
    saved = read_json(path)
    if saved and not refit:
        return saved
    c = fit_coefs(ds, until, verbose)
    write_json(path, c)
    return c


# ---------------------------------------------------------------------------
# 前の晩の準備
# ---------------------------------------------------------------------------
TARGET_COLUMNS = ["レースID", "レース日付", "競馬場コード", "競馬場名", "競走名", "リステッド・重賞競走", "芝・ダート区分",
                  "距離(m)", "発走時刻", "公式出走頭数", "馬番", "馬名表示", "血統登録番号", "脚質スコア", "着順"]


def style_scores(history: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """対象の馬の脚質スコア（過去の4角の位置の平均。検証の 3b と同じ作り方。その日より前のレースだけ）。"""
    day = pd.to_datetime(targets["レース日付"]).min()
    hist = preprocess.basic_clean(history.loc[pd.to_datetime(history["レース日付"]) < day])
    hist["_対象"] = False
    tgt = preprocess.clean_entries(predict.blank_results(targets))
    tgt["_対象"] = True
    df = pd.concat([hist, tgt], ignore_index=True)
    df = df.sort_values(["レース日付", "レースID", "馬番"], kind="mergesort").reset_index(drop=True)
    df = corner.add_running_style_features(corner.attach_corner_from_columns(df))
    t = df.loc[df["_対象"].astype(bool), ["レースID", "血統登録番号", "脚質スコア"]]
    return targets[["レースID", "血統登録番号"]].merge(t, how="left")["脚質スコア"].to_numpy()


def build_prep(ds, targets: pd.DataFrame, date: str, coefs: dict, source: str) -> dict:
    t = targets.copy()
    t["脚質スコア"] = style_scores(ds.race_result, t)
    t["着順"] = np.nan                                    # 結果は使わない（ドライランでも空にする）
    for c in TARGET_COLUMNS:
        if c not in t.columns:
            t[c] = np.nan
    races = t.drop_duplicates("レースID")
    return {"日付": date, "作成": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "出馬表": source,
            "係数": coefs, "対象": t[TARGET_COLUMNS].to_dict("records"),
            "レース": [{"レースID": str(r["レースID"]), "発走時刻": str(r["発走時刻"]), "競馬場名": r["競馬場名"],
                       "競走名": r["競走名"], "グレード": r["リステッド・重賞競走"], "芝ダ": r["芝・ダート区分"],
                       "距離": r["距離(m)"], "頭数": int((t["レースID"] == r["レースID"]).sum())}
                      for _, r in races.iterrows()]}


def prep_live(date: str, data_dir=None, refit: bool = False, verbose: bool = True) -> dict | None:
    raw = jvmap.load_raw(data_dir)
    day = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    entries = jvmap.prepare_entries(raw, from_date=day, to_date=day, graded_only=True)
    if entries.empty:
        print(f"{date} の重賞の出馬表がありません（先に `py -m src.jvlink fetch --option 1` で出馬表を取ってください）")
        return None
    if not entries["枠順確定"].all():
        print("【注意】馬番がまだそろっていないレースがあります（出走馬名表の段階）。金曜・土曜の出馬表のあとに、もう一度 prep を")
    ds = jvmap.build_dataset(data_dir)
    coefs = load_or_fit_coefs(ds, date, refit, verbose)
    p = build_prep(ds, entries, date, coefs, "出馬表（データ区分 1・2）")
    write_json(day_dir(date) / "prep.json", p)
    return p


def prep_replay(ds, date: str, refit: bool = False, verbose: bool = True) -> dict | None:
    """ドライラン用：過去のその日の重賞を、確定データの出走馬で（結果の列は使わない）。"""
    rr = ds.race_result
    d = pd.to_datetime(rr["レース日付"]) == pd.Timestamp(date)
    targets = rr.loc[d & experiment.is_graded(rr)]
    if targets.empty:
        print(f"{date} の重賞が確定データにありません")
        return None
    coefs = load_or_fit_coefs(ds, date, refit, verbose)
    p = build_prep(ds, predict.blank_results(targets), date, coefs, "確定データの出走馬（ドライラン）")
    write_json(day_dir(date, replay=True) / "prep.json", p)
    return p


def post_time(date: str, hhmm: str) -> datetime:
    s = str(hhmm).strip().zfill(4)
    return datetime(int(date[:4]), int(date[4:6]), int(date[6:]), int(s[:2]), int(s[2:]))


# ---------------------------------------------------------------------------
# 時計の確認
# ---------------------------------------------------------------------------
@dataclass
class Clock:
    pc_local: datetime
    now: datetime                    # 判定に使う日本時間（ネットの時刻が取れればそれで補正）
    offset_min: float | None         # ネットの時刻 − PC の時刻（分）
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def net_time_utc(url: str = "https://www.google.com", timeout: float = 5.0) -> datetime | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = r.headers.get("Date")
        return email.utils.parsedate_to_datetime(d).astimezone(timezone.utc) if d else None
    except Exception:
        return None


def check_clock(net=net_time_utc, pc_utc=None, pc_local=None) -> Clock:
    t0 = time.perf_counter()
    pc_utc = pc_utc or datetime.now(timezone.utc)
    pc_local = pc_local or datetime.now()
    jst_from_pc = pc_utc.astimezone(JST).replace(tzinfo=None)
    warnings = []
    server = net()
    offset = None
    if server is not None:
        offset = (server - pc_utc).total_seconds() / 60
        if abs(offset) >= CLOCK_TOLERANCE_MIN:
            warnings.append(f"PC の時計がインターネットの時刻より {offset:+.0f} 分ずれています"
                            f"（{abs(offset) / 60:.1f} 時間）。発走までの時間はインターネットの時刻で判定します。"
                            "Windows の「日付と時刻」で「時刻を自動的に設定する」→「今すぐ同期」を")
        now = server.astimezone(JST).replace(tzinfo=None) + timedelta(seconds=time.perf_counter() - t0)
    else:
        now = jst_from_pc
    if abs((pc_local - jst_from_pc).total_seconds()) >= CLOCK_TOLERANCE_MIN * 60:
        warnings.append("PC のタイムゾーンが日本（UTC+9）になっていません。Windows の「日付と時刻」でタイムゾーンを (UTC+09:00) 大阪、札幌、東京 に")
    return Clock(pc_local, now, offset, warnings)


def check_announce(clock: Clock, snap: dict | None, date: str) -> str | None:
    """速報オッズの発表時刻が、判定に使う時刻より5分以上未来なら警告（PC の時計が遅れている疑い）。"""
    if not snap or not snap.get("発表") or len(str(snap["発表"])) != 8:
        return None
    a = str(snap["発表"])
    try:
        ann = datetime(int(date[:4]), int(a[:2]), int(a[2:4]), int(a[4:6]), int(a[6:]))
    except ValueError:
        return None
    if (ann - clock.now).total_seconds() >= CLOCK_TOLERANCE_MIN * 60:
        return f"速報オッズの発表時刻 {ann:%H:%M} が、今の時刻 {clock.now:%H:%M} より先です（時計が遅れている疑い）"
    return None


# ---------------------------------------------------------------------------
# 当日の傾向・確率・買い目
# ---------------------------------------------------------------------------
def day_terms(race: pd.DataFrame, day_results: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """対象レースの馬ごとの 3b の項と、使えたレース（表示用）。day_results はその時点で届いていた当日の結果。"""
    tgt = preprocess.clean_entries(predict.blank_results(race))
    tgt["_対象"] = True
    parts = [tgt]
    if day_results is not None and not day_results.empty:
        h = preprocess.basic_clean(day_results.loc[~day_results["レースID"].isin(set(race["レースID"]))])
        h["_対象"] = False
        parts.insert(0, h)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["レース日付", "レースID", "馬番"], kind="mergesort").reset_index(drop=True)
    style = df["脚質スコア"].copy() if "脚質スコア" in df.columns else None
    if "_4コーナー順位" not in df.columns:                  # 対象レースの行だけのとき（先に終わったレースが無い）
        df["_4コーナー順位"] = np.nan
    df = corner.attach_corner_from_columns(df)
    if style is not None:
        df["脚質スコア"] = style
    df = trackbias.add_track_bias_features(df, cutoff_min=CUTOFF_MIN, delay_min=DELAY_MIN)
    out = df.loc[df["_対象"].astype(bool)].sort_values("馬番")
    used = []
    if len(parts) > 1:
        r0 = race.iloc[0]
        start = trackbias.start_minutes(pd.Series([r0["発走時刻"]])).iloc[0]
        first = parts[0].groupby("レースID", sort=False).head(1)
        same = first.loc[(first["競馬場コード"].astype(str) == str(r0["競馬場コード"]))
                         & (first["芝・ダート区分"].astype(str) == str(r0["芝・ダート区分"]))]
        ok = trackbias.start_minutes(same["発走時刻"]) <= start - (CUTOFF_MIN + DELAY_MIN)
        used = sorted(f"{int(str(x)[-2:])}R" for x in same.loc[ok.to_numpy(), "レースID"])
        used = sorted(used, key=lambda s: int(s[:-1]))
    return out, used


ORDERED_KINDS = ("馬単", "3連単")                     # 着順のある券種


def _combo_text(kind: str, combo) -> str:
    """スマッピー・即PAT と同じ形：馬連・3連複は小さい順に「-」（6-8、2-8-10）、馬単・3連単は着順の順に「→」（10→11、3→8→2）。"""
    if kind in ORDERED_KINDS:
        return "→".join(str(int(x)) for x in combo)
    return "-".join(str(int(x)) for x in sorted(combo))


def sheet_text(s: str) -> str:
    """シートに文字のまま書く（先頭の ' は表示されない）。書き込みは USER_ENTERED なので、6-8 などが日付に化けないように。"""
    return "'" + s


def bet_text(b: dict) -> str:
    """保存した買い目の組番を、券種と馬番から作り直す（前の版で保存した "3-8-2" 形式の組番も正しく表示するため）。"""
    return _combo_text(b["券種"], b["馬番"]) if b.get("馬番") else str(b["組番"])


def compute_plan(race: pd.DataFrame, snap: dict, coefs: dict, day_results: pd.DataFrame | None,
                 results_ok: bool) -> dict:
    """1レースの F の買い目。race は prep の対象の行（1頭1行）、snap は速報オッズ（realodds.load_race_odds）。"""
    race = race.copy()
    race["馬番"] = pd.to_numeric(race["馬番"], errors="coerce")
    win = snap.get("単勝", {})
    race = race.loc[race["馬番"].isin(set(win))].sort_values("馬番")          # 単勝オッズの無い馬（取消など）は除く
    if len(race) < 4:
        raise ValueError(f"単勝オッズのある馬が {len(race)} 頭しかいません")
    race["公式出走頭数"] = len(race)
    posts = race["馬番"].astype(int).to_numpy()
    inv = 1.0 / np.array([win[int(x)] for x in posts])
    q = inv / inv.sum()
    used = []
    if results_ok:
        terms, used = day_terms(race, day_results)
        x = terms.set_index("馬番").reindex(posts)
        X = np.column_stack([np.log(np.clip(q, 1e-6, None))]
                            + [x[c].astype("float64").fillna(0).to_numpy() for c in coefs["項"]])
        u = X @ np.asarray(coefs["β"])
        e = np.exp(u - u.max())
        p = e / e.sum()
        mode = "当日の傾向を使った" if used else "当日の傾向：使えるレースがまだ無い（項は 0）"
        n_used = int(x["当日_使えたレース数"].fillna(0).max()) if "当日_使えたレース数" in x.columns else 0
    else:
        p = q
        mode = "市場だけ（当日の結果を取れなかったため自動で切り替え）"
        n_used = 0
    g, d = coefs["γ"], coefs["δ"]
    P = market.trifecta_tensor(p, g, d)
    Q = market.combo_probabilities(market.trifecta_tensor(q, g, d))
    k, m = plan_settings(coefs)
    plan = bets.mixed(P, Q, p, posts, margin=k, max_points=MAX_POINTS)
    # 目標を下げて選び直した場合（参考）。5万円の買い目より悪ければ、5万円の買い目でその目標に届く確率を使う（低い目標ほど低くならないように）
    reselect = {int(TARGET): plan.predicted}
    prev = plan.predicted
    for tg in HEAT_TARGETS[1:]:
        v = max(bets.mixed(P, Q, p, posts, target=tg, margin=k, max_points=MAX_POINTS).predicted,
                bets.predicted_success(plan, P, Q, posts, tg), prev)
        reselect[int(tg)] = v
        prev = v
    T = realodds.payout_tables({k: snap[k] for k in realodds.KIND_TYPES if k in snap}, posts)
    index = {int(x): i for i, x in enumerate(posts)}
    rows = []
    for b in plan.bets:
        ix = [index[x] for x in b.combo]
        qv = Q[b.kind][tuple(ix)] if b.kind in ("3連単", "3連複") else Q[b.kind][ix[0], ix[1]]
        est = market.estimated_payout(float(qv), b.kind)
        real = np.nan
        if T is not None:
            real = T[b.kind][tuple(ix)] if b.kind in ("3連単", "3連複") else T[b.kind][ix[0], ix[1]]
        real = float(real) if np.isfinite(real) else np.nan
        rows.append({"券種": b.kind, "組番": _combo_text(b.kind, b.combo), "馬番": list(b.combo), "金額": int(b.stake),
                     "推定配当": est, "本物のオッズ": real / 100 if np.isfinite(real) and real > 0 else None})
    rows.sort(key=lambda r: (SMAPPY_ORDER.index(r["券種"]), r["組番"]))
    moved = rebalance(rows, target=TARGET * m)
    short = [r for r in rows if r["届く"] is False]
    heat = current_heat(rows, P, Q, posts, T)
    return {"レースID": str(race["レースID"].iloc[0]), "確率の出し方": mode, "当日の結果を使えた": bool(results_ok),
            "使ったレース": used, "使ったレース数": n_used, "予測の達成確率": plan.predicted,
            "熱さ": heat, "熱さ（目標を下げて選び直した場合）": reselect,
            "例外を考えてもいい場面": bool(heat[int(TARGET)] < EXCEPTION_RATE), "k": k, "m": m,
            "買い目": rows, "点数": len(rows), "合計": int(sum(r["金額"] for r in rows)),
            "本物のオッズで確かめた": T is not None, "付け替え": moved,
            "届かない買い目": len(short), "足りない額の合計": int(sum(r["足りない額"] for r in short)),
            "オッズの発表": snap.get("発表", ""), "オッズの取得時刻": snap.get("取得時刻", ""),
            "オッズのデータ区分": snap.get("データ区分", ""), "オッズの出どころ": snap.get("出どころ", "速報"),
            "γ": g, "δ": d}


def plan_settings(coefs: dict) -> tuple[float, float]:
    """最初の金額の余裕 k と付け替えの目標 m（係数の設定から。無ければ今の既定 k = 1.4・m = 1.00）。"""
    return float(coefs.get("k", MARGIN)), float(coefs.get("m", 1.0))


def current_heat(rows: list[dict], P: np.ndarray, Q: dict, posts: np.ndarray, T: dict | None) -> dict[int, float]:
    """今の買い目（付け替えたあとの金額）で、払戻が 5万・4万・3万円以上になる確率。

    当たったときの払戻は、本物のオッズがある組は本物のオッズで、無い組は推定配当で数える。
    同じ買い目で数えるので、必ず 5万 ≤ 4万 ≤ 3万 の順になる（v13 の直し。前は目標ごとに買い目を選び直していた）。
    """
    table = {}
    for kind in bets.KINDS:
        q = Q[kind]
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.where(q > 0, 100.0 * market.PAYOUT_RATE[kind] / q, np.nan)
        table[kind] = np.where(np.isfinite(T[kind]), T[kind], est) if T is not None else est
    plan = bets.Plan("今の買い目", [bets.Bet(r["券種"], tuple(int(x) for x in r["馬番"]), int(r["金額"])) for r in rows])
    return {int(tg): bets.predicted_success(plan, P, Q, posts, tg, odds=table) for tg in HEAT_TARGETS}


def stake_to_reach(odds: float, target: float = TARGET) -> int:
    """本物のオッズ（倍）で、当たったときに target に届く金額（100円単位で切り上げ）。"""
    return int(math.ceil(round(target / odds / 100, 6)) * 100)


def _check_row(r: dict) -> None:
    odds = r.get("本物のオッズ")
    if odds is None:
        r.update({"当たったときの払戻": None, "届く": None, "届く金額": None, "足りない額": 0})
        return
    need = stake_to_reach(odds)
    r.update({"当たったときの払戻": odds * r["金額"], "届く": bool(odds * r["金額"] >= TARGET - 1e-6),
              "届く金額": need, "足りない額": max(need - r["金額"], 0)})


def rebalance(rows: list[dict], target: float = TARGET) -> dict:
    """本物のオッズで 5万円に届かない買い目を、予算の中で付け替える（v11 の当日の運用ルール。F の選び方は変えない）。

    - 届かない買い目を、届く金額（100円単位で切り上げ）まで増やす。足りない額の小さい買い目から
    - 増やす分は、ほかの買い目から 100円ずつ回す。回せるのは「減らしたあとも本物のオッズで 5万円以上」の買い目だけで、
      余裕（本物のオッズでの払戻 − 5万円）が一番大きい買い目から。100円未満にはしない
    - 合計・点数は変えない。回しきれなければ回せる分だけ（残りは警告）。本物のオッズが無い買い目は対象にしない
    rows は書き換える（元の金額・付け替え の列を足す）。戻り値: {"点数": 金額が変わった買い目の数, "回した金額": 増やした額の合計}
    target：増やすときの目標と、回すときに保つ額（既定 5万円。v12 の試算で 5万円 × m に変える。「届く」の表示は 5万円のまま）
    """
    for r in rows:
        r["元の金額"] = r["金額"]
    has = [r for r in rows if r.get("本物のオッズ")]
    shorts = sorted([r for r in has if r["本物のオッズ"] * r["金額"] < target - 1e-6],
                    key=lambda r: stake_to_reach(r["本物のオッズ"], target) - r["金額"])
    for s in shorts:
        need = stake_to_reach(s["本物のオッズ"], target)
        while s["金額"] < need:
            donors = [r for r in has if r is not s and r["金額"] - 100 >= max(100, stake_to_reach(r["本物のオッズ"], target))]
            if not donors:
                break
            d = max(donors, key=lambda r: r["本物のオッズ"] * r["金額"] - target)
            d["金額"] -= 100
            s["金額"] += 100
    for r in rows:
        r["付け替え"] = r["金額"] - r["元の金額"]
        _check_row(r)
    return {"点数": sum(1 for r in rows if r["付け替え"]), "回した金額": int(sum(r["付け替え"] for r in rows if r["付け替え"] > 0))}


# ---------------------------------------------------------------------------
# 1回の実行（当日・ドライラン共通）
# ---------------------------------------------------------------------------
def load_rt_results(rt_dir) -> dict:
    """0B12 を追記した RA.csv・SE.csv を読む。

    fetch_rt_results の CSV には、ふだんの fetch の CSV にある "_seq"（取得した順）の列が無い（v10 の最初の版）。
    無ければ**ファイルの行の順**で振る（追記した順＝取得した順なので、同じ版が何回も追記されていても、後の行が新しい）。
    これからの書き出しも今のまま（列を変えると、すでに書いた CSV に追記できなくなるため）。
    """
    raw = jvmap.load_raw(rt_dir, record_types=("RA", "SE"))
    for df in raw.values():
        if "_seq" not in df.columns:
            df.insert(0, "_seq", np.arange(len(df)).astype(str))
    return raw


RESULTS_TIMEOUT = 60             # 0B12 の取り込みの時間の上限（秒）
ODDS_TIMEOUT = 30                # 0B30
ODDS31_TIMEOUT = 20              # 予備の 0B31
MAX_SAME_ERRORS = 3              # 同じエラーがこの回数続いたら、残りのレースは試さない
STALE_ODDS_MIN = 15              # この回にオッズが取れなかったとき、これより古いオッズでは買い目を作らない（分）
RUN_LIMIT_SEC = 300              # 1回の実行の上限（秒）。スリープ明けで止まった回も、これを過ぎたらすぐ終わる
NET_WAIT_SEC = 60                # ネットにつながっていないとき、つながるまで待つ上限（秒）
GAP_WARN_MIN = 15                # 前回の実行からこれ以上空いていたら「止まっていた可能性」
STALE_SHEET_MIN = 15             # シートの「最終更新から ○分」がこれを超えたら赤
JOIN_MARGIN = 5                  # 時間の上限のあと、スレッドの終わりを待つ余白（秒）


def call_with_timeout(fn, timeout: float):
    """fn を別のスレッドで動かし、timeout 秒で待つのをやめる（JV-Link の呼び出しが戻らなくても、この回は先に進む）。

    戻り値: (終わったか, 戻り値)。終わらなかったスレッドは捨てる（daemon なので、実行の終わりに一緒に終わる）。
    """
    import threading
    box: dict = {}

    def run():
        try:
            try:
                import pythoncom                            # Windows：このスレッドで COM を使えるように
                pythoncom.CoInitialize()
            except ImportError:
                pass
            box["r"] = fn()
        except BaseException as e:                         # noqa: BLE001  呼び出し元でそのまま投げ直す
            box["e"] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return False, None
    if "e" in box:
        raise box["e"]
    return True, box.get("r")


def day_results_from_raw(raw: dict, date: str) -> pd.DataFrame:
    """速報（0B12）の RA・SE から、その日の着順が確定したレース（データ区分 5〜7）の馬の行。払戻（HR）は使わない。"""
    races = jvmap.prepare_races(raw["RA"])
    races = races.loc[pd.to_datetime(races["レース日付"]) == pd.Timestamp(date)]
    if races.empty:
        return pd.DataFrame()
    return jvmap.prepare_horses(raw["SE"], races)


class LiveSource:
    """当日：JV-Link の速報（0B12 その日の結果・0B30 全券種のオッズ）。"""

    replay = False

    def __init__(self, date: str, data_dir=None, client_factory=None, struct_module=None, log=print):
        self.date, self.data_dir, self.log = date, data_dir, log
        self.client_factory = client_factory
        self.struct_module = struct_module
        self.rt_dir = day_dir(date) / "rt_results"
        self.odds_dir = realodds.odds.odds_dir(data_dir) / "rt" / date
        self.timing: dict = {}

    def _client(self):
        return self.client_factory() if self.client_factory else None

    def fetch_results(self, now: datetime) -> tuple[pd.DataFrame | None, bool, str]:
        """0B12（その日の結果）。時間の上限（RESULTS_TIMEOUT 秒）で打ち切り、取れた分だけで進める。取れなければ市場だけ。"""
        from . import jvlink
        t0 = time.perf_counter()
        if not getattr(self, "network_ok", True):
            self.timing["結果の取得（秒）"] = 0.0
            return None, False, "ネットにつながっていないので 0B12 を取らなかった（市場だけ）"
        deadline = time.time() + RESULTS_TIMEOUT

        def get(keys=None):
            return jvlink.fetch_rt_results(self.date, self.rt_dir, client=self._client(), struct_module=self.struct_module,
                                           keys=keys, log=self.log, deadline=deadline, max_same_errors=MAX_SAME_ERRORS)
        try:
            done, counts = call_with_timeout(get, RESULTS_TIMEOUT + JOIN_MARGIN)
            if not done:
                raise TimeoutError(f"0B12 が {RESULTS_TIMEOUT} 秒で終わらなかった（打ち切り）")
            codes = counts.get("エラーコード", [])
            comm = any(-499 <= c <= -400 for c in codes)
            if any("JVRTOpen" in e for e in counts["エラー"]) and not comm and not counts.get("打ち切り") \
                    and time.time() < deadline:
                # 開催日単位のキーで読めなかったときの予備：その日の全レースをレース単位のキーで
                # （通信のエラー -4xx のときはしない。v14：11/14 はここで全36レースが -413 になり長く止まった）
                keys = [k for k, _ in jvlink.race_keys_for_date(self.date, self.data_dir, graded_only=False)]
                if keys:
                    done, counts = call_with_timeout(lambda: get(keys), max(deadline - time.time(), 1) + JOIN_MARGIN)
                    if not done:
                        raise TimeoutError(f"0B12 が {RESULTS_TIMEOUT} 秒で終わらなかった（打ち切り）")
            ok = not counts["エラー"] and not counts.get("打ち切り")
            msg = (f"0B12: {sum(counts['レコード'].values())} レコード" + (f"、エラー {counts['エラー'][:3]}" if counts["エラー"] else "")
                   + (f"（ほか {len(counts['エラー']) - 3} 件）" if len(counts["エラー"]) > 3 else "")
                   + (f"、データなし {counts['データなし']}" if counts["データなし"] else "")
                   + (f"、打ち切り（{counts['打ち切り']}）" if counts.get("打ち切り") else ""))
        except Exception as e:                                 # 取り込みが失敗しても止めない（市場だけに切り替える）
            ok, msg = False, f"0B12 の取り込みに失敗: {type(e).__name__}: {e}"
        self.timing["結果の取得（秒）"] = round(time.perf_counter() - t0, 1)
        day = None
        if ok and (self.rt_dir / "RA.csv").exists() and (self.rt_dir / "SE.csv").exists():
            try:
                day = day_results_from_raw(load_rt_results(self.rt_dir), self.date)
            except Exception as e:
                ok, msg = False, msg + f" / 結果の変換に失敗: {type(e).__name__}: {e}"
        if day is not None and not day.empty:
            self._first_seen(day, now)
        return day, ok, msg

    def _first_seen(self, day: pd.DataFrame, now: datetime) -> None:
        path = day_dir(self.date) / "result_times.csv"
        seen = set(pd.read_csv(path, dtype=str, encoding="utf-8-sig")["レースID"]) if path.exists() else set()
        rows = []
        for rid, g in day.groupby("レースID", sort=False):
            if str(rid) in seen:
                continue
            post = post_time(self.date, g["発走時刻"].iloc[0])
            rows.append({"レースID": str(rid), "発走時刻": post.strftime("%H:%M"), "初めて取れた時刻": now.strftime("%H:%M:%S"),
                         "発走からの分": round((now - post).total_seconds() / 60, 1),
                         "SEデータ区分": str(g["_SEデータ区分"].iloc[0]) if "_SEデータ区分" in g.columns else ""})
        append_csv(path, rows)

    def fetch_odds(self, race_keys: list[str], now: datetime) -> str:
        """0B30（全券種）。取れなかった（エラー・データなし・失敗・時間の上限）ときは 0B31（単複枠）で単勝だけでも取る。"""
        from . import jvlink
        t0 = time.perf_counter()
        self.fallback = False
        self.got_odds = False
        if not getattr(self, "network_ok", True):
            self.timing["オッズの取得（秒）"] = 0.0
            return "ネットにつながっていないので 0B30 を取らなかった"
        ok = False

        def get(spec, limit):
            deadline = time.time() + limit
            return call_with_timeout(lambda: jvlink.fetch_rt_odds(race_keys, dataspec=spec, out_dir=self.data_dir, date=self.date,
                                                                  client=self._client(), log=self.log, deadline=deadline,
                                                                  max_same_errors=MAX_SAME_ERRORS), limit + JOIN_MARGIN)
        try:
            done, c = get("0B30", ODDS_TIMEOUT)
            if not done:
                raise TimeoutError(f"0B30 が {ODDS_TIMEOUT} 秒で終わらなかった（打ち切り）")
            msg = f"0B30: {c['レース']} レース・{c['レコード']} レコード（データなし {c['データなし']}・エラー {c['エラー']}）" + \
                (f"、打ち切り（{c['打ち切り']}）" if c.get("打ち切り") else "")
            ok = c["エラー"] == 0 and c["データなし"] == 0 and c["レース"] > 0
            self.got_odds = c["レース"] > 0 and c["レコード"] > 0
        except Exception as e:
            msg = f"0B30 の取り込みに失敗: {type(e).__name__}: {e}"
        if not ok and race_keys:
            try:
                done, c = get("0B31", ODDS31_TIMEOUT)
                if not done:
                    raise TimeoutError(f"0B31 が {ODDS31_TIMEOUT} 秒で終わらなかった（打ち切り）")
                msg += f" / 予備 0B31（単勝）: {c['レース']} レース（データなし {c['データなし']}・エラー {c['エラー']}）"
                self.fallback = c["レース"] > 0 and c["レコード"] > 0
                self.got_odds = self.got_odds or self.fallback
            except Exception as e:
                msg += f" / 予備 0B31 も失敗: {type(e).__name__}: {e}"
        self.timing["オッズの取得（秒）"] = round(time.perf_counter() - t0, 1)
        return msg

    def odds(self, race_id: str, now: datetime) -> dict | None:
        snap = realodds.load_race_odds(self.odds_dir, race_id)
        if not snap:
            return None
        if not getattr(self, "got_odds", True):
            # この回はオッズが取れなかった。保存してある古いオッズ（STALE_ODDS_MIN 分より前）では買い目を作らない（v14）
            try:
                age = (now - datetime.strptime(snap.get("取得時刻", ""), "%Y-%m-%d %H:%M:%S")).total_seconds() / 60
            except ValueError:
                age = None
            if age is None or age > STALE_ODDS_MIN:
                self.stale = f"この回はオッズが取れず、保存してあるオッズが古い（{snap.get('取得時刻', '')[11:16]} 取得）"
                return None
        if getattr(self, "fallback", False):
            # 0B30 が取れず 0B31 の単勝で作る。ほかの券種の古いオッズでは確かめない（古い値で「届く」と出さないため）
            for k in realodds.KIND_TYPES:
                snap.pop(k, None)
            snap["出どころ"] = "速報（0B31・単勝だけ。0B30 が取れなかったので予備）"
        else:
            snap["出どころ"] = "速報（0B30）"
        return snap


class ReplaySource:
    """ドライラン：その日の確定データ（結果は発走の delay 分後に届いたとみなす）と、保存した速報オッズ（無ければ確定オッズ）。"""

    replay = True

    def __init__(self, ds, date: str, data_dir=None, delay_min: float = DELAY_MIN):
        self.date, self.delay = date, delay_min
        rr = ds.race_result
        self.day = rr.loc[pd.to_datetime(rr["レース日付"]) == pd.Timestamp(date)]
        self.rt_odds = realodds.odds.odds_dir(data_dir) / "rt" / date
        self.final_odds = realodds.odds.odds_dir(data_dir)
        self.timing: dict = {}

    def fetch_results(self, now: datetime):
        if self.day.empty:
            return self.day, True, "確定データ：その日のレースなし"
        start = trackbias.start_minutes(self.day["発走時刻"])
        cut = now.hour * 60 + now.minute - self.delay
        got = self.day.loc[(start <= cut).to_numpy()]
        return got, True, f"確定データから {got['レースID'].nunique()} レース（発走の{self.delay:.0f}分後に届いたとみなす）"

    def fetch_odds(self, race_keys, now):
        return "保存したオッズを使う"

    def odds(self, race_id: str, now: datetime) -> dict | None:
        before = now.strftime("%Y-%m-%d %H:%M:%S")
        snap = realodds.load_race_odds(self.rt_odds, race_id, before=before) if self.rt_odds.exists() else None
        if snap:
            snap["出どころ"] = "保存した速報（rt-odds）"
            return snap
        snap = realodds.load_race_odds(self.final_odds, race_id, final_only=True, require_win=False)
        if snap and "単勝" in snap:
            snap["出どころ"] = "確定オッズで代用（その時点の速報が無い）"
            return snap
        # 確定オッズの単勝（O1）も無い：成績の確定単勝オッズ（SE）で代用する（ほかの券種の確定オッズがあれば確かめに使う）
        r = self.day.loc[self.day["レースID"].astype(str) == str(race_id)]
        o = pd.to_numeric(r["単勝オッズ"], errors="coerce")
        win = {int(p): float(v) for p, v in zip(pd.to_numeric(r["馬番"], errors="coerce"), o) if pd.notna(p) and v > 0}
        if not win:
            return None
        snap = {**(snap or {}), "単勝": win, "発表": "", "取得時刻": "", "データ区分": "",
                "出どころ": "成績の確定単勝オッズ（SE）で代用（O1 の保存が無い）"}
        return snap


def stage_of(now: datetime, post: datetime) -> str:
    m = (post - now).total_seconds() / 60
    if m <= 0:
        return "発走後"
    if m <= FINAL_BEFORE:
        return "final"
    if m <= PROVISIONAL_BEFORE:
        return "provisional"
    return "latest"


def run_once(prep: dict, source, now: datetime, clock: Clock | None = None, replay: bool = False,
             log=print) -> list[dict]:
    """1回分：取り込み → レースごとに買い目 → 保存（latest / provisional / final）。ログの行を返す。"""
    date = prep["日付"]
    ddir = day_dir(date, replay)
    t0 = time.perf_counter()
    targets = pd.DataFrame(prep["対象"])
    targets["レースID"] = targets["レースID"].astype(str)
    races = prep["レース"]
    pending = [r for r in races if not (ddir / r["レースID"] / "final.json").exists()
               and post_time(date, r["発走時刻"]) > now]
    logs = []
    if not pending:
        log("[raceday] 最終の買い目が全部そろっているか、発走後です。何もしません")
    day_results, results_ok, msg_r = source.fetch_results(now) if pending else (None, True, "")
    keys = [rt_key(date, r["レースID"]) for r in pending]
    msg_o = source.fetch_odds(keys, now) if pending else ""
    for r in races:
        rid = r["レースID"]
        post = post_time(date, r["発走時刻"])
        stage = stage_of(now, post)
        rdir = ddir / rid
        row = {"実行時刻": now.strftime("%Y-%m-%d %H:%M:%S"), "PCの時刻": (clock.pc_local if clock else now).strftime("%H:%M:%S"),
               "時計のずれ（分）": None if not clock or clock.offset_min is None else round(clock.offset_min, 2),
               "レースID": rid, "発走": post.strftime("%H:%M"), "分前": round((post - now).total_seconds() / 60, 1),
               "段階": STAGE_LABELS.get(stage, stage), "結果": msg_r, "オッズ": msg_o, **source.timing}
        if (rdir / "final.json").exists():
            row["結果の要約"] = "最終の買い目は保存済み（更新しない）"
            logs.append(row)
            continue
        if stage == "発走後":
            last = read_json(rdir / "latest.json")
            if last:
                row["結果の要約"] = f"発走後。最終の買い目は作れなかった（{last['作成時刻'][11:16]} の買い目が残っています）"
            elif (rdir / "backup.json").exists():
                row["結果の要約"] = "発走後。当日の買い目は一度も作れなかった（予備の買い目を表示）"
            else:
                row["結果の要約"] = "発走後。買い目なし（一度も作れなかった）"
            logs.append(row)
            continue
        snap = source.odds(rid, now)
        warn = check_announce(clock, snap, date) if clock else None
        if warn:
            clock.warnings.append(warn)
        if not snap:
            why = getattr(source, "stale", "") or "発売前・取得の失敗"
            row["結果の要約"] = f"単勝オッズが無い（{why}）。前回の買い目のまま"
            logs.append(row)
            continue
        try:
            plan = compute_plan(targets.loc[targets["レースID"] == rid], snap, prep["係数"], day_results, results_ok)
        except Exception as e:                              # そのレースだけ失敗。前回の買い目は残る
            row["結果の要約"] = f"買い目の計算に失敗: {type(e).__name__}: {e}"
            logs.append(row)
            continue
        plan.update({"作成時刻": now.strftime("%Y-%m-%d %H:%M:%S"), "発走": post.strftime("%H:%M"),
                     "分前": row["分前"], "段階": STAGE_LABELS[stage], "レース": r,
                     "時計の警告": list(clock.warnings) if clock else []})
        write_json(rdir / "latest.json", plan)
        if stage in ("provisional", "final") and not (rdir / "provisional.json").exists():
            write_json(rdir / "provisional.json", plan)
        if stage == "final":
            write_json(rdir / "final.json", plan)
        row.update({"結果の要約": f"{plan['点数']}点 {plan['合計']:,}円 予測 {plan['予測の達成確率']:.1%}",
                    "確率の出し方": plan["確率の出し方"], "使ったレース": "・".join(plan["使ったレース"]),
                    "オッズの発表": plan["オッズの発表"], "オッズの取得時刻": plan["オッズの取得時刻"],
                    "届かない買い目": plan["届かない買い目"]})
        logs.append(row)
    for row in logs:
        row["処理時間（秒）"] = round(time.perf_counter() - t0, 1)
        row["時計の警告"] = " / ".join(clock.warnings) if clock else ""
    append_csv(ddir / "log.csv", logs)
    return logs


def rt_key(date: str, race_id: str) -> str:
    """12桁のレースID（年4 場2 回2 日2 R2）と日付 → JVRTOpen のキー（年月日8 場2 回2 日2 R2）。"""
    return str(date) + str(race_id)[4:]


# ---------------------------------------------------------------------------
# シート
# ---------------------------------------------------------------------------
def _yen(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}"


def _plan_rows(cur: dict) -> list[list]:
    """1つの買い目（F の形）をシートの行に。付け替えた金額を正として並べ、元の金額も添える。"""
    rows: list[list] = []
    mark = "  ← 例外を考えてもいい場面（ふだんの半分の 6% 未満）" if cur["例外を考えてもいい場面"] else ""
    now50 = (cur.get("熱さ") or {}).get(int(TARGET), (cur.get("熱さ") or {}).get(str(int(TARGET))))
    now_txt = f"／今の買い目（付け替えたあと）{now50:.1%}" if now50 is not None else ""
    rows.append(["予測の達成確率（5万円以上）", f"選んだとき {cur['予測の達成確率']:.1%}{now_txt}{mark}"])
    used = "・".join(cur["使ったレース"]) if cur["使ったレース"] else "なし"
    rows.append(["当日の傾向", cur["確率の出し方"], f"使ったレース: {used}"])
    ann = cur["オッズの発表"]
    ann = f"{ann[4:6]}:{ann[6:]} 発表" if len(str(ann)) == 8 else "発表時刻なし"
    got = f"{cur['オッズの取得時刻'][11:19]} 取得・" if cur["オッズの取得時刻"] else ""
    rows.append(["オッズ", f"{ann}（{got}{cur['オッズの出どころ']}）"])
    moved = cur.get("付け替え") or {}
    if moved.get("点数"):
        rows.append(["付け替え", f"本物のオッズで付け替えた（{moved['点数']}点・{moved['回した金額']:,}円）。合計・点数は変えていない"])
    rows.append(["券種", "組番", "金額", "推定配当（100円）", "本物のオッズ（倍）", "当たったときの払戻", "確認", "付け替え"])
    for b in cur["買い目"]:
        ok = "" if b["届く"] is None else ("OK" if b["届く"] else f"届かない → {b['届く金額']:,}円で届く（+{b['足りない額']:,}円）")
        orig = b.get("元の金額", b["金額"])
        note = f"元 {orig:,}円 → {b['金額']:,}円" if orig != b["金額"] else ""
        rows.append([b["券種"], sheet_text(bet_text(b)), b["金額"], _yen(b["推定配当"]),
                     "" if b["本物のオッズ"] is None else f"{b['本物のオッズ']:.1f}", _yen(b["当たったときの払戻"]), ok, note])
    rows.append(["合計", f"{cur['点数']}点", cur["合計"]])
    if not cur["本物のオッズで確かめた"]:
        rows.append(["確認", "本物のオッズ（馬連・馬単・3連複・3連単）が取れず、届くかを確かめられなかった"])
    elif cur["届かない買い目"]:
        rows.append(["警告", f"本物のオッズで5万円に届かない買い目が {cur['届かない買い目']} 点（予算の中で回しきれなかった）。"
                             f"全部届かせるには あと +{cur['足りない額の合計']:,}円（予算 12,000円を超えるので、買い足すかは自分で判断）"])
    heat = cur["熱さ"]
    rows.append(["熱さ（参考）", "今の買い目で払戻が " + " / ".join(f"{int(k) // 10000}万円以上 {v:.1%}" for k, v in heat.items())])
    re = cur.get("熱さ（目標を下げて選び直した場合）")
    if re:
        rows.append(["", "目標を下げて選び直した場合 " + " / ".join(f"{int(k) // 10000}万円 {v:.1%}" for k, v in re.items())])
    return rows


def _backup_rows(bk: dict) -> list[list]:
    rows = [["予備の買い目", f"予備（{bk['作り方']}・{bk['作成'][5:16]} 作成）。当日の買い目を一度も作れなかったので、これを表示しています"]]
    if bk.get("買い目"):
        rows += _plan_rows(bk["買い目"])
    elif bk.get("BOX"):
        box = "・".join(str(x) for x in bk["BOX"])
        rows.append(["券種", "組番", "金額"])
        rows.append(["3連単", sheet_text(f"BOX {box}"), 12_000])
        rows.append(["合計", "120点（6頭BOX・1点100円）", 12_000])
    return rows


def render_sheet(prep: dict, replay: bool = False, clock: Clock | None = None) -> list[list]:
    date = prep["日付"]
    ddir = day_dir(date, replay)
    now_dt = clock.now if clock else datetime.now()
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    k, m = plan_settings(prep.get("係数", {}))
    rows: list[list] = [[f"当日の買い目（F）{'【ドライラン】' if replay else ''}", sheet_text(f"{date[:4]}-{date[4:6]}-{date[6:]}"),
                         f"最終更新 {now}", f"k = {k:.1f}・m = {m:.2f}", "お金は賭けない記録（練習）。有馬記念は同じ仕組みで買う"]]
    # v14：シートを開いたときにシートの側で計算する「最終更新から ○分」（PC が止まっていても分かる）。B2 の分が 15 を超えたら赤
    rows.append([f'=IF(B2>{STALE_SHEET_MIN},"!!!!! 最終更新から "&B2&" 分（{STALE_SHEET_MIN}分を超えた：PC が止まっている可能性） !!!!!",'
                 f'"最終更新から "&B2&" 分")',
                 f"=ROUND((NOW()-DATE({now_dt.year},{now_dt.month},{now_dt.day})-TIME({now_dt.hour},{now_dt.minute},{now_dt.second}))*1440)",
                 "分（シートを開いた時刻で計算）"])
    if clock and clock.warnings:
        for w in clock.warnings:
            rows.append(["!!!!! 時計の警告など !!!!!", w])
    gaps = read_json(ddir / "gaps.json") or []
    if gaps:
        rows.append(["PC が止まっていた可能性", " / ".join(f"{g['から']}〜{g['まで']}" for g in gaps)])
    for r in prep["レース"]:
        rid = r["レースID"]
        st = {s: read_json(ddir / rid / f"{s}.json") for s in STAGES}
        backup = read_json(ddir / rid / "backup.json")
        cur = st["final"] or st["latest"]
        post = post_time(date, r["発走時刻"])
        rows.append([])
        head = f"■ {r['競馬場名']}{int(rid[-2:])}R {r['競走名']}（{r['グレード']}）"
        if cur is None:
            if now_dt >= post - timedelta(minutes=FINAL_BEFORE) and backup:
                rows.append([head, f"発走 {post:%H:%M}", "状態: 当日の買い目は一度も作れなかった → 予備を表示"])
                rows += _backup_rows(backup)
            else:
                late = now_dt >= post
                state = "買い目なし（発走後。一度も作れなかった）" if late else "まだ買い目なし（発売前・オッズ待ち）"
                extra = "予備あり（最終の時刻までに作れなければ表示）" if backup else "予備なし"
                rows.append([head, f"発走 {post:%H:%M}", f"状態: {state}", extra])
            continue
        if st["final"]:
            state = "最終（これ以降は更新しない）"
        elif now_dt >= post - timedelta(minutes=FINAL_BEFORE):
            state = f"{cur['段階']}（最終の買い目を作れていない。{cur['作成時刻'][11:16]} の買い目のまま）"
        else:
            state = f"{cur['段階']}（最終は発走{FINAL_BEFORE}分前）"
        rows.append([head, f"発走 {cur['発走']}", f"状態: {state}", f"作成 {cur['作成時刻'][11:16]}（発走 {cur['分前']:.0f} 分前）"])
        rows += _plan_rows(cur)
        if st["provisional"] and st["provisional"]["作成時刻"] != cur["作成時刻"]:
            pv = st["provisional"]
            rows.append([f"仮の買い目（{pv['作成時刻'][11:16]}）",
                         sheet_text(" / ".join(f"{b['券種']} {bet_text(b)} {b['金額']:,}円" for b in pv["買い目"])),
                         f"{pv['点数']}点 {pv['合計']:,}円 予測 {pv['予測の達成確率']:.1%}"])
    return rows


def write_sheet(prep: dict, replay: bool = False, clock: Clock | None = None, settings=None, ws=None) -> str:
    from . import sheets
    values = render_sheet(prep, replay, clock)
    title = REPLAY_SHEET_TITLE if replay else SHEET_TITLE
    if ws is None:
        settings = settings or sheets.load_settings()
        ws = sheets.open_or_add_worksheet(settings, title)
    sheets.write_grid(ws, values)
    try:
        sheets.ensure_red_rule(ws, "A2:E2", f"=$B$2>{STALE_SHEET_MIN}")
    except Exception:
        pass                                              # 色付けに失敗しても、文字の警告は出ている
    return title


# ---------------------------------------------------------------------------
# タスクスケジューラ
# ---------------------------------------------------------------------------
TASK_PREFIX = "keiba_raceday_"


def task_xml(date: str, prep: dict, python: str, workdir: str, start_hhmm: str = "0900") -> str:
    """タスクの定義（XML）。10分おき＋各レースの発走60分前・20分前ちょうど。ログオン中だけ動く（管理者権限は不要）。"""
    posts = [post_time(date, r["発走時刻"]) for r in prep["レース"]]
    start = post_time(date, start_hhmm)
    end = max(posts) if posts else start + timedelta(hours=8)
    trig = [f"""    <TimeTrigger>
      <StartBoundary>{start:%Y-%m-%dT%H:%M:%S}</StartBoundary>
      <EndBoundary>{end:%Y-%m-%dT%H:%M:%S}</EndBoundary>
      <Enabled>true</Enabled>
      <Repetition><Interval>PT{INTERVAL}M</Interval><Duration>PT{max(int((end - start).total_seconds() // 60), 10)}M</Duration><StopAtDurationEnd>true</StopAtDurationEnd></Repetition>
    </TimeTrigger>"""]
    for p in posts:
        for before in (PROVISIONAL_BEFORE, FINAL_BEFORE):
            t = p - timedelta(minutes=before)
            trig.append(f"""    <TimeTrigger><StartBoundary>{t:%Y-%m-%dT%H:%M:%S}</StartBoundary><Enabled>true</Enabled></TimeTrigger>""")
    log = Path(workdir) / "data" / "raceday" / date / "run.log"
    args = f"-m src.raceday update --date {date} --log-file &quot;{log}&quot;"
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>競馬 当日の買い目（F）{date}。py -m src.raceday schedule --date {date} --remove で削除</Description></RegistrationInfo>
  <Triggers>
{chr(10).join(trig)}
  </Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>Queue</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT6M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>{python}</Command><Arguments>{args}</Arguments><WorkingDirectory>{workdir}</WorkingDirectory></Exec>
  </Actions>
</Task>
"""


def _pythonw() -> str:
    """コンソールの窓を出さない pythonw.exe（あれば）。"""
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


REPO = Path(__file__).resolve().parents[1]


def install_task(date: str, prep: dict, run=None, start_hhmm: str = "0900"):
    """タスクを登録する。戻り値: (タスク名, XML のパス, schtasks の結果)。"""
    import subprocess
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
    xml = task_xml(date, prep, _pythonw(), str(REPO), start_hhmm)
    path = day_dir(date) / "task.xml"
    path.write_text(xml, encoding="utf-16")
    name = TASK_PREFIX + date
    return name, path, run(["schtasks", "/create", "/tn", name, "/xml", str(path), "/f"])


def remove_task(date: str, run=None):
    import subprocess
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
    return run(["schtasks", "/delete", "/tn", TASK_PREFIX + date, "/f"])


# ---------------------------------------------------------------------------
# 答え合わせ
# ---------------------------------------------------------------------------
def score_plan(plan: dict, wins: dict, scratched: set[int], stake: str = "金額") -> dict:
    """最終の買い目を実際の払戻で。取消・除外の馬を含む買い目は返還（払戻ではなく、戻ってくるお金として別に数える）。

    stake="金額" は付け替えたあと（実際に買う方）、"元の金額" は付け替える前（参考）。
    """
    pay, refund = 0.0, 0
    for b in plan["買い目"]:
        amount = int(b.get(stake, b["金額"]))
        combo = tuple(int(x) for x in b["馬番"])
        if set(combo) & scratched:
            refund += amount
            continue
        key = tuple(sorted(combo)) if b["券種"] in ("3連複", "馬連") else combo
        v = wins.get(b["券種"], {}).get(key)
        if v:
            pay += v * amount / 100
    invest = int(sum(int(b.get(stake, b["金額"])) for b in plan["買い目"]))
    return {"投資": invest, "払戻": pay, "返還": refund, "的中": pay > 0, "熱い": pay >= TARGET,
            "収支": pay + refund - invest}


def odds_drift_rows(plan: dict, final_snap: dict | None) -> list[dict]:
    """最終の買い目（20分前ごろのオッズで確かめたもの）を、確定オッズでも 5万円に届いていたか（付け替えたあとの金額で）。"""
    out = []
    if not final_snap:
        return out
    posts = sorted({int(x) for b in plan["買い目"] for x in b["馬番"]} | set(final_snap.get("単勝", {})))
    T = realodds.payout_tables({k: final_snap[k] for k in realodds.KIND_TYPES if k in final_snap}, np.array(posts))
    if T is None:
        return out
    index = {p: i for i, p in enumerate(posts)}
    for b in plan["買い目"]:
        ix = [index[int(x)] for x in b["馬番"]]
        v = T[b["券種"]][tuple(ix)] if b["券種"] in ("3連単", "3連複") else T[b["券種"]][ix[0], ix[1]]
        if not np.isfinite(v):
            continue
        conf = float(v) / 100
        out.append({"レースID": plan["レースID"], "券種": b["券種"], "組番": bet_text(b), "金額": b["金額"],
                    "元の金額": b.get("元の金額", b["金額"]), "付け替えた": bool(b.get("付け替え")),
                    "20分前のオッズ": b.get("本物のオッズ"), "確定オッズ": conf,
                    "確定÷20分前": conf / b["本物のオッズ"] if b.get("本物のオッズ") else None,
                    "20分前で届く": b.get("届く"), "確定で届く": bool(conf * b["金額"] >= TARGET - 1e-6)})
    return out


def score_finals(raw: dict, finals: list[dict], data_dir=None, drift: list | None = None) -> pd.DataFrame:
    """drift にリストを渡すと、確定オッズ（fetch-odds で保存したもの）とのずれの行を足していく。"""
    from . import results
    ids = [f["レースID"] for f in finals]
    res = results.race_results(raw, ids)
    payout = jvmap.prepare_payout(raw["HR"]).set_index("レースID")
    rows = []
    for f in finals:
        rid = f["レースID"]
        r = f.get("レース", {})
        base = {"レースID": rid, "日付": f["作成時刻"][:10], "レース": f"{r.get('競馬場名', '')}{int(rid[-2:])}R {r.get('競走名', '')}",
                "グレード": r.get("グレード", ""), "最終の作成時刻": f["作成時刻"][11:16], "確率の出し方": f["確率の出し方"],
                "点数": f["点数"], "予測の達成確率": f["予測の達成確率"]}
        h = res.loc[res["レースID"] == rid]
        if h.empty or rid not in payout.index:
            rows.append({**base, "照合": "結果待ち"})
            continue
        scratched = {int(x) for x in h.loc[h["異常区分"].isin(results.REFUND_IJYO), "馬番"].dropna()}
        wins = bets.winning_combos(payout.loc[rid])
        s = score_plan(f, wins, scratched)
        orig = score_plan(f, wins, scratched, stake="元の金額")
        moved = f.get("付け替え") or {}
        top = h.loc[h["着順"].isin([1, 2, 3])].sort_values("着順")
        rows.append({**base, **s, "着順": "-".join(str(int(x)) for x in top["馬番"]),
                     "付け替えた点数": moved.get("点数", 0), "元の金額での払戻": orig["払戻"], "元の金額での熱い": orig["熱い"],
                     "照合": "済"})
        if drift is not None:
            drift += odds_drift_rows(f, realodds.load_race_odds(realodds.odds.odds_dir(data_dir), rid, final_only=True,
                                                                 require_win=False))
    return pd.DataFrame(rows)


def score_summary(races: pd.DataFrame, box: pd.DataFrame | None = None) -> pd.DataFrame:
    from . import trifecta
    out = []
    done = races.loc[races["照合"] == "済"] if not races.empty else races
    if not done.empty:
        n, hot = len(done), int(done["熱い"].sum())
        lo, hi = trifecta.wilson_interval(hot, n)
        out.append({"買い方": "F（券種混合・市場＋3b・余裕×1.4・15点まで）", "レース数": n, "熱い当たり": hot,
                    "達成率": hot / n, "下限": lo, "上限": hi, "的中": int(done["的中"].sum()),
                    "投資": int(done["投資"].sum()), "払戻": float(done["払戻"].sum()), "返還": int(done["返還"].sum()),
                    "回収率": float((done["払戻"].sum() + done["返還"].sum()) / done["投資"].sum()),
                    "予測の達成確率（平均）": float(done["予測の達成確率"].mean())})
        if "元の金額での熱い" in done.columns:
            hot0 = int(done["元の金額での熱い"].fillna(False).astype(bool).sum())
            lo0, hi0 = trifecta.wilson_interval(hot0, n)
            out.append({"買い方": "（参考）F の付け替える前の金額", "レース数": n, "熱い当たり": hot0, "達成率": hot0 / n,
                        "下限": lo0, "上限": hi0, "払戻": float(done["元の金額での払戻"].sum())})
    if box is not None and not box.empty:
        b = box.loc[box["照合"] == "済"]
        if not b.empty:
            n, hot = len(b), int(b["熱い"].sum())
            lo, hi = trifecta.wilson_interval(hot, n)
            out.append({"買い方": "今の6頭BOX（predict --sheet の最終の予測。同じレース）", "レース数": n, "熱い当たり": hot,
                        "達成率": hot / n, "下限": lo, "上限": hi, "的中": int(b["的中"].sum()),
                        "投資": int(b["投資"].sum()), "払戻": float(b["払戻"].sum()),
                        "回収率": float(b["払戻"].sum() / b["投資"].sum()) if b["投資"].sum() else np.nan})
    return pd.DataFrame(out)


REPLAY_LABEL = "ドライラン（本番の自動実行ではない）"


def collect_finals(replay: bool = False, dates=None) -> list[dict]:
    """保存した最終の買い目。replay=True ならドライランの分（dates で日付を絞る）。"""
    base = raceday_dir() / "replay" if replay else raceday_dir()
    out = []
    for p in sorted(base.glob("2*/*/final.json")):
        if dates is not None and p.parent.parent.name not in set(dates):
            continue
        f = read_json(p)
        if f:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# コマンド
# ---------------------------------------------------------------------------
def _tomorrow() -> str:
    return (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")


def record_box_prediction(date: str, data_dir=None, run=None) -> int:
    """今の6頭BOX の予想を、ふだんの `py -m src.v6 predict --date … --sheet` と同じ形で記録する（predict の動きは変えずに呼ぶだけ）。"""
    from . import v6
    run = run or v6.main
    argv = (["--data", str(data_dir)] if data_dir else []) + \
        ["predict", "--date", f"{date[:4]}-{date[4:6]}-{date[6:]}", "--sheet"]
    try:
        return int(run(argv) or 0)
    except SystemExit as e:                                # シートの設定が無いなど。CSV は保存済み
        print(f"  predict --sheet の途中で止まりました（予想の CSV は保存されています）: {e}")
        return 1
    except Exception as e:
        print(f"  predict に失敗しました: {type(e).__name__}: {e}")
        return 1


def latest_box(date: str) -> dict[str, list[int]]:
    """その日の重賞の、いちばん新しい predict の予想（CSV）の上位6頭（レースID → 馬番）。"""
    files = sorted(predict.prediction_dir().glob(f"predict_{date}_*.csv"))
    if not files:
        return {}
    t = pd.read_csv(files[-1], dtype={"レースID": str}, encoding="utf-8-sig")
    t = t.loc[pd.to_numeric(t["予測順位"], errors="coerce") <= predict.BOX_SIZE]
    out = {}
    for rid, g in t.groupby("レースID"):
        posts = pd.to_numeric(g.sort_values("予測順位")["馬番"], errors="coerce").dropna().astype(int).tolist()
        if len(posts) >= 3:
            out[str(rid)] = posts
    return out


def make_backups(prep: dict, data_dir=None, source=None, now: datetime | None = None) -> dict[str, str]:
    """前の晩に、予備の買い目を作って <レースID>/backup.json に保存する（当日の買い目を一度も作れなかったときに表示）。

    前の晩に単勝の速報オッズ（前日発売）が取れれば、そのオッズで F（当日の傾向なし）。
    取れなければ、今の6頭BOX の予想（predict の CSV）。どちらも無ければ予備なし。
    """
    date = prep["日付"]
    now = now or datetime.now()
    targets = pd.DataFrame(prep["対象"])
    targets["レースID"] = targets["レースID"].astype(str)
    src = source or LiveSource(date, data_dir, log=lambda *a, **k: None)
    try:
        src.fetch_odds([rt_key(date, r["レースID"]) for r in prep["レース"]], now)
    except Exception:
        pass
    box = latest_box(date)
    out = {}
    for r in prep["レース"]:
        rid = r["レースID"]
        bk = None
        try:
            snap = src.odds(rid, now)
        except Exception:
            snap = None
        if snap and snap.get("単勝"):
            try:
                plan = compute_plan(targets.loc[targets["レースID"] == rid], snap, prep["係数"], None, results_ok=False)
                plan["確率の出し方"] = "前の晩のオッズ（当日の傾向なし・市場だけ）"
                plan.update({"作成時刻": now.strftime("%Y-%m-%d %H:%M:%S"), "発走": post_time(date, r["発走時刻"]).strftime("%H:%M"),
                             "段階": "予備", "レース": r})
                when = snap.get("発表") or ""
                when = f"{when[4:6]}:{when[6:]} 発表" if len(when) == 8 else (snap.get("取得時刻") or "")[5:16]
                bk = {"作り方": f"前の晩のオッズ（{when}）で F", "買い目": plan}
            except Exception:
                bk = None
        if bk is None and rid in box:
            bk = {"作り方": "今のモデルの6頭BOX（前の晩にオッズが無かったため）", "BOX": box[rid]}
        if bk:
            bk["作成"] = now.strftime("%Y-%m-%d %H:%M:%S")
            write_json(day_dir(date) / rid / "backup.json", bk)
            out[rid] = bk["作り方"]
    return out


def cleanup_old_tasks(today: str, run=None) -> list[str]:
    """過去の日付のタスク（keiba_raceday_YYYYMMDD で、その日が過ぎたもの）を削除する。"""
    import re
    import subprocess
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
    r = run(["schtasks", "/query", "/fo", "csv", "/nh"])
    names = sorted(set(re.findall(r"(" + TASK_PREFIX + r"(\d{8}))", getattr(r, "stdout", "") or "")))
    removed = []
    for name, d in names:
        if d < today and getattr(run(["schtasks", "/delete", "/tn", name, "/f"]), "returncode", 1) == 0:
            removed.append(name)
    return removed


def _cmd_prep(args) -> int:
    date = args.date or _tomorrow()
    p = prep_live(date, args.data, refit=args.refit)
    if p is None:
        return 1
    if args.k is not None or args.m is not None:          # 練習用に、固定せずに k・m を変える
        if args.k is not None:
            p["係数"]["k"] = args.k
        if args.m is not None:
            p["係数"]["m"] = args.m
        write_json(day_dir(date) / "prep.json", p)
    k, m = plan_settings(p["係数"])
    print(f"最初の金額の余裕 k = {k:.2f}・付け替えの目標 m = {m:.2f}{'（今の既定）' if (k, m) == (MARGIN, 1.0) else ''}")
    try:
        day_schedule(date, args.data)                     # v12：当日、全レースの20分前ごろのオッズを集めるための一覧
    except Exception as e:
        print(f"  （全レースの一覧を作れませんでした。当日に作り直します: {type(e).__name__}）")
    if args.schedule and not args.no_predict:
        print(f"\n[raceday] 今の6頭BOX の予想を記録します（py -m src.v6 predict --date … --sheet と同じ）...")
        record_box_prediction(date, args.data)
    if not args.no_backup:
        made = make_backups(p, args.data)
        print(f"\n[raceday] 予備の買い目: " + (" / ".join(f"{rid[-2:]}R {how}" for rid, how in made.items()) if made
                                         else "作れませんでした（前の晩のオッズも 6頭BOX の予想も無い）"))
    c = p["係数"]
    print(f"{date} の重賞 {len(p['レース'])} レースの準備を保存しました: {day_dir(date) / 'prep.json'}")
    for r in p["レース"]:
        print(f"  {r['競馬場名']}{int(r['レースID'][-2:])}R {r['競走名']}（{r['グレード']}）発走 {r['発走時刻'][:2]}:{r['発走時刻'][2:]}"
              f"  {r['頭数']}頭")
    print(f"係数: β {', '.join(f'{b:.4f}' for b in c['β'])}  γ {c['γ']:.2f}  δ {c['δ']:.2f}"
          f"（{c['期間'][0]}〜{c['期間'][1]} の平場 {c['推定レース数']:,} レース{'・固定' if c.get('固定') else ''}）")
    if args.schedule:
        return _schedule(date, p, args)
    print(f"タスクの登録: py -m src.raceday schedule --date {date}")
    return 0


def _schedule(date: str, prep: dict, args) -> int:
    if sys.platform != "win32":
        xml = task_xml(date, prep, _pythonw(), str(REPO), args.start)
        path = day_dir(date) / "task.xml"
        path.write_text(xml, encoding="utf-16")
        print(f"Windows ではないので登録はしません（定義だけ保存: {path}）")
        return 0
    removed = cleanup_old_tasks(datetime.now().strftime("%Y%m%d"))
    if removed:
        print(f"過去の日付のタスクを削除しました: {', '.join(removed)}")
    r = install_task(date, prep, start_hhmm=args.start)
    if r[2].returncode != 0:
        print(f"タスクの登録に失敗しました: {r[2].stderr or r[2].stdout}")
        print("「アクセスが拒否されました」なら、PowerShell を「管理者として実行」して同じコマンドを")
        return 1
    print(f"タスク「{r[0]}」を登録しました（{args.start[:2]}:{args.start[2:]} から10分おき＋各重賞の発走60分前・20分前）。"
          f"確認: schtasks /query /tn {r[0]}  削除: py -m src.raceday schedule --date {date} --remove")
    return 0


def _cmd_schedule(args) -> int:
    date = args.date or _tomorrow()
    if args.remove:
        if sys.platform != "win32":
            print("Windows ではないので何もしません")
            return 0
        r = remove_task(date)
        print("削除しました" if r.returncode == 0 else f"削除に失敗: {r.stderr or r.stdout}")
        return r.returncode
    prep = read_json(day_dir(date) / "prep.json")
    if prep is None:
        print(f"先に py -m src.raceday prep --date {date} を")
        return 1
    return _schedule(date, prep, args)


class _Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, s):
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.f.flush()


# ---------------------------------------------------------------------------
# v14：止まらない・眠らない・止まったら分かる
# ---------------------------------------------------------------------------
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
KEEPAWAKE_AFTER_MIN = 10          # 最後の発走のこの分あとまで、待機中のスリープを防ぐ


def start_watchdog(limit: float = RUN_LIMIT_SEC, clock=time.time, exit_fn=None, interval: float = 2.0, log=print):
    """壁の時計（スリープ中も進む）で、実行が limit 秒を超えたらプロセスを終える。スリープ明けに止まっていた回もすぐ終わるので、
    待たされていた次の回（発走60分前・20分前など）がすぐ動く。"""
    import os
    import threading
    exit_fn = exit_fn or (lambda code: os._exit(code))
    t0 = clock()
    stop = threading.Event()

    def watch():
        while not stop.wait(interval):
            if clock() - t0 > limit:
                try:
                    log(f"!!!!! 1回の実行の上限（{limit:.0f} 秒）を超えたので打ち切ります"
                        f"（スリープ明けか、取り込みが止まった。保存した買い目はそのまま）")
                    sys.stdout.flush()
                except Exception:
                    pass
                exit_fn(3)
                return
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    return stop


def keep_system_awake(on: bool = True) -> bool:
    """Windows に「システムを起こしておく」を求める（このプロセスが終わるか、on=False で戻す）。画面は消えてよい。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
        return True
    except Exception:
        return False


def keepawake_path(date: str) -> Path:
    return day_dir(date) / "keepawake.json"


def ensure_keepawake(date: str, prep: dict, now: datetime, spawn=None) -> bool:
    """重賞のある日、最後の発走の 10 分あとまで待機中のスリープを防ぐ小さなプロセスを、動いていなければ起こす。"""
    posts = [post_time(date, r["発走時刻"]) for r in prep.get("レース", [])]
    if not posts:
        return False
    until = max(posts) + timedelta(minutes=KEEPAWAKE_AFTER_MIN)
    if now >= until:
        return False
    hb = read_json(keepawake_path(date))
    if hb:
        try:
            if (datetime.now() - datetime.strptime(hb["heartbeat"], "%Y-%m-%d %H:%M:%S")).total_seconds() < 120:
                return False                                    # もう動いている
        except (KeyError, ValueError):
            pass
    if spawn is None:
        if sys.platform != "win32":
            return False
        import subprocess
        DETACHED, NEW_GROUP = 0x00000008, 0x00000200

        def spawn(argv):
            subprocess.Popen([_pythonw()] + argv, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=DETACHED | NEW_GROUP, close_fds=True)
    spawn(["-m", "src.raceday", "keepawake", "--date", date, "--until", until.strftime("%H%M")])
    write_json(keepawake_path(date), {"heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "起こした": "update",
                                      "まで": until.strftime("%H:%M")})
    return True


def _cmd_keepawake(args) -> int:
    """待機中のスリープを防ぐ（最後の発走の 10 分あとまで）。update が起こす。ふたを閉じたときのスリープは防げない。"""
    until = post_time(args.date, args.until)
    keep_system_awake(True)
    while datetime.now() < until:
        write_json(keepawake_path(args.date), {"heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                               "まで": until.strftime("%H:%M")})
        time.sleep(30)
    keep_system_awake(False)
    return 0


def wait_for_network(probe=None, max_wait: float = NET_WAIT_SEC, step: float = 10, sleep=time.sleep) -> bool:
    """インターネットにつながるまで、最大 max_wait 秒待つ（スリープ明けの Wi-Fi がまだつながっていないとき）。"""
    probe = probe or (lambda: net_time_utc() is not None or net_time_utc("https://www.jra.go.jp") is not None)
    waited = 0.0
    while True:
        if probe():
            return True
        if waited >= max_wait:
            return False
        sleep(step)
        waited += step


def note_gap(date: str, now: datetime) -> str | None:
    """前回の実行から GAP_WARN_MIN 分以上空いていたら、止まっていた可能性として記録して返す。"""
    path = day_dir(date) / "last_run.json"
    last = read_json(path)
    write_json(path, {"実行時刻": now.strftime("%Y-%m-%d %H:%M:%S")})
    if not last:
        return None
    try:
        t = datetime.strptime(last["実行時刻"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError):
        return None
    if (now - t).total_seconds() < GAP_WARN_MIN * 60:
        return None
    msg = f"PC が眠っていた／止まっていた可能性（{t:%H:%M}〜{now:%H:%M}。{(now - t).total_seconds() / 60:.0f} 分あいた）"
    gaps = read_json(day_dir(date) / "gaps.json") or []
    write_json(day_dir(date) / "gaps.json", gaps + [{"から": t.strftime("%H:%M"), "まで": now.strftime("%H:%M")}])
    return msg


def _cmd_update(args) -> int:
    if args.log_file:
        sys.stdout = sys.stderr = _Tee(Path(args.log_file))
    date = args.date or datetime.now().strftime("%Y%m%d")
    print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} update {date} =====")
    prep = read_json(day_dir(date) / "prep.json")
    if prep is None:
        print(f"前の晩の準備がありません（py -m src.raceday prep --date {date}）")
        return 1
    watchdog = start_watchdog()                           # v14：1回の実行は RUN_LIMIT_SEC 秒まで（スリープ明けで止まった回もすぐ終わる）
    keep_system_awake(True)                               # v14：この回の実行中は PC を眠らせない
    try:
        return _update_body(args, date, prep)
    finally:
        watchdog.set()
        keep_system_awake(False)


def _update_body(args, date: str, prep: dict) -> int:
    clock = check_clock() if not args.now else Clock(datetime.now(), _parse_now(date, args.now), None)
    network_ok = True
    if not args.now and clock.offset_min is None:
        print("  インターネットの時刻が取れません。ネットにつながるのを待ちます（最大 60 秒）...")
        network_ok = wait_for_network()
        if network_ok:
            clock = check_clock()
        else:
            clock.warnings.append("ネットにつながっていない（60 秒待ってもつながらなかった）。この回は速報を取らずに、前の買い目のままにしています")
    gap = note_gap(date, clock.now)
    if gap:
        clock.warnings.append(gap)
    for w in clock.warnings:
        print(f"!!!!! 警告: {w}")
    now = clock.now
    if not args.now:
        try:
            ensure_keepawake(date, prep, now)
        except Exception as e:
            print(f"  （待機中のスリープを防ぐ仕組みを動かせませんでした: {type(e).__name__}: {e}）")
    source = LiveSource(date, args.data)
    source.network_ok = network_ok
    logs = run_once(prep, source, now, clock)
    if logs and logs[0].get("結果"):
        print(f"  当日の結果: {logs[0]['結果']}（取得 {source.timing.get('結果の取得（秒）', '')}秒）")
    if logs and logs[0].get("オッズ"):
        print(f"  オッズ: {logs[0]['オッズ']}（取得 {source.timing.get('オッズの取得（秒）', '')}秒）")
    for row in logs:
        print(f"  {row['レースID']} 発走 {row['発走']}（{row['分前']:.0f}分前・{row['段階']}）: {row.get('結果の要約', '')}"
              f"  [{row.get('確率の出し方', '')}]  当日の傾向に使ったレース: {row.get('使ったレース') or 'なし'}")
    if not args.no_sheet:
        try:
            print(f"  シート「{write_sheet(prep, clock=clock)}」に書きました")
        except SystemExit as e:
            print(f"  シートに書けませんでした: {e}")
        except Exception as e:
            print(f"  シートに書けませんでした（保存した買い目は残っています）: {type(e).__name__}: {e}")
    if not args.no_drift and network_ok:
        # v12：重賞の処理とシートが終わったあとに、20分前ごろのオッズを集める（失敗しても止めない。v14：30 秒まで）
        try:
            done, c = call_with_timeout(lambda: collect_drift_odds(date, now, args.data), 30)
            if not done:
                raise TimeoutError("30 秒で終わらなかった")
            if c["レース"]:
                print(f"  20分前ごろのオッズを集めました: {c['レース']} レース・{c.get('レコード', 0)} レコード（{c.get('秒', '')}秒）")
        except Exception as e:
            print(f"  20分前ごろのオッズを集められませんでした（買い目には影響なし）: {type(e).__name__}: {e}")
            try:
                append_csv(day_dir(date) / "drift_errors.csv", [{"実行時刻": now.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 "エラー": f"{type(e).__name__}: {e}"}])
            except Exception:
                pass
    return 0


# ---------------------------------------------------------------------------
# v12 2：20分前ごろのオッズのデータ集め（全レース。本来の処理のあとに、失敗しても止めない）
# ---------------------------------------------------------------------------
DRIFT_WINDOW = (15, 25)          # 発走まで 15〜25 分のレース


def drift_dir(data_dir, date: str) -> Path:
    """集めた20分前ごろのオッズの保存先（重賞の買い目に使う rt とは別にする）。"""
    return realodds.odds.odds_dir(data_dir) / "rt_drift" / date


def day_schedule(date: str, data_dir=None) -> pd.DataFrame:
    """その日の全レース（レースID・JVRTOpen のキー・発走）。初回だけ RA.csv から作って data/raceday/<日付>/all_races.csv に保存。"""
    path = day_dir(date) / "all_races.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    from . import results
    sch = results.race_schedule(jvmap.load_raw(data_dir, record_types=("RA",)))
    sch = sch.loc[pd.to_datetime(sch["レース日付"]) == pd.Timestamp(date)]
    t = pd.DataFrame({"レースID": sch["レースID"].astype(str), "キー": [rt_key(date, r) for r in sch["レースID"].astype(str)],
                      "発走": pd.to_datetime(sch["発走日時"]).dt.strftime("%H%M")}).drop_duplicates("レースID")
    t.to_csv(path, index=False, encoding="utf-8-sig")
    return t


def collect_drift_odds(date: str, now: datetime, data_dir=None, fetch=None, schedule: pd.DataFrame | None = None) -> dict:
    """発走まで 15〜25 分のレース（まだ取っていないもの）の 0B30 を取って保存する。各レース1回。"""
    from . import jvlink
    fetch = fetch or (lambda keys: jvlink.fetch_rt_odds(keys, dataspec="0B30", out_dir=data_dir, date=date,
                                                        log=lambda *a, **k: None, deadline=time.time() + 20,
                                                        max_same_errors=MAX_SAME_ERRORS))
    sch = schedule if schedule is not None else day_schedule(date, data_dir)
    done_path = day_dir(date) / "drift_collected.csv"
    done = set(pd.read_csv(done_path, dtype=str, encoding="utf-8-sig")["レースID"]) if done_path.exists() else set()
    rows = []
    for _, r in sch.iterrows():
        m = (post_time(date, r["発走"]) - now).total_seconds() / 60
        if DRIFT_WINDOW[0] <= m <= DRIFT_WINDOW[1] and r["レースID"] not in done:
            rows.append({"レースID": r["レースID"], "キー": r["キー"], "分前": round(m, 1)})
    if not rows:
        return {"レース": 0}
    t0 = time.perf_counter()
    c = fetch([x["キー"] for x in rows])
    # fetch_rt_odds は rt/<日付> に書くので、集めた分は rt_drift/<日付> に移す（重賞の買い目の読み込みと混ぜない）
    moved = []
    dst = drift_dir(data_dir, date)
    dst.mkdir(parents=True, exist_ok=True)
    for f in c.get("保存ファイル", []):
        f = Path(f)
        if f.exists():
            target = dst / f.name.replace("_0B30_", "_drift_")
            f.replace(target)
            moved.append(str(target))
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    append_csv(done_path, [{**x, "取得時刻": stamp, "レコード": c.get("レコード", "")} for x in rows])
    return {"レース": len(rows), "レコード": c.get("レコード", 0), "秒": round(time.perf_counter() - t0, 1), "保存": moved}


def drift_pairs(data_dir=None, since: str | None = None) -> pd.DataFrame:
    """集めた20分前ごろのオッズと確定オッズ（fetch-odds で保存）を、全部の組で比べる（1組1行）。"""
    base = realodds.odds.odds_dir(data_dir) / "rt_drift"
    parts = []
    for d in sorted(base.glob("2*")) if base.exists() else []:
        if since and d.name < since:
            continue
        ids = set()
        for f in d.glob("O*_*.npz"):
            with np.load(f, allow_pickle=False) as z:
                ids |= set(map(str, z["race_id"]))
        for rid in sorted(ids):
            pre = realodds.load_race_odds(d, rid, require_win=False)
            fin = realodds.load_race_odds(realodds.odds.odds_dir(data_dir), rid, final_only=True, require_win=False)
            if not pre or not fin:
                continue
            for kind in realodds.KIND_TYPES:
                if kind not in pre or kind not in fin:
                    continue
                a = pd.DataFrame({"k": [tuple(x) for x in pre[kind][0]], "pre": pre[kind][1] / 100})
                b = pd.DataFrame({"k": [tuple(x) for x in fin[kind][0]], "fin": fin[kind][1] / 100})
                m = a.merge(b, on="k")
                m = m.loc[(m["pre"] > 0) & (m["fin"] > 0)]
                if m.empty:
                    continue
                parts.append(pd.DataFrame({"日付": d.name, "レースID": rid, "券種": kind, "20分前のオッズ": m["pre"].to_numpy(),
                                           "確定オッズ": m["fin"].to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=["日付", "レースID", "券種", "20分前のオッズ", "確定オッズ"])
    t = pd.concat(parts, ignore_index=True)
    t["r"] = t["確定オッズ"] / t["20分前のオッズ"]           # 20分前のオッズ = 確定オッズ ÷ r
    t["帯"] = pd.cut(t["20分前のオッズ"], realodds.ODDS_BINS, labels=realodds.ODDS_BIN_LABELS, right=False).astype(str)
    stake = np.ceil(np.round(TARGET / t["20分前のオッズ"] / 100, 6)) * 100
    t["確定でも届く"] = t["確定オッズ"] * stake >= TARGET - 1e-6
    return t


def drift_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, band), g in pairs.groupby(["券種", "帯"], sort=False):
        lr = np.log(g["r"])
        rows.append({"券種": kind, "20分前のオッズの帯": band, "組の数": len(g), "レース数": g["レースID"].nunique(),
                     "r の中央値（確定÷20分前）": float(g["r"].median()), "r の第1四分位": float(g["r"].quantile(0.25)),
                     "r の第3四分位": float(g["r"].quantile(0.75)), "log r の平均": float(lr.mean()),
                     "log r の標準偏差": float(lr.std()), "確定でも届く割合": float(g["確定でも届く"].mean())})
    order = {k: i for i, k in enumerate(realodds.KIND_TYPES)}
    band = {b: i for i, b in enumerate(realodds.ODDS_BIN_LABELS)}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["券種", "20分前のオッズの帯"], key=lambda s: s.map(order) if s.name == "券種" else s.map(band)) \
        .reset_index(drop=True)


def drift_samples_path() -> Path:
    return raceday_dir() / "drift_samples.csv.gz"


def _cmd_drift(args) -> int:
    pairs = drift_pairs(args.data, args.since)
    if pairs.empty:
        print("集めた20分前ごろのオッズと確定オッズの組がありません"
              "（当日の自動実行で data\\jvlink\\odds\\rt_drift に集め、月曜の fetch-odds で確定オッズを保存してから）")
        return 1
    t = drift_table(pairs)
    print(f"=== 20分前ごろのオッズ → 確定オッズ（{pairs['日付'].nunique()} 日・{pairs['レースID'].nunique()} レース・"
          f"{len(pairs):,} 組。r = 確定 ÷ 20分前） ===")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(t.to_string(index=False))
    print("    確定でも届く割合 = 20分前のオッズで5万円に届く金額（100円単位で切り上げ）を買ったとき、確定オッズでも5万円以上になる割合")
    t.to_csv(raceday_dir() / "drift_table.csv", index=False, encoding="utf-8-sig")
    pairs[["日付", "レースID", "券種", "帯", "r"]].to_csv(drift_samples_path(), index=False, encoding="utf-8-sig",
                                                     compression="gzip")
    print(f"\n保存: {raceday_dir() / 'drift_table.csv'}・{drift_samples_path()}（py -m src.v9 rebalance --drift data で使う）")
    return 0


def check_results(date: str, prep: dict | None, rt_dir: Path, now: datetime | None = None) -> tuple[pd.DataFrame, list]:
    """保存した 0B12 を読んで、レースごとの版と着順の確定、対象の重賞の当日の傾向に使えるレース（買い目は作らない）。"""
    raw = load_rt_results(rt_dir)
    ra = jvmap.central_only(jvmap.dedupe_records(raw["RA"]))
    ra = ra.loc[(ra["id.Year"] + ra["id.MonthDay"]) == date]
    races = jvmap.map_races(ra)
    day = day_results_from_raw(raw, date)
    done = set(day["レースID"]) if not day.empty else set()
    rows = []
    for _, r in races.sort_values(["競馬場コード", "レースID"]).iterrows():
        rid = r["レースID"]
        top = day.loc[day["レースID"] == rid].sort_values("着順").head(3) if rid in done else pd.DataFrame()
        rows.append({"レースID": rid, "場": r["競馬場名"], "R": int(str(rid)[-2:]), "発走": f"{str(r['発走時刻'])[:2]}:{str(r['発走時刻'])[2:]}",
                     "芝ダ": r["芝・ダート区分"], "データ区分（最新の版）": r["_RAデータ区分"],
                     "着順の確定": "済" if rid in done else "まだ（速報の途中）",
                     "1〜3着": "-".join(str(int(x)) for x in top["馬番"]) if len(top) else ""})
    table = pd.DataFrame(rows)
    usable = []
    for t in (prep or {}).get("レース", []):
        post = post_time(date, t["発走時刻"])
        limit = post - timedelta(minutes=CUTOFF_MIN + DELAY_MIN)
        tgt = pd.DataFrame(prep["対象"])
        tgt = tgt.loc[tgt["レースID"].astype(str) == t["レースID"]]
        _, used = day_terms(tgt, day) if not tgt.empty else (None, [])
        usable.append({"対象": f"{t['競馬場名']}{int(t['レースID'][-2:])}R {t['競走名']}", "発走": f"{post:%H:%M}",
                       "使える条件": f"同じ場・{t['芝ダ']}・{limit:%H:%M} までに発走し着順が確定",
                       "使えるレース（今の時点）": "・".join(used) or "なし"})
    return table, usable


def _cmd_check_results(args) -> int:
    date = args.date or datetime.now().strftime("%Y%m%d")
    rt = day_dir(date) / "rt_results"
    if not (rt / "RA.csv").exists() or not (rt / "SE.csv").exists():
        print(f"その日の 0B12 の保存がありません: {rt}")
        return 1
    table, usable = check_results(date, read_json(day_dir(date) / "prep.json"), rt)
    print(f"=== {date} の当日の結果（0B12 を保存したもの。{rt}） ===")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False) if not table.empty else "（レースなし）")
    print("\n=== 対象の重賞の当日の傾向に使えるレース ===")
    for u in usable:
        print(f"  {u['対象']}（発走 {u['発走']}）: {u['使えるレース（今の時点）']}   条件: {u['使える条件']}")
    if not usable:
        print("  （prep.json が無いか、対象の重賞なし）")
    return 0


def _cmd_sheet(args) -> int:
    """保存した買い目（latest / provisional / final の JSON）から、シートだけを書き直す。買い目は作り直さない。"""
    date = args.date or datetime.now().strftime("%Y%m%d")
    prep = read_json(day_dir(date, args.replay) / "prep.json")
    if prep is None:
        print(f"{date} の prep.json がありません（{day_dir(date, args.replay)}）")
        return 1
    print(f"シート「{write_sheet(prep, replay=args.replay)}」を、保存した買い目から書き直しました（買い目は作り直していません）")
    return 0


def _parse_now(date: str, hhmm: str) -> datetime:
    return post_time(date, hhmm.replace(":", ""))


def replay_day(ds, date: str, data_dir=None, at: str | None = None, refit: bool = False, log=print) -> dict | None:
    """その日の流れを再現する。at を渡せばその時刻の1回だけ、無ければ朝から10分おき＋発走60分前・20分前の回。"""
    import shutil
    ddir = day_dir(date, replay=True)
    for p in ddir.iterdir():
        if p.is_dir():
            shutil.rmtree(p)                                   # 前回のドライランの買い目は消して作り直す
    (ddir / "log.csv").unlink(missing_ok=True)
    prep = prep_replay(ds, date, refit)
    if prep is None:
        return None
    source = ReplaySource(ds, date, data_dir)
    if at:
        times = [_parse_now(date, at)]
    else:
        posts = [post_time(date, r["発走時刻"]) for r in prep["レース"]]
        t = post_time(date, "0900")
        times = set()
        while t <= max(posts):
            times.add(t)
            t += timedelta(minutes=INTERVAL)
        for p in posts:
            times |= {p - timedelta(minutes=PROVISIONAL_BEFORE), p - timedelta(minutes=FINAL_BEFORE)}
        times = sorted(times)
    for now in times:
        run_once(prep, source, now, None, replay=True, log=lambda *_: None)
    return prep


def _cmd_replay(args) -> int:
    ds = jvmap.build_dataset(args.data)
    prep = replay_day(ds, args.date, args.data, args.at, args.refit)
    if prep is None:
        return 1
    ddir = day_dir(args.date, replay=True)
    log = pd.read_csv(ddir / "log.csv", encoding="utf-8-sig", dtype={"レースID": str})
    print(f"=== ドライラン {args.date}（{'時刻 ' + args.at if args.at else '朝9時から10分おき＋発走60分前・20分前'}）===")
    for r in prep["レース"]:
        rid = r["レースID"]
        g = log.loc[log["レースID"] == rid]
        print(f"\n■ {r['競馬場名']}{int(rid[-2:])}R {r['競走名']}（{r['グレード']}）発走 {r['発走時刻'][:2]}:{r['発走時刻'][2:]}")
        for _, x in g.iterrows():
            v = {k: (str(x[k]) if k in x and pd.notna(x[k]) else "") for k in ("結果の要約", "確率の出し方", "使ったレース")}
            if v["結果の要約"]:
                extra = f"  [{v['確率の出し方']}・使ったレース: {v['使ったレース'] or 'なし'}]" if v["確率の出し方"] else ""
                print(f"  {x['実行時刻'][11:16]} {x['段階']:<3} {v['結果の要約']}{extra}")
        f = read_json(ddir / rid / "final.json")
        if f:
            print("  最終の買い目: " + " / ".join(f"{b['券種']} {bet_text(b)} {b['金額']:,}円" for b in f["買い目"]))
            print(f"  オッズ: {f['オッズの出どころ']}  届かない買い目 {f['届かない買い目']}")
    finals = [read_json(ddir / r["レースID"] / "final.json") for r in prep["レース"]]
    finals = [f for f in finals if f]
    if finals:
        sc = score_finals(jvmap.load_raw(args.data), finals)
        print("\n=== 答え合わせ（ドライラン。確定データ） ===")
        cols = [c for c in ["レース", "着順", "点数", "投資", "払戻", "返還", "熱い", "予測の達成確率"] if c in sc.columns]
        print(sc[cols].to_string(index=False))
    if args.sheet:
        print(f"シート「{write_sheet(prep, replay=True)}」に書きました")
    return 0


def _box_records(raw) -> pd.DataFrame:
    from . import results
    try:
        preds = results.load_predictions()
    except Exception:
        return pd.DataFrame()
    if preds is None or preds.empty:
        return pd.DataFrame()
    rec = results.reconcile(preds, raw)["races"]
    return rec.loc[rec["区分"] == results.LIVE] if not rec.empty else rec


def _cmd_score(args) -> int:
    finals = collect_finals()
    rdates = [d for d in (args.replay_dates or "").split(",") if d.strip()]
    replays = collect_finals(replay=True, dates=rdates) if rdates else []
    live_ids = {f["レースID"] for f in finals}
    replays = [f for f in replays if f["レースID"] not in live_ids]      # 本番の買い目があるレースは本番を使う
    if not finals and not replays:
        print("保存した最終の買い目がありません（data/raceday/<日付>/<レースID>/final.json）")
        return 1
    raw = jvmap.load_raw(args.data)
    drift: list = []
    races = score_finals(raw, finals, args.data, drift) if finals else pd.DataFrame()
    if not races.empty:
        races.insert(0, "区分", "本番（自動実行）")
    rep_races = score_finals(raw, replays, args.data) if replays else pd.DataFrame()
    if not rep_races.empty:
        rep_races.insert(0, "区分", REPLAY_LABEL)
        races = pd.concat([races, rep_races], ignore_index=True)
    box = _box_records(raw)
    if not box.empty:
        box = box.loc[box["レースID"].astype(str).isin(set(races["レースID"]))]
        races = races.merge(box[["レースID", "払戻", "熱い"]].rename(columns={"払戻": "6頭BOXの払戻", "熱い": "6頭BOXの熱い"}),
                            on="レースID", how="left")
    summ = score_summary(races.loc[races["区分"] != REPLAY_LABEL], box)
    if not rep_races.empty:
        rs = score_summary(races.loc[races["区分"] == REPLAY_LABEL])
        rs["買い方"] = f"【{REPLAY_LABEL}】" + rs["買い方"].astype(str)
        summ = pd.concat([summ, rs], ignore_index=True)
    print("=== 答え合わせ：各重賞の最終の買い目（F）を実際の払戻で ===")
    if rdates:
        print(f"    ドライランの分（{', '.join(rdates)}）は「{REPLAY_LABEL}」として分けて数える（本番の通算には入れない）")
    cols = [c for c in ["区分", "日付", "レース", "グレード", "着順", "点数", "投資", "払戻", "返還", "熱い", "予測の達成確率",
                        "付け替えた点数", "元の金額での払戻", "確率の出し方", "6頭BOXの払戻", "6頭BOXの熱い", "照合"]
            if c in races.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(races[cols].to_string(index=False))
        print("\n=== 通算 ===")
        print(summ.to_string(index=False) if not summ.empty else "（照合済みのレースなし）")
    dr = pd.DataFrame(drift)
    print("\n=== 参考：20分前ごろのオッズ → 確定オッズ（最終の買い目。付け替えたあとの金額で、確定オッズでも 5万円に届いたか） ===")
    if dr.empty:
        print("（確定オッズの保存がありません。月曜の fetch のあと `py -m src.jvlink fetch-odds` を）")
    else:
        ratio = pd.to_numeric(dr["確定÷20分前"], errors="coerce")
        print(f"買い目 {len(dr)} 点：確定÷20分前 の中央値 {ratio.median():.3f}（四分位 {ratio.quantile(0.25):.3f}〜{ratio.quantile(0.75):.3f}）"
              f"／20分前で届く {int(dr['20分前で届く'].fillna(False).astype(bool).sum())} 点のうち 確定でも届く "
              f"{int((dr['20分前で届く'].fillna(False).astype(bool) & dr['確定で届く']).sum())} 点")
        moved = dr.loc[dr["付け替えた"]]
        if len(moved):
            print(f"付け替えた買い目 {len(moved)} 点のうち、確定オッズでも届いた {int(moved['確定で届く'].sum())} 点")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(dr.to_string(index=False))
        dr.to_csv(raceday_dir() / "score_odds_drift.csv", index=False, encoding="utf-8-sig")
    out = raceday_dir() / "score_races.csv"
    races.to_csv(out, index=False, encoding="utf-8-sig")
    summ.to_csv(raceday_dir() / "score_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n保存: {out}")
    if args.sheet:
        from . import sheets
        ws = sheets.open_or_add_worksheet(sheets.load_settings(), SCORE_SHEET_TITLE)
        shown = races[cols].astype(object).where(races[cols].notna(), "")
        for c in ("着順", "日付"):                          # 7-6-4・2026-11-01 が日付に化けないように文字で
            if c in shown.columns:
                shown[c] = [sheet_text(str(v)) if v != "" else "" for v in shown[c]]
        values = [list(summ.columns)] + summ.astype(object).where(summ.notna(), "").values.tolist() + [[]] + \
                 [cols] + shown.values.tolist()
        sheets.write_grid(ws, values)
        print(f"シート「{SCORE_SHEET_TITLE}」に書きました")
    return 0


def _cmd_freeze(args) -> int:
    src = Path(args.source) if args.source else None
    if src is None:
        files = sorted((raceday_dir() / "coefs").glob("coefs_*.json"))
        if not files:
            print("推定済みの係数がありません（先に prep）")
            return 1
        src = files[-1]
    c = read_json(src)
    k = args.k if args.k is not None else float(c.get("k", MARGIN))
    m = args.m if args.m is not None else float(c.get("m", 1.0))
    write_json(frozen_path(), {**c, "k": k, "m": m, "固定した日": datetime.now().strftime("%Y-%m-%d"), "元": src.name})
    print(f"係数を固定しました（{src.name} → {frozen_path().name}。最初の金額の余裕 k = {k:.2f}・付け替えの目標 m = {m:.2f}）。"
          "これ以降の prep はこの係数を使います")
    return 0


def _cmd_unfreeze(args) -> int:
    frozen_path().unlink(missing_ok=True)
    print("係数の固定をやめました（prep のたびに、その日より前の平場で推定し直します）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m src.raceday")
    parser.add_argument("--data", default=None, help="JV-Data CSV の場所（既定: data/jvlink）")
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prep", help="前の晩：翌日の重賞の準備（出馬表の取り込みのあとに）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は明日）")
    q.add_argument("--refit", action="store_true", help="保存・固定した係数を使わず推定し直す")
    q.add_argument("--schedule", action="store_true", help="タスクスケジューラにも登録する")
    q.add_argument("--start", default="0900", help="当日の10分おきの開始時刻（既定 0900）")
    q.add_argument("--no-predict", action="store_true", help="--schedule のときの 6頭BOX の予想の記録（predict --sheet）をしない")
    q.add_argument("--no-backup", action="store_true", help="予備の買い目を作らない")
    q.add_argument("--k", type=float, default=None, help="この日だけ最初の金額の余裕 k を変える（既定は固定した値か 1.4）")
    q.add_argument("--m", type=float, default=None, help="この日だけ付け替えの目標 m を変える（既定は固定した値か 1.00）")
    q.set_defaults(func=_cmd_prep)
    q = sub.add_parser("update", help="当日：速報 → 当日の傾向 → 買い目 → シート（タスクスケジューラから）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は今日）")
    q.add_argument("--now", default=None, help="時刻を指定して動かす（HHMM。確認用）")
    q.add_argument("--no-sheet", action="store_true", help="シートに書かない")
    q.add_argument("--log-file", default=None, help="出力をこのファイルに追記する（タスクスケジューラ用）")
    q.add_argument("--no-drift", action="store_true", help="20分前ごろのオッズ（全レース）を集めない")
    q.set_defaults(func=_cmd_update)
    q = sub.add_parser("check-results", help="保存した当日の結果（0B12）の確認：着順の確定と、当日の傾向に使えるレース（買い目は作らない）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は今日）")
    q.set_defaults(func=_cmd_check_results)
    q = sub.add_parser("sheet", help="保存した買い目からシートだけを書き直す（買い目は作り直さない）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は今日）")
    q.add_argument("--replay", action="store_true", help="ドライランの分（シート「当日の買い目（再現）」）")
    q.set_defaults(func=_cmd_sheet)
    q = sub.add_parser("keepawake", help="（update が起こす）最後の発走のあとまで待機中のスリープを防ぐ")
    q.add_argument("--date", required=True)
    q.add_argument("--until", required=True, help="HHMM")
    q.set_defaults(func=_cmd_keepawake)
    q = sub.add_parser("schedule", help="タスクスケジューラへの登録（--remove で削除）")
    q.add_argument("--date", default=None)
    q.add_argument("--remove", action="store_true")
    q.add_argument("--start", default="0900")
    q.set_defaults(func=_cmd_schedule)
    q = sub.add_parser("replay", help="練習（ドライラン）：過去の日の流れを再現")
    q.add_argument("--date", required=True, help="YYYYMMDD")
    q.add_argument("--at", default=None, help="この時刻の1回だけ（HHMM）。省略で朝から10分おき")
    q.add_argument("--refit", action="store_true")
    q.add_argument("--sheet", action="store_true", help="シート「当日の買い目（再現）」にも書く")
    q.set_defaults(func=_cmd_replay)
    q = sub.add_parser("score", help="答え合わせ：最終の買い目を実際の払戻で（月曜の fetch のあと）")
    q.add_argument("--sheet", action="store_true", help="シート「当日の答え合わせ」にも書く")
    q.add_argument("--replay-dates", default=None,
                   help="ドライランの最終の買い目も、本番とは分けて数える日付（カンマ区切り。例 20261114）")
    q.set_defaults(func=_cmd_score)
    q = sub.add_parser("drift", help="集めた20分前ごろのオッズと確定オッズのずれの表（月曜の fetch-odds のあと）")
    q.add_argument("--since", default=None, help="この日以降（YYYYMMDD）")
    q.set_defaults(func=_cmd_drift)
    q = sub.add_parser("freeze", help="係数を固定する（12月中旬）")
    q.add_argument("--source", default=None, help="固定する係数のファイル（既定は最新の coefs_*.json）")
    q.add_argument("--k", type=float, default=None, help="最初の金額の余裕 k（既定 1.4）")
    q.add_argument("--m", type=float, default=None, help="付け替えの目標 m（5万円 × m。既定 1.00）")
    q.set_defaults(func=_cmd_freeze)
    sub.add_parser("unfreeze", help="係数の固定をやめる").set_defaults(func=_cmd_unfreeze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
