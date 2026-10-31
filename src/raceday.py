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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


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
    plan = bets.mixed(P, Q, p, posts, margin=MARGIN, max_points=MAX_POINTS)
    heat = {int(TARGET): plan.predicted}
    for tg in HEAT_TARGETS[1:]:
        heat[int(tg)] = bets.mixed(P, Q, p, posts, target=tg, margin=MARGIN, max_points=MAX_POINTS).predicted
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
        hit = real * b.stake / 100 if np.isfinite(real) else np.nan
        need = int(math.ceil(TARGET / real) * 100) if np.isfinite(real) and real > 0 else None
        rows.append({"券種": b.kind, "組番": _combo_text(b.kind, b.combo), "馬番": list(b.combo), "金額": int(b.stake),
                     "推定配当": est, "本物のオッズ": real / 100 if np.isfinite(real) else None,
                     "当たったときの払戻": hit,
                     "届く": None if not np.isfinite(hit) else bool(hit >= TARGET),
                     "届く金額": need, "足りない額": (need - b.stake) if need and need > b.stake else 0})
    rows.sort(key=lambda r: (SMAPPY_ORDER.index(r["券種"]), r["組番"]))
    short = [r for r in rows if r["届く"] is False]
    return {"レースID": str(race["レースID"].iloc[0]), "確率の出し方": mode, "当日の結果を使えた": bool(results_ok),
            "使ったレース": used, "使ったレース数": n_used, "予測の達成確率": plan.predicted,
            "熱さ": heat, "例外を考えてもいい場面": bool(plan.predicted < EXCEPTION_RATE),
            "買い目": rows, "点数": len(rows), "合計": int(sum(r["金額"] for r in rows)),
            "本物のオッズで確かめた": T is not None,
            "届かない買い目": len(short), "足りない額の合計": int(sum(r["足りない額"] for r in short)),
            "オッズの発表": snap.get("発表", ""), "オッズの取得時刻": snap.get("取得時刻", ""),
            "オッズのデータ区分": snap.get("データ区分", ""), "オッズの出どころ": snap.get("出どころ", "速報"),
            "γ": g, "δ": d}


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
        from . import jvlink
        t0 = time.perf_counter()
        try:
            counts = jvlink.fetch_rt_results(self.date, self.rt_dir, client=self._client(),
                                             struct_module=self.struct_module, log=self.log)
            if any("JVRTOpen" in e for e in counts["エラー"]):
                # 開催日単位のキーで読めなかったときの予備：その日の全レースをレース単位のキーで
                keys = [k for k, _ in jvlink.race_keys_for_date(self.date, self.data_dir, graded_only=False)]
                if keys:
                    counts = jvlink.fetch_rt_results(self.date, self.rt_dir, client=self._client(),
                                                     struct_module=self.struct_module, keys=keys, log=self.log)
            ok = not counts["エラー"]
            msg = (f"0B12: {sum(counts['レコード'].values())} レコード" + (f"、エラー {counts['エラー']}" if counts["エラー"] else "")
                   + (f"、データなし {counts['データなし']}" if counts["データなし"] else ""))
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
        from . import jvlink
        t0 = time.perf_counter()
        try:
            c = jvlink.fetch_rt_odds(race_keys, dataspec="0B30", out_dir=self.data_dir, date=self.date,
                                     client=self._client(), log=self.log)
            msg = f"0B30: {c['レース']} レース・{c['レコード']} レコード（データなし {c['データなし']}・エラー {c['エラー']}）"
        except Exception as e:
            msg = f"0B30 の取り込みに失敗: {type(e).__name__}: {e}"
        self.timing["オッズの取得（秒）"] = round(time.perf_counter() - t0, 1)
        return msg

    def odds(self, race_id: str, now: datetime) -> dict | None:
        snap = realodds.load_race_odds(self.odds_dir, race_id)
        if snap:
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
        snap = realodds.load_race_odds(self.final_odds, race_id, final_only=True)
        if snap:
            snap["出どころ"] = "確定オッズで代用（その時点の速報が無い）"
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
            row["結果の要約"] = "発走後。最終の買い目を作れなかった（前回の買い目が残っています）"
            logs.append(row)
            continue
        snap = source.odds(rid, now)
        warn = check_announce(clock, snap, date) if clock else None
        if warn:
            clock.warnings.append(warn)
        if not snap:
            row["結果の要約"] = "単勝オッズが無い（発売前・取得の失敗）。前回の買い目のまま"
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


def render_sheet(prep: dict, replay: bool = False, clock: Clock | None = None) -> list[list]:
    date = prep["日付"]
    ddir = day_dir(date, replay)
    now = (clock.now if clock else datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    rows: list[list] = [[f"当日の買い目（F）{'【ドライラン】' if replay else ''}", sheet_text(f"{date[:4]}-{date[4:6]}-{date[6:]}"),
                         f"最終更新 {now}", "", "お金は賭けない記録（練習）。有馬記念は同じ仕組みで買う"]]
    if clock and clock.warnings:
        for w in clock.warnings:
            rows.append(["!!!!! 時計の警告 !!!!!", w])
    for r in prep["レース"]:
        rid = r["レースID"]
        st = {s: read_json(ddir / rid / f"{s}.json") for s in STAGES}
        cur = st["final"] or st["latest"]
        rows.append([])
        head = f"■ {r['競馬場名']}{int(rid[-2:])}R {r['競走名']}（{r['グレード']}）"
        if cur is None:
            rows.append([head, f"発走 {post_time(date, r['発走時刻']):%H:%M}", "状態: まだ買い目なし（発売前・オッズ待ち）"])
            continue
        state = "最終（これ以降は更新しない）" if st["final"] else f"{cur['段階']}（最終は発走{FINAL_BEFORE}分前）"
        rows.append([head, f"発走 {cur['発走']}", f"状態: {state}", f"作成 {cur['作成時刻'][11:16]}（発走 {cur['分前']:.0f} 分前）"])
        mark = "  ← 例外を考えてもいい場面（ふだんの半分の 6% 未満）" if cur["例外を考えてもいい場面"] else ""
        rows.append(["予測の達成確率（5万円以上）", f"{cur['予測の達成確率']:.1%}{mark}"])
        used = "・".join(cur["使ったレース"]) if cur["使ったレース"] else "なし"
        rows.append(["当日の傾向", cur["確率の出し方"], f"使ったレース: {used}"])
        ann = cur["オッズの発表"]
        ann = f"{ann[4:6]}:{ann[6:]} 発表" if len(str(ann)) == 8 else "発表時刻なし"
        got = f"{cur['オッズの取得時刻'][11:19]} 取得・" if cur["オッズの取得時刻"] else ""
        rows.append(["オッズ", f"{ann}（{got}{cur['オッズの出どころ']}）"])
        rows.append(["券種", "組番", "金額", "推定配当（100円）", "本物のオッズ（倍）", "当たったときの払戻", "確認"])
        for b in cur["買い目"]:
            ok = "" if b["届く"] is None else ("OK" if b["届く"] else f"届かない → {b['届く金額']:,}円で届く（+{b['足りない額']:,}円）")
            rows.append([b["券種"], sheet_text(bet_text(b)), b["金額"], _yen(b["推定配当"]),
                         "" if b["本物のオッズ"] is None else f"{b['本物のオッズ']:.1f}", _yen(b["当たったときの払戻"]), ok])
        rows.append(["合計", f"{cur['点数']}点", cur["合計"]])
        if not cur["本物のオッズで確かめた"]:
            rows.append(["確認", "本物のオッズ（馬連・馬単・3連複・3連単）が取れず、届くかを確かめられなかった"])
        elif cur["届かない買い目"]:
            total = cur["合計"] + cur["足りない額の合計"]
            note = "（予算 12,000円を超える）" if total > BUDGET else ""
            rows.append(["警告", f"本物のオッズで5万円に届かない買い目が {cur['届かない買い目']} 点。"
                                 f"全部届かせるなら +{cur['足りない額の合計']:,}円 = {total:,}円{note}"])
        heat = cur["熱さ"]
        rows.append(["熱さ（参考）", " / ".join(f"{int(k) // 10000}万円 {v:.1%}" for k, v in heat.items())])
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
    <ExecutionTimeLimit>PT15M</ExecutionTimeLimit>
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
def score_plan(plan: dict, wins: dict, scratched: set[int]) -> dict:
    """最終の買い目を実際の払戻で。取消・除外の馬を含む買い目は返還（払戻ではなく、戻ってくるお金として別に数える）。"""
    pay, refund = 0.0, 0
    for b in plan["買い目"]:
        combo = tuple(int(x) for x in b["馬番"])
        if set(combo) & scratched:
            refund += int(b["金額"])
            continue
        key = tuple(sorted(combo)) if b["券種"] in ("3連複", "馬連") else combo
        v = wins.get(b["券種"], {}).get(key)
        if v:
            pay += v * b["金額"] / 100
    invest = int(plan["合計"])
    return {"投資": invest, "払戻": pay, "返還": refund, "的中": pay > 0, "熱い": pay >= TARGET,
            "収支": pay + refund - invest}


def score_finals(raw: dict, finals: list[dict]) -> pd.DataFrame:
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
        s = score_plan(f, bets.winning_combos(payout.loc[rid]), scratched)
        top = h.loc[h["着順"].isin([1, 2, 3])].sort_values("着順")
        rows.append({**base, **s, "着順": "-".join(str(int(x)) for x in top["馬番"]), "照合": "済"})
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


def collect_finals(replay: bool = False) -> list[dict]:
    base = raceday_dir() / "replay" if replay else raceday_dir()
    out = []
    for p in sorted(base.glob("2*/*/final.json")):
        f = read_json(p)
        if f:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# コマンド
# ---------------------------------------------------------------------------
def _tomorrow() -> str:
    return (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")


def _cmd_prep(args) -> int:
    date = args.date or _tomorrow()
    p = prep_live(date, args.data, refit=args.refit)
    if p is None:
        return 1
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


def _cmd_update(args) -> int:
    if args.log_file:
        sys.stdout = sys.stderr = _Tee(Path(args.log_file))
    date = args.date or datetime.now().strftime("%Y%m%d")
    print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} update {date} =====")
    prep = read_json(day_dir(date) / "prep.json")
    if prep is None:
        print(f"前の晩の準備がありません（py -m src.raceday prep --date {date}）")
        return 1
    clock = check_clock() if not args.now else Clock(datetime.now(), _parse_now(date, args.now), None)
    for w in clock.warnings:
        print(f"!!!!! 時計の警告: {w}")
    now = clock.now
    source = LiveSource(date, args.data)
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
    if not finals:
        print("保存した最終の買い目がありません（data/raceday/<日付>/<レースID>/final.json）")
        return 1
    raw = jvmap.load_raw(args.data)
    races = score_finals(raw, finals)
    box = _box_records(raw)
    if not box.empty:
        box = box.loc[box["レースID"].astype(str).isin(set(races["レースID"]))]
        races = races.merge(box[["レースID", "払戻", "熱い"]].rename(columns={"払戻": "6頭BOXの払戻", "熱い": "6頭BOXの熱い"}),
                            on="レースID", how="left")
    summ = score_summary(races, box)
    print("=== 答え合わせ：各重賞の最終の買い目（F）を実際の払戻で ===")
    cols = [c for c in ["日付", "レース", "グレード", "着順", "点数", "投資", "払戻", "返還", "熱い", "予測の達成確率",
                        "確率の出し方", "6頭BOXの払戻", "6頭BOXの熱い", "照合"] if c in races.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(races[cols].to_string(index=False))
        print("\n=== 通算 ===")
        print(summ.to_string(index=False) if not summ.empty else "（照合済みのレースなし）")
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
    write_json(frozen_path(), {**c, "固定した日": datetime.now().strftime("%Y-%m-%d"), "元": src.name})
    print(f"係数を固定しました（{src.name} → {frozen_path().name}）。これ以降の prep はこの係数を使います")
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
    q.set_defaults(func=_cmd_prep)
    q = sub.add_parser("update", help="当日：速報 → 当日の傾向 → 買い目 → シート（タスクスケジューラから）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は今日）")
    q.add_argument("--now", default=None, help="時刻を指定して動かす（HHMM。確認用）")
    q.add_argument("--no-sheet", action="store_true", help="シートに書かない")
    q.add_argument("--log-file", default=None, help="出力をこのファイルに追記する（タスクスケジューラ用）")
    q.set_defaults(func=_cmd_update)
    q = sub.add_parser("check-results", help="保存した当日の結果（0B12）の確認：着順の確定と、当日の傾向に使えるレース（買い目は作らない）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は今日）")
    q.set_defaults(func=_cmd_check_results)
    q = sub.add_parser("sheet", help="保存した買い目からシートだけを書き直す（買い目は作り直さない）")
    q.add_argument("--date", default=None, help="YYYYMMDD（既定は今日）")
    q.add_argument("--replay", action="store_true", help="ドライランの分（シート「当日の買い目（再現）」）")
    q.set_defaults(func=_cmd_sheet)
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
    q.set_defaults(func=_cmd_score)
    q = sub.add_parser("freeze", help="係数を固定する（12月中旬）")
    q.add_argument("--source", default=None, help="固定する係数のファイル（既定は最新の coefs_*.json）")
    q.set_defaults(func=_cmd_freeze)
    sub.add_parser("unfreeze", help="係数の固定をやめる").set_defaults(func=_cmd_unfreeze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
