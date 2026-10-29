"""v8c：馬のレーティング（能力が時間とともに変わることを扱う。検証用。本番の特徴量には入れていない）。

2つの方式：逐次の Plackett-Luce（ベイズ。下の説明）と、AtCoder 型（パフォーマンスの重み付き平均。ファイルの後半）。

■ 方式：Weng & Lin (2011) のベイズ近似による逐次の Plackett-Luce（OpenSkill の PlackettLuce と同じ式）
  各馬の強さを正規分布 N(μ, σ²) で持ち、レースの着順（全頭）を見るたびに閉じた式で更新する。
  - なぜこれか：多頭数の着順をそのまま扱える（TrueSkill は多頭数だと因子グラフの反復が要り遅い。
    Elo を多頭数にするのは「全ペアの1対1」に分ける近似で、18頭なら153組の勝敗を独立に扱ってしまう）。
    不確かさ σ を持つので「出走数が少ない馬ほど不確か」が自然に出る。強い相手（μ が高い馬）に先着すると大きく上がるので、
    レースの格は手で定義しなくても入る
  - 能力の変化：前走からの日数に応じて σ² を増やす（σ² += τ² × 日数/30。上限は初期値 σ0²）。
    休み明けの馬ほど「今の強さ」が不確かになり、次のレースで大きく動く
  - **着差（タイム差）は使わず、着順だけ**にした。着差は距離・ペース・馬場で大きさの意味が変わり、
    使うにはその補正の設定が要る（設定を選ぶこと自体が試行の数を増やす）。Plackett-Luce は着順のモデルなので、そのまま使える
  - 設定（μ0=25, σ0=25/3, β=σ0/2, τ=σ0/50 /30日, κ=1e-4）は TrueSkill・OpenSkill の既定値に近いものを**決め打ち**し、調整していない
  - 同着は着順の並び（馬番順）で切る（同着はまれで、影響は小さい）

■ リーク対策（as_of=day）
  日付ごとに、その日のレースの特徴量（レース前のレーティング）を先に全部記録してから、その日の結果で更新する。
  したがって、あるレースに使うレーティングは前日までに確定したレースだけで更新したもの（tests/test_v8_ratings.py）。

■ 特徴量（市場の確率への傾きの項）
  レーティング_相対   = μ − そのレースの出走馬の μ の平均
  レーティング_伸び   = μ − 3走前のレース前の μ（3走未満は 0）
  レーティング_不確かさ = σ − そのレースの出走馬の σ の平均
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MU0 = 25.0
SIGMA0 = 25.0 / 3
BETA = SIGMA0 / 2
TAU_30 = SIGMA0 / 50          # 30日あたりの能力の変化の大きさ
KAPPA = 1e-4
GROWTH_RUNS = 3
RATING_TERMS = ["レーティング_相対", "レーティング_伸び", "レーティング_不確かさ"]


def update_race(mu: np.ndarray, sig2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1レース分の更新。mu・sig2 は着順の順（1着が先頭）。戻り値は更新後の (μ, σ²)。"""
    c = np.sqrt(np.sum(sig2 + BETA ** 2))
    e = np.exp((mu - mu.max()) / c)                   # 定数を引いても比は同じ（あふれ防止）
    S = np.cumsum(e[::-1])[::-1]                      # S_q = Σ_{自分より下の着順（自分を含む）} e
    inv = np.cumsum(1.0 / S)                          # Σ_{q ≤ i} 1/S_q
    inv2 = np.cumsum(1.0 / S ** 2)
    omega = 1.0 - e * inv                             # Σ_{q≤i} [q=i] − e_i/S_q
    delta = e * inv - e ** 2 * inv2                   # Σ_{q≤i} (e_i/S_q)(1 − e_i/S_q)
    sigma = np.sqrt(sig2)
    new_mu = mu + sig2 / c * omega
    gamma = sigma / c
    new_sig2 = sig2 * np.maximum(1.0 - gamma * sig2 / c ** 2 * delta, KAPPA)
    return new_mu, new_sig2


def compute_ratings(race_result: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """全レースを日付順に処理し、各出走のレース前のレーティングを返す（1頭1行）。

    戻り値の列: レースID, 血統登録番号, μ, σ, 出走数（それまで）, μ_3走前, と RATING_TERMS
    """
    df = race_result[["レースID", "レース日付", "血統登録番号", "着順", "馬番"]].copy()
    df["_rank"] = pd.to_numeric(df["着順"], errors="coerce").fillna(99)
    df["_post"] = pd.to_numeric(df["馬番"], errors="coerce").fillna(99)
    df = df.sort_values(["レース日付", "レースID", "_rank", "_post"], kind="mergesort").reset_index(drop=True)
    horse = pd.factorize(df["血統登録番号"])[0]
    day = (pd.to_datetime(df["レース日付"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    H = horse.max() + 1 if len(horse) else 0
    mu = np.full(H, MU0)
    sig2 = np.full(H, SIGMA0 ** 2)
    last = np.full(H, -1)
    runs = np.zeros(H, dtype=np.int64)
    hist = np.full((H, GROWTH_RUNS), np.nan)          # 直近のレース前の μ（リングバッファ）

    n = len(df)
    out_mu, out_sig, out_runs, out_prev = (np.empty(n), np.empty(n), np.empty(n, dtype=np.int64), np.empty(n))
    race_codes = pd.factorize(df["レースID"])[0]
    starts = np.flatnonzero(np.r_[True, race_codes[1:] != race_codes[:-1]])
    bounds = np.r_[starts, n]
    day_of_race = day[starts]
    # 日付ごとに：その日の全レースのレース前の値を記録 → その日の結果で更新
    day_starts = np.flatnonzero(np.r_[True, day_of_race[1:] != day_of_race[:-1]])
    day_bounds = np.r_[day_starts, len(starts)]
    for di in range(len(day_starts)):
        r0, r1 = day_bounds[di], day_bounds[di + 1]
        pending = []
        for r in range(r0, r1):
            a, b = bounds[r], bounds[r + 1]
            h = horse[a:b]
            d = day[a]
            gap = np.where(last[h] >= 0, d - last[h], 0)
            s2 = np.minimum(sig2[h] + TAU_30 ** 2 * gap / 30.0, SIGMA0 ** 2)   # 休んだ分だけ不確かに
            out_mu[a:b] = mu[h]
            out_sig[a:b] = np.sqrt(s2)
            out_runs[a:b] = runs[h]
            out_prev[a:b] = hist[h, runs[h] % GROWTH_RUNS]                    # 3走前のレース前の μ
            pending.append((a, b, h, s2))
        for a, b, h, s2 in pending:
            new_mu, new_s2 = update_race(mu[h], s2)
            hist[h, runs[h] % GROWTH_RUNS] = mu[h]
            mu[h], sig2[h] = new_mu, new_s2
            last[h] = day[a]
            runs[h] += 1
        if verbose and di % 2000 == 0:
            print(f"    {di:,} / {len(day_starts):,} 日", flush=True)

    res = pd.DataFrame({"レースID": df["レースID"].to_numpy(), "血統登録番号": df["血統登録番号"].to_numpy(),
                        "μ": out_mu, "σ": out_sig, "出走数": out_runs, "μ_3走前": out_prev})
    g = res.groupby("レースID")
    res["レーティング_相対"] = res["μ"] - g["μ"].transform("mean")
    res["レーティング_伸び"] = np.where(res["出走数"] >= GROWTH_RUNS, res["μ"] - res["μ_3走前"], 0.0)
    res["レーティング_不確かさ"] = res["σ"] - g["σ"].transform("mean")
    return res


# ===========================================================================
# 2つ目の方式：AtCoder 型（パフォーマンスの重み付き平均）
# ===========================================================================
# ■ AtCoder のレーティング（2020年からの方式）と同じ考え方
#   1. パフォーマンス：そのレースの出走馬の「平均パフォーマンス（APerf）」から、
#        Σ_j 1 / (1 + 6^((X − APerf_j) / 400)) = 着順 − 0.5
#      となる X（自分を含む全頭で。二分探索）。初出走の馬の APerf は 1200（AtCoder の ARC・AGC の初参加と同じ）
#   2. レーティング（補正前）= 800 × log2( Σ 2^(perf_i/800) × 0.9^i / Σ 0.9^i )（i = 1 が一番新しいレース）
#   3. 出走数が少ないうちの割り引き：f(n) = (sqrt(1 − 0.81^n) / (1 − 0.9^n) − 1) / (sqrt(19) − 1) × 1200 を引く
#      （1走で 1200、2走で約 750、10走で約 160、だんだん 0 に）。出走 0 回の馬のレーティングは 0（AtCoder の未参加と同じ）
#   APerf = Σ perf_i × 0.9^i / Σ 0.9^i（パフォーマンスの重み付き平均）
#   0.9・1200・400・800 は AtCoder の値をそのまま使い、調整していない。AtCoder の「レーティングが 400 未満の表示の変換」は
#   表示のためのものなので使わない
#
# ■ 傾きの項
#   AtCoder型_相対         = レーティング − そのレースの出走馬の平均
#   AtCoder型_伸び         = 前走のパフォーマンス − 補正前のレーティング（前走までの重み付き平均。2走未満は 0）
#                            （「直近数走の傾き」は3走以上必要で、3点の傾きはぶれが大きいので、こちらにした）
#   AtCoder型_出走の少なさ = f(n) / 1200（0〜1走で 1、走るほど 0 に）− そのレースの平均
#   AtCoder型_ばらつき     = 直近5走（それより少なければある分）のパフォーマンスの標準偏差 − そのレースの平均
#                            （3走以上の馬だけで計算。2走以下の馬は 0 ＝ レースの平均と同じ扱い）。調子の波が大きい馬
#                            （大敗の後に人気を落として好走する馬など）を、市場が正しく値付けしているかを見る

AC_DECAY = 0.9
AC_NEW_APERF = 1200.0
AC_SCALE = 400.0
AC_SPREAD_RUNS = 5           # ばらつきに使う直近の走数
AC_SPREAD_MIN = 3            # ばらつきを出す最少の走数
AC_TERMS = ["AtCoder型_相対", "AtCoder型_伸び", "AtCoder型_出走の少なさ", "AtCoder型_ばらつき"]


def atcoder_correction(n) -> np.ndarray:
    """出走 n 回のときの割り引き f(n)（n = 0 は 1200 とする）。"""
    n = np.asarray(n, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (np.sqrt(1 - 0.81 ** n) / (1 - 0.9 ** n) - 1) / (np.sqrt(19) - 1) * 1200
    return np.where(n >= 1, f, 1200.0)


def atcoder_performance(aperf: np.ndarray, iters: int = 60) -> np.ndarray:
    """着順の順（1着が先頭）に並んだ出走馬の APerf から、各馬のパフォーマンス。"""
    n = len(aperf)
    target = np.arange(1, n + 1) - 0.5
    lo = np.full(n, -10000.0)
    hi = np.full(n, 10000.0)
    for _ in range(iters):
        x = (lo + hi) / 2
        expected = (1.0 / (1.0 + 6.0 ** ((x[:, None] - aperf[None, :]) / AC_SCALE))).sum(axis=1)
        above = expected > target              # 期待される順位が実際より下 → X が小さすぎる
        lo = np.where(above, x, lo)
        hi = np.where(above, hi, x)
    return (lo + hi) / 2


def compute_atcoder_ratings(race_result: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """AtCoder 型のレース前のレーティング（1頭1行）。日付ごとに、その日の値を先に記録してから更新する（as_of=day）。

    戻り値の列: レースID, 血統登録番号, AtCoder型_レーティング, AtCoder型_出走数, AtCoder型_前走パフォーマンス,
    AtCoder型_パフォーマンスの標準偏差, と AC_TERMS
    """
    df = race_result[["レースID", "レース日付", "血統登録番号", "着順", "馬番"]].copy()
    df["_rank"] = pd.to_numeric(df["着順"], errors="coerce").fillna(99)
    df["_post"] = pd.to_numeric(df["馬番"], errors="coerce").fillna(99)
    df = df.sort_values(["レース日付", "レースID", "_rank", "_post"], kind="mergesort").reset_index(drop=True)
    horse = pd.factorize(df["血統登録番号"])[0]
    day = (pd.to_datetime(df["レース日付"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    H = horse.max() + 1 if len(horse) else 0
    s_exp = np.zeros(H)          # Σ 2^(perf/800) × 0.9^(何走前)
    s_w = np.zeros(H)            # Σ 0.9^(何走前)
    s_perf = np.zeros(H)         # Σ perf × 0.9^(何走前)
    runs = np.zeros(H, dtype=np.int64)
    last_perf = np.full(H, np.nan)
    recent = np.full((H, AC_SPREAD_RUNS), np.nan)    # 直近のパフォーマンス（リングバッファ）

    n = len(df)
    out_r, out_raw, out_runs, out_last = np.empty(n), np.empty(n), np.empty(n, dtype=np.int64), np.empty(n)
    out_sd = np.full(n, np.nan)
    race_codes = pd.factorize(df["レースID"])[0]
    starts = np.flatnonzero(np.r_[True, race_codes[1:] != race_codes[:-1]])
    bounds = np.r_[starts, n]
    day_of_race = day[starts]
    day_starts = np.flatnonzero(np.r_[True, day_of_race[1:] != day_of_race[:-1]])
    day_bounds = np.r_[day_starts, len(starts)]
    for di in range(len(day_starts)):
        r0, r1 = day_bounds[di], day_bounds[di + 1]
        pending = []
        for r in range(r0, r1):                                   # その日の全レース：レース前の値を記録
            a, b = bounds[r], bounds[r + 1]
            h = horse[a:b]
            has = runs[h] > 0
            raw = np.where(has, 800 * np.log2(np.where(has, s_exp[h] / np.where(has, s_w[h], 1), 1)), np.nan)
            out_raw[a:b] = raw
            out_r[a:b] = np.where(has, raw - atcoder_correction(runs[h]), 0.0)
            out_runs[a:b] = runs[h]
            out_last[a:b] = last_perf[h]
            enough = runs[h] >= AC_SPREAD_MIN
            if enough.any():
                out_sd[a:b][enough] = np.nanstd(recent[h[enough]], axis=1, ddof=1)
            aperf = np.where(has, s_perf[h] / np.where(has, s_w[h], 1), AC_NEW_APERF)
            pending.append((h, atcoder_performance(aperf)))
        for h, perf in pending:                                   # その日の結果で更新
            s_exp[h] = AC_DECAY * s_exp[h] + 2.0 ** (perf / 800)
            s_w[h] = AC_DECAY * s_w[h] + 1.0
            s_perf[h] = AC_DECAY * s_perf[h] + perf
            runs[h] += 1
            last_perf[h] = perf
            recent[h, (runs[h] - 1) % AC_SPREAD_RUNS] = perf
        if verbose and di % 2000 == 0:
            print(f"    AtCoder 型 {di:,} / {len(day_starts):,} 日", flush=True)

    res = pd.DataFrame({"レースID": df["レースID"].to_numpy(), "血統登録番号": df["血統登録番号"].to_numpy(),
                        "AtCoder型_レーティング": out_r, "AtCoder型_出走数": out_runs,
                        "AtCoder型_前走パフォーマンス": out_last, "AtCoder型_パフォーマンスの標準偏差": out_sd})
    g = res.groupby("レースID")
    res["AtCoder型_相対"] = res["AtCoder型_レーティング"] - g["AtCoder型_レーティング"].transform("mean")
    res["AtCoder型_伸び"] = np.where(out_runs >= 2, out_last - out_raw, 0.0)
    few = pd.Series(atcoder_correction(out_runs) / 1200, index=res.index)
    res["AtCoder型_出走の少なさ"] = few - few.groupby(res["レースID"]).transform("mean")
    sd = res["AtCoder型_パフォーマンスの標準偏差"]
    res["AtCoder型_ばらつき"] = (sd - sd.groupby(res["レースID"]).transform("mean")).fillna(0.0)
    return res
