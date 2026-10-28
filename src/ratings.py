"""v8c：馬のレーティング（能力が時間とともに変わることを扱う。検証用。本番の特徴量には入れていない）。

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
