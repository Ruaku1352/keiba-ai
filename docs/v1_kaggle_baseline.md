# v1：Kaggle データでの土台（特徴量・回収率評価・期待値）

> 開発記録。最新の状況と使い方はリポジトリ直下の README を参照。

Kaggle「JRA日本中央競馬会 Horse Racing Dataset」(1986-2021) を使い、
**オッズに対して割安な馬（市場が過小評価している馬）** を見つけて
期待値プラスの買い目を出すためのコード一式。

進捗まとめの依頼1〜3に対応している。

| 依頼 | 対応ファイル |
|---|---|
| 依頼1：特徴量エンジニアリングの拡充 | `src/features.py` |
| 依頼2：回収率ベースの評価関数 | `src/evaluate.py` |
| 依頼3：期待値ベースの買い目抽出 | `src/betting.py` |
| （前処理・時系列分割） | `src/preprocess.py` |
| （学習〜評価の一気通貫） | `src/train.py` |

---

## Colab での使い方

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ruaku1352/keiba_kaggle.git
%cd keiba_kaggle

from src import train
result = train.run_pipeline("/content/drive/MyDrive/keiba/race_result.csv")
```

これで
**読み込み → 前処理 → 特徴量作成 → 時系列分割 → LightGBM学習 → 回収率評価 → 期待値検証**
まで走り、重要度・回収率・期待値スイープの表が出力される。

戻り値 `result` から個別に触れる：

```python
result.importance        # gain 重要度
result.pred              # テスト期間の予測確率
result.test_df           # テスト期間のデータ
result.feature_cols      # 使った特徴量
```

### メモリ・時間がきついとき

```python
# 特徴量は全期間から計算した上で、2000年以降だけを学習に使う
result = train.run_pipeline(csv_path, start_year=2000)
```

162万行 × 40特徴量で概ね 1.5〜2.5 GB 程度。Colab 無料枠（12GB）に収まる設計だが、
学習が遅い場合は `start_year` で絞るのが一番効く。
`preprocess.load_race_result(path, usecols=[...])` で読む列を絞るのも有効。

---

## 依頼1：追加した特徴量

### Data Leakage を防ぐ仕組み

全ての過去実績系は **その行自身を含まない過去だけ** から作る。実装は 2 つの道具に統一：

```python
過去の件数 = groupby.cumcount()              # 自分より前の行数
過去の合計 = groupby.cumsum() - 自分の値      # 自分より前の合計
```

`cumsum()` は自分を含む累積和なので、自分の値を引けば「自分より前だけ」になる。
直近N走は **累積和の差分** で取る（`groupby.rolling` は162万行だと遅すぎるため）：

```
直近N走の合計 = (自分より前の累積和) - (N行前時点の累積和)
```

この前提として `preprocess.basic_clean()` が
`[レース日付, レースID, 馬番]` でソートしてから返している。

### 特徴量一覧

**馬の実績系**

| 特徴量 | 内容 |
|---|---|
| 通算出走回数 | その時点までの出走数 |
| 通算勝率 / 通算連対率 / 通算複勝率 | 1着 / 2着以内 / 3着以内の率 |
| 過去3走平均着順 / 過去5走平均着順 | 直近N走の着順平均 |
| 前走着順 | 直前レースの着順 |
| 前走からの日数 | 休養明けかどうか |
| 前走との距離差 | 距離延長／短縮 |
| 前走上がり3F | 末脚の速さ（`後３Ｆタイム` があれば） |
| 同距離帯_過去勝率 | 200m刻みのビンごとの勝率 |
| 同芝ダ_過去勝率 | 芝／ダート別の勝率 |
| 同馬場状態_過去勝率 | 良／重などの適性 |
| 同競馬場_過去勝率 | コース適性 |

> 依頼にあった「同距離帯（±200m）」はスライド窓で groupby できないため、
> **200m刻みのビン**（`距離 // 200`）で代用している。
> 1600m の馬は 1500〜1699m の実績とだけ比較される点に注意。

**騎手系**：騎手通算勝率 / 騎手通算騎乗数 / 騎手直近100走勝率 / 騎手×競馬場_勝率 / 騎手×芝ダ_勝率

**馬×騎手系**：コンビ回数 / コンビ勝率 / 乗り替わりフラグ

**調教師系**：調教師通算勝率 / 調教師通算出走数

**レース単位**：出走頭数 / 通算勝率_レース内順位 / 騎手勝率_レース内順位 / 斤量_レース内平均差

### 同日レースの扱い（重要）

騎手・調教師の勝率は既定で「同じ開催日の、より前のレース」も過去に含む。
実際の運用（当日朝に予測を出す）で厳密にしたい場合：

```python
df = features.add_all_features(df, strict_daily_lag=True)
```

とすると **前日終了時点** の成績に切り替わる。

### 単独で使う場合

```python
from src import preprocess, features

df = preprocess.load_race_result(path)
df = preprocess.basic_clean(df)
df = features.add_all_features(df)
cols = features.feature_columns(df)          # オッズ・人気は除外される
cols = features.feature_columns(df, include_odds=True)  # 比較用に含める
```

---

## 依頼2：回収率で評価する

正解率は無意味（1着は7.6%なので「全部0予測」で92.4%）。実際の馬券収支で見る。

```python
from src import evaluate

# 予測上位1頭に単勝100円
r = evaluate.backtest_top_n(result.test_df, result.pred, n=1)
print(r)                    # レース数 / 的中率 / 回収率 / 収支

# 上位1〜3頭を横並び比較
evaluate.backtest_by_top_n(result.test_df, result.pred, ns=(1, 2, 3))

# オッズ帯ごとの成績（どのゾーンで勝てているか）
evaluate.odds_band_report(result.test_df, result.pred)

# 収支推移のグラフ
evaluate.plot_profit_curve(r)
```

**基準となる数字**

| 指標 | 意味 |
|---|---|
| 回収率 80% 前後 | 適当に買った場合（JRAの単勝控除率が約20%） |
| 回収率 100% | 損益トントン |
| `evaluate.favorite_baseline(test_df)` | 1番人気を買い続けた場合。**モデルはこれを超えて初めて意味がある** |
| `evaluate.random_baseline(test_df)` | ランダムに1頭選んだ場合 |

検証は必ず時系列で後半20%（`preprocess.time_series_split`）だけを使う。

---

## 依頼3：期待値ベースの買い目抽出

```
期待値 EV = 予測確率 × 単勝オッズ
```

EV > 1.0 の馬 = 市場が過小評価している馬。

```python
from src import betting

# 期待値つきの一覧（prob は レース内で合計1に正規化済み）
ev = betting.compute_expected_value(result.test_df, result.pred)

# 期待値1.0以上を抽出
picks = betting.select_value_bets(result.test_df, result.pred, ev_threshold=1.0)

# 高配当（オッズ10倍以上）だけに絞ったときの回収率
betting.backtest_value_bets(result.test_df, result.pred,
                            ev_threshold=1.0, min_odds=10)

# 閾値を振って安定性を確認
betting.ev_threshold_sweep(result.test_df, result.pred)

# オッズ下限ごとの成績
betting.high_odds_report(result.test_df, result.pred)
```

**確率の正規化について**：LightGBM の binary 出力はレース内で合計1にならない。
そのまま掛けると期待値が系統的にずれるので、`compute_expected_value` は
レース単位で正規化してから期待値を計算している。

**閾値スイープの読み方**：閾値を上げるほど点数が減って回収率が上がるなら、
モデルの期待値はある程度信頼できている。バラバラならノイズを拾っているだけ。

**賭け金の配分**

```python
picks = betting.select_value_bets(test_df, pred, ev_threshold=1.2, min_odds=5)
betting.bet_amounts(picks, bankroll=100000, use_kelly=True)  # ケリー基準（上限5%）
```

**本番レースの確認**

```python
betting.race_picks(test_df, pred, race_id="2026...", top=8)
```

---

## テスト

リーク検出を含めたテストを同梱している（実データ不要、合成データで走る）。

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

主な検証内容：

- 初出走行では過去実績が空（0回 / NaN）になっている
- 通算勝率が「自分より前の行だけ」の平均と厳密に一致する
- 過去3走平均着順が直前3走の平均と一致する
- **後半のレース結果をシャッフルしても前半の特徴量が1ミリも変わらない**（＝未来を見ていない決定的な証拠）
- 回収率の計算が定義どおり（払戻＝オッズ×100、外れは0）
- レース内の正規化確率の合計が1

pandas 2.2 / 3.0 の両方で通ることを確認済み。
