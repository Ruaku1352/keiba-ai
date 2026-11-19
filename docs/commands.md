# コマンドの一覧

> README から分けたもの。どれも Windows で `py -m ...` として動かす。オプションの詳細は `py -m src.<モジュール> <サブコマンド> --help`。
> 時間の目安と期待される出力は、各 docs（とくに [v10_raceday.md](v10_raceday.md)）にある。

## データの取り込み：`src.jvlink`（JV-Link。Windows 専用）

| コマンド | 何をする |
|---|---|
| `check` | 疎通確認（データは取らない） |
| `fetch --option 1` | 確定データ・出馬表の差分（毎週）。全期間のセットアップは `fetch --from 19860101000000 --option 4`（約20時間。止まっても同じコマンドで続きから） |
| `fetch-odds` | 確定オッズ（O1〜O6）を `data\jvlink\odds` に保存（月曜の fetch のあと）。`--targets <CSV>` でその CSV のレースだけ（検証用に過去の分を取り直すとき） |
| `rt-odds --date … [--race …]` | 当日の速報オッズ（0B30 など）を取得時刻つきで保存 |
| `summary` / `odds-summary` | 保存した CSV・確定オッズの集計 |
| `bench` | 先頭の数ファイルだけ読んで、取り込みの遅い原因の内訳を測る |

## 変換と再検証：`src.v5`

`build`（変換）/ `grades`（年 × 格付け）/ `coverage`（列の欠損率）/ `validate [--ablation]`（5 fold の時系列CV）。[v5_jravan.md](v5_jravan.md)

## 今のモデル（6頭BOX。比べるための物差し）：`src.v6`

| コマンド | 何をする |
|---|---|
| `train [--until …]` | 学習して `data\models` に保存（2003年〜、454ラウンド） |
| `models` | 保存したモデルの一覧 |
| `predict --date … [--sheet] [--going 稍重]` | 出馬表の重賞を予測。`--sheet` で「評価入力」に書く |
| `results [--sheet]` | 保存した予測と結果を照合（「記録」に着順など） |
| `replay --date …` | 過去の日を出馬表として予測し、答え合わせ |
| `diagnose weight / payout-trend / check-3f / style-zero / hot-payouts / weekend / start-year / dup-races / arima` | 確認・診断 |

## 検証（平場で試す）：`src.v8`

新しい案の成績は平場だけで出す（重賞の行が混ざると止まる）。[v8_validation.md](v8_validation.md)

`baseline` / `odds-coverage` / `levels` / `market` / `trifecta` / `bets` / `trackbias` / `tilt` / `odds-targets` / `odds-drift` /
`stakes` / `rating` / `final-compare` / `growth-check` / `place-check` / `real-odds`

## 重賞での確認・付け替えの試算・k と m の決定：`src.v9`

| コマンド | 何をする |
|---|---|
| `confirm` | 重賞で1回だけの確認のテスト（F・F2・B。判定のルールは先に固定。やり直しは `--rerun-reason`）。[v9_confirm.md](v9_confirm.md) |
| `curse [--flat-all]` | 点数の少ない買い目の「勝者の呪い」の確認（記述だけ） |
| `rebalance [--drift assumed\|data\|none] [--drift-source all\|jusho] [--drift-at 20\|10] [--k 1.0,…,1.4] [--draws 10]` | 付け替えの試算（k × m。記述だけ） |
| `rebalance-decide` | 12月中旬に1回だけ：k・m を決める（平場で選び、ずれは重賞のサンプルから。最終を10分前に決めていたら10分前のずれ） |

## 当日の仕組み：`src.raceday`

| コマンド | いつ | 何をする |
|---|---|---|
| `prep --date … [--schedule] [--k …] [--m …] [--final-minutes …]` | 開催の前夜 | 出走馬・係数・予備の買い目を保存。`--schedule` でタスクの登録と6頭BOX の予想の記録も |
| `schedule --date … [--remove]` | 前夜 | タスクスケジューラへの登録・削除 |
| `update [--date …] [--now HHMM] [--no-sheet] [--no-drift]` | 当日（タスクが呼ぶ） | 1回分：速報の結果 → 当日の傾向 → 買い目 → 付け替え → 保存 → シート・app_data → 20分前・10分前ごろのオッズ集め |
| `keepawake` | （update が起こす） | 最後の発走のあとまで待機中のスリープを防ぐ |
| `check-results --date …` | 当日・確認 | 保存した速報の結果（0B12）の確認 |
| `sheet --date … [--now HHMM] [--mobile-only] [--replay]` / `sheet --sample` | 確認 | 保存した買い目からシート（当日の買い目・スマホ・app_data）を書き直す / スマホ用タブの見本 |
| `app-preview [--date … --now HHMM] [--open]` | 確認 | ウェブアプリの画面を PC のブラウザで見る HTML に書き出す（シートには書かない） |
| `score [--sheet] [--replay-dates …]` | 月曜 | F の最終の買い目の答え合わせ（6頭BOX と並べる。シート「当日の答え合わせ」「成績」） |
| `drift [--since …]` | 月曜（fetch-odds のあと） | 20分前ごろ（と10分前ごろ）のオッズと確定オッズのずれの表。`drift_samples.csv.gz` を保存 |
| `practice --date … [--sheet]` | 月曜 | 全レースの練習の記録（集めた20分前ごろのオッズで F を作り、実際の払戻で答え合わせ。記述だけ） |
| `practice --explain` | 平日 | 当たったのに5万円に届かなかったレースの調査（バグかずれか） |
| `practice --date … --at 10` | 月曜 | 同じレースで20分前と10分前に作った F の比べ |
| `replay --date … [--at HHMM] [--sheet]` | 平日 | ドライラン（過去の日の流れを再現） |
| `decide-final` | 12月中旬に1回 | 最終の時刻（20分前か10分前か）を決める |
| `freeze [--k …] [--m …] [--final-minutes …]` / `unfreeze` | 12月中旬 | 係数・k・m・最終の時刻を固定 / 固定をやめる |

ウェブアプリ（Apps Script）の設定は [v17_webapp.md](v17_webapp.md)。

## テスト

`py -m pytest -q`（418 passed, 7 skipped。JRA-VAN の SDK が無い環境での件数）
