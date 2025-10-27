# TimesFM、Chronosベンチマーク実行計画

## ステータス: ✅ 完了

2つの時系列予測基盤モデル(TimesFM、Chronos)に対してベンチマークを実行し、性能を比較評価する。

## 実施結果サマリー

両モデルで統一設定(context=96, prediction=96)でのベンチマークが完了しました。

### パフォーマンス比較

| モデル | MSE | MAE | RMSE | MAPE | 推論時間/サンプル | 備考 |
|--------|-----|-----|------|------|-------------------|------|
| **TimesFM** | **5.98** | **1.86** | **2.44** | **27.3%** | **3.5ms** | Zero-shot、最高精度・最速 |
| Chronos (tiny) | 9.98 | 2.44 | 3.16 | 37.5% | 136.6ms | Zero-shot |

**結論**:
- **精度**: TimesFMが全指標で優れた性能
- **速度**: TimesFMが圧倒的に高速(約39倍)
- **使いやすさ**: 両モデルともzero-shot利用可能

### 重要な発見

1. **TimesFMの優位性**
   - 精度、速度ともに優れている
   - Zero-shot予測で即座に利用可能
   - 実用性が高い

2. **Chronosの特性**
   - 複数のモデルサイズが利用可能(tiny/mini/small/base/large)
   - Tinyモデルでも実用的な精度
   - 推論速度はTimesFMより遅い

## 実行環境
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **CUDA**: 12.9
- **パッケージ管理**: uv
- **OS**: Linux (WSL2)

## プロジェクト構造
```
timemodel-benchmark/
├── pyproject.toml          # uv用の依存関係管理
├── .python-version         # Pythonバージョン指定(3.10または3.11)
├── README.md               # プロジェクト説明
├── PLAN.md                 # この計画書
├── data/                   # データセット格納
│   └── ETTh1.csv          # 最小規模のベンチマークデータ
├── timesfm/               # TimesFMモデル実装
│   ├── models/
│   │   └── timesfm_model.py
│   ├── benchmark_timesfm.py
│   └── results/
├── chronos/               # Chronosモデル実装
│   ├── models/
│   │   └── chronos_model.py
│   ├── benchmark_chronos.py
│   └── results/
├── utils/                  # ユーティリティ
│   ├── __init__.py
│   ├── data_loader.py     # データ読み込み・前処理
│   ├── metrics.py         # 評価指標計算(MSE, MAE, RMSE, MAPE)
│   └── visualization.py   # グラフ生成・可視化
├── benchmark.py            # メインベンチマークスクリプト
└── results/                # 結果保存先
    ├── metrics.json       # 評価指標の数値結果
    └── plots/             # グラフ保存ディレクトリ
```

## ベンチマーク設定

### データセット
- **選択**: ETTh1 (Electricity Transformer Temperature - Hourly)
- **理由**: 最小規模のベンチマークデータセット(約17,420データポイント)
- **特徴**:
  - 7つの変数(OT: 油温度 + 6つの負荷特性)
  - 1時間ごとのサンプリング
  - 単変量/多変量予測の両方に対応

### 推論設定(最小構成)
- **コンテキスト長(入力系列長)**: 96ステップ
- **予測ホライゾン(予測期間)**: 96ステップ
- **バッチサイズ**: 1(最小設定)
- **予測対象**: OT(油温度)のみ(単変量予測)

### 評価指標
1. **定量評価**:
   - MSE (Mean Squared Error)
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)

2. **定性評価**:
   - 予測値と正解値を重ねたグラフ
   - 複数サンプルの予測結果を可視化

## 実装ステップ

### Phase 1: 環境セットアップ
1. uvでプロジェクト初期化
   ```bash
   uv init
   uv venv
   ```

2. 依存関係の追加
   ```bash
   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   uv add timesfm
   uv add chronos-forecasting
   uv add pandas numpy matplotlib seaborn scikit-learn
   ```

### Phase 2: データ準備
1. ETTh1データセットのダウンロード
   - ソース: https://github.com/zhouhaoyi/ETDataset

2. データ分割
   - 学習: 12ヶ月
   - 検証: 4ヶ月
   - テスト: 4ヶ月

3. データローダーの実装

### Phase 3: モデル実装(順次)

#### 3.1 TimesFM (最初に実装)
- **モデル**: Google Research製の事前学習済み時系列基盤モデル
- **特徴**: ゼロショット予測が可能
- **使用モデル**: `google/timesfm-1.0-200m` (Hugging Face)
- **実装**: `models/timesfm_model.py`

#### 3.2 Chronos (2番目に実装)
- **モデル**: Amazon製の事前学習済み時系列基盤モデル
- **特徴**: Transformerベース、複数サイズのモデルあり
- **使用モデル**: `amazon/chronos-t5-tiny` (最小サイズ)
- **実装**: `chronos/models/chronos_model.py`

### Phase 4: 評価・可視化
1. 評価指標の計算(`utils/metrics.py`)
2. グラフ生成(`utils/visualization.py`)
   - 時系列プロット(予測 vs 正解)
   - 誤差分布
   - モデル間比較

### Phase 5: ベンチマーク実行
1. TimesFMで1サンプル推論テスト
2. 結果確認・デバッグ
3. 全サンプルでの評価
4. 他のモデルでも同様に実行
5. 結果の比較・分析

## 期待される出力

### 1. 数値結果
- `timesfm/results/metrics.json`: TimesFMの結果
- `chronos/results/metrics.json`: Chronosの結果

### 2. グラフ
- `timesfm/results/plots/`: TimesFMの予測グラフ
- `chronos/results/plots/`: Chronosの予測グラフ

## 実装状況
1. ✅ 計画書作成(このファイル)
2. ✅ uv環境セットアップ
3. ✅ プロジェクト構造作成
4. ✅ データダウンロード・準備
5. ✅ TimesFM実装と推論テスト
6. ✅ Chronos実装
7. ✅ ベンチマーク実行・結果分析

## 注意事項
- GPU メモリ使用量に注意(必要に応じてモデルサイズを調整)
- 各モデルの入力形式の違いに注意
- 推論速度も計測する
- 再現性のためランダムシードを固定
