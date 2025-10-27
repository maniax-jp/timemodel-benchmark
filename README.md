# 時系列予測モデルベンチマーク

TimesFM、Chronos、PatchTSTの3つの時系列基盤モデルに対するベンチマーク実装。

## セットアップ

```bash
# 依存関係のインストール
uv sync

# 仮想環境の有効化
source .venv/bin/activate
```

## 実行

### TimesFM ベンチマーク

```bash
# TimesFM単体でベンチマーク
python timesfm/benchmark_timesfm.py --num-samples 10

# カスタム設定で実行
python timesfm/benchmark_timesfm.py \
  --num-samples 50 \
  --context-length 96 \
  --prediction-length 96
```

### Chronos ベンチマーク

```bash
# Chronos単体でベンチマーク (デフォルト: tiny モデル)
python chronos/benchmark_chronos.py --num-samples 10

# 軽量モデルで実行
python chronos/benchmark_chronos.py \
  --num-samples 10 \
  --model-name amazon/chronos-t5-tiny

# カスタム設定で実行
python chronos/benchmark_chronos.py \
  --num-samples 50 \
  --model-name amazon/chronos-t5-base
```

### PatchTST ベンチマーク

```bash
# PatchTST単体でベンチマーク（データの90%で学習、10%で評価）
python patchtst/benchmark_patchtst.py \
  --epochs 10 \
  --batch-size 128 \
  --num-test-samples 10

# カスタム設定で実行
python patchtst/benchmark_patchtst.py \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --num-test-samples 50 \
  --train-ratio 0.9

# モデル保存機能
# 初回実行時にモデルが patchtst/saved_model/ に保存されます
# 2回目以降の実行では保存されたモデルを自動的に読み込み、訓練をスキップします
# 保存されたモデルを使用しない場合は --no-use-saved-model を指定してください
```

## ベンチマーク結果

統一設定(context=96, prediction=96)での比較:

| モデル | MSE | MAE | RMSE | MAPE | 推論時間/サンプル | 備考 |
|--------|-----|-----|------|------|-------------------|------|
| **TimesFM** | **5.98** ⭐ | **1.86** ⭐ | **2.44** ⭐ | **27.3%** ⭐ | 3.5ms | Zero-shot |
| Chronos (tiny) | 9.98 | 2.44 | 3.16 | 37.5% | 136.6ms | Zero-shot |
| **PatchTST** | 11.55 | 2.59 | 3.40 | **21.4%** ⭐ | **0.26ms** ⭐ | 90%学習/10%評価 |

**結論**:
- **精度（MSE）**: TimesFMが最高精度
- **MAPE**: PatchTSTが最も低い相対誤差
- **推論速度**: PatchTSTが最速（学習済みモデル）
- **使いやすさ**: TimesFM/Chronosはzero-shot利用可能

## プロジェクト構造

```
timemodel-benchmark/
├── README.md                    # このファイル
├── PLAN.md                      # 実装計画書
├── pyproject.toml              # uv依存関係管理
├── data/
│   └── ETTh1.csv               # ベンチマークデータセット
├── utils/                       # 共通ユーティリティ
│   ├── data_loader.py          # データローダー
│   ├── metrics.py              # 評価指標計算
│   └── visualization.py        # グラフ可視化
├── timesfm/                     # TimesFM専用
│   ├── README.md               # TimesFM詳細
│   ├── benchmark_timesfm.py    # ベンチマークスクリプト
│   ├── models/
│   │   └── timesfm_model.py    # モデル実装
│   └── results/
│       ├── metrics.json        # 評価結果
│       └── plots/              # グラフ
├── chronos/                     # Chronos専用
│   ├── README.md               # Chronos詳細
│   ├── benchmark_chronos.py    # ベンチマークスクリプト
│   ├── models/
│   │   └── chronos_model.py    # モデル実装
│   └── results/
│       ├── metrics.json        # 評価結果
│       └── plots/              # グラフ
└── patchtst/                    # PatchTST専用
    ├── benchmark_patchtst.py   # ベンチマークスクリプト
    ├── models/
    │   └── patchtst_model.py   # モデル実装
    └── results/
        ├── metrics.json        # 評価結果
        └── plots/              # グラフ
```

## 実装状況

- ✅ **TimesFM**: 実装完了（zero-shot）
- ✅ **Chronos**: 実装完了（zero-shot）
- ✅ **PatchTST**: 実装完了（学習・評価）

## データセット

- **ETTh1** (Electricity Transformer Temperature - Hourly)
  - 総データ数: 17,420
  - 目標変数: OT (油温度)
  - サンプリング: 1時間ごと

## 評価指標

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **推論時間** (GPU)

## Git管理

### 追跡されるファイル
- ソースコード (`*.py`)
- 設定ファイル (`pyproject.toml`, `README.md`, `PLAN.md`)
- 評価結果 (`**/results/metrics.json`)
- データセット (`data/ETTh1.csv`)

### 無視されるファイル (.gitignore)
- 仮想環境 (`.venv/`, `venv/`, `env/`)
- キャッシュ (`__pycache__/`, `*.pyc`)
- 学習済みモデル (`**/saved_model/`, `*.pt`, `*.pth`, `*.safetensors`)
- プロット画像 (`**/results/plots/`, `*.png`, `*.jpg`)
- ログファイル (`*.log`, `nohup.out`)
- IDE設定 (`.vscode/`, `.idea/`)
- OS固有ファイル (`.DS_Store`, `Thumbs.db`)

**理由**:
- 学習済みモデルは容量が大きく、初回実行時に自動生成されるため
- プロット画像は結果から再生成可能
- metrics.jsonは小さく、モデル比較に重要なため追跡対象
