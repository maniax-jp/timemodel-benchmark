# TimesFM ベンチマーク

GoogleのTimesFM (Time Series Foundation Model) を使用した時系列予測のベンチマーク。

## モデル情報

- **モデル名**: TimesFM
- **提供元**: Google Research
- **バージョン**: timesfm-1.0-200m-pytorch
- **モデルサイズ**: 200M parameters
- **Hugging Face**: [google/timesfm-1.0-200m-pytorch](https://huggingface.co/google/timesfm-1.0-200m-pytorch)

## 実行方法

```bash
# 10サンプルでベンチマーク
python timesfm/benchmark_timesfm.py --num-samples 10

# カスタム設定
python timesfm/benchmark_timesfm.py \
  --num-samples 50 \
  --context-length 96 \
  --prediction-length 96
```

## 結果

評価結果は以下に保存されます:

- **評価指標**: `timesfm/results/metrics.json`
- **グラフ**: `timesfm/results/plots/`

## ディレクトリ構造

```
timesfm/
├── README.md                           # このファイル
├── benchmark_timesfm.py                # ベンチマークスクリプト
├── models/
│   ├── __init__.py
│   └── timesfm_model.py               # TimesFM実装
└── results/
    ├── metrics.json                    # 評価指標
    └── plots/
        ├── timesfm_single_prediction.png
        └── timesfm_multiple_predictions.png
```

## 評価指標

- **MSE** (Mean Squared Error): 平均二乗誤差
- **MAE** (Mean Absolute Error): 平均絶対誤差
- **RMSE** (Root Mean Squared Error): 二乗平均平方根誤差
- **MAPE** (Mean Absolute Percentage Error): 平均絶対パーセント誤差
- **推論時間**: GPU上での推論時間
