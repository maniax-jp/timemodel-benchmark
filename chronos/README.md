# Chronos ベンチマーク

AmazonのChronos (Time Series Foundation Model) を使用した時系列予測のベンチマーク。

## モデル情報

- **モデル名**: Chronos
- **提供元**: Amazon
- **アーキテクチャ**: T5ベース (Transformerエンコーダー・デコーダー)
- **特徴**: 複数サイズのモデルから選択可能

### 利用可能なモデル

| モデル名 | パラメータ数 | 推奨用途 |
|---------|------------|---------|
| `amazon/chronos-t5-tiny` | 8M | 高速テスト・開発 |
| `amazon/chronos-t5-mini` | 20M | 軽量推論 |
| `amazon/chronos-t5-small` | 46M | バランス型 (デフォルト) |
| `amazon/chronos-t5-base` | 200M | 高精度 |
| `amazon/chronos-t5-large` | 710M | 最高精度 |

## 実行方法

```bash
# デフォルト設定で実行 (small モデル)
python chronos/benchmark_chronos.py --num-samples 10

# 軽量モデルで実行
python chronos/benchmark_chronos.py \
  --num-samples 10 \
  --model-name amazon/chronos-t5-tiny

# 高精度モデルで実行
python chronos/benchmark_chronos.py \
  --num-samples 10 \
  --model-name amazon/chronos-t5-base

# カスタム設定
python chronos/benchmark_chronos.py \
  --num-samples 50 \
  --context-length 96 \
  --prediction-length 96 \
  --model-name amazon/chronos-t5-small
```

## 結果

評価結果は以下に保存されます:

- **評価指標**: `chronos/results/metrics.json`
- **グラフ**: `chronos/results/plots/`

## ディレクトリ構造

```
chronos/
├── README.md                           # このファイル
├── benchmark_chronos.py                # ベンチマークスクリプト
├── models/
│   ├── __init__.py
│   └── chronos_model.py               # Chronos実装
└── results/
    ├── metrics.json                    # 評価指標
    └── plots/
        ├── chronos_single_prediction.png
        └── chronos_multiple_predictions.png
```

## 評価指標

- **MSE** (Mean Squared Error): 平均二乗誤差
- **MAE** (Mean Absolute Error): 平均絶対誤差
- **RMSE** (Root Mean Squared Error): 二乗平均平方根誤差
- **MAPE** (Mean Absolute Percentage Error): 平均絶対パーセント誤差
- **推論時間**: GPU上での推論時間

## ベンチマーク結果例 (tiny モデル, 3サンプル)

```json
{
  "mse": 9.983,
  "mae": 2.441,
  "rmse": 3.160,
  "mape": 37.54%,
  "inference_time": 0.410秒,
  "inference_time_per_sample": 0.137秒
}
```

## 注意事項

- Chronosは予測長64以下での使用を推奨しています
- 長い予測ホライゾンでは精度が低下する可能性があります
- より大きいモデルほど精度は向上しますが、推論時間も増加します

## 参考

- [Chronos論文](https://arxiv.org/abs/2403.07815)
- [Hugging Face Model Hub](https://huggingface.co/amazon/chronos-t5-small)
