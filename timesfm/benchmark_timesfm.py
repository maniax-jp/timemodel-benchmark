"""
TimesFM専用ベンチマークスクリプト
"""
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import json
import time
from pathlib import Path
import numpy as np

from utils.data_loader import ETTh1Loader
from utils.metrics import calculate_all_metrics, print_metrics
from utils.visualization import plot_single_prediction, plot_multiple_predictions

# TimesFMモデルをインポート
timesfm_dir = os.path.join(project_root, 'timesfm')
sys.path.insert(0, timesfm_dir)
from models.timesfm_model import TimesFMPredictor


def run_timesfm_benchmark(
    data_loader: ETTh1Loader,
    num_samples: int = 10,
    output_dir: str = "timesfm/results"
):
    """TimesFMベンチマークを実行"""
    print("\n" + "="*60)
    print("TimesFM ベンチマーク実行")
    print("="*60)

    # モデル初期化
    predictor = TimesFMPredictor(
        context_length=data_loader.context_length,
        horizon_length=data_loader.prediction_length,
        backend="gpu"
    )

    # テストデータから1サンプル取得(確認用)
    print("\n[ステップ1] 単一サンプルでテスト...")
    context_single, target_single = data_loader.get_single_sample(split="test", index=0)

    # 単一予測
    start_time = time.time()
    prediction_single = predictor.predict_single(context_single)
    single_inference_time = time.time() - start_time

    print(f"単一サンプル推論時間: {single_inference_time:.3f}秒")

    # 単一サンプルの評価
    metrics_single = calculate_all_metrics(target_single, prediction_single)
    print_metrics(metrics_single, "TimesFM (Single Sample)")

    # グラフ保存
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_single_prediction(
        context=context_single,
        target=target_single,
        prediction=prediction_single,
        model_name="TimesFM",
        save_path=str(output_path / "timesfm_single_prediction.png"),
        show=False
    )

    # 複数サンプルでベンチマーク
    print(f"\n[ステップ2] {num_samples}サンプルでベンチマーク...")
    contexts, targets = data_loader.create_samples(split="test", num_samples=num_samples)

    print(f"Contexts shape: {contexts.shape}")
    print(f"Targets shape: {targets.shape}")

    # バッチ予測
    result = predictor.benchmark(
        contexts=contexts,
        targets=targets,
        frequency=0,
        verbose=True
    )

    predictions = result["predictions"]
    inference_time = result["inference_time"]

    # 評価指標計算
    metrics = calculate_all_metrics(targets, predictions)
    metrics["inference_time"] = inference_time
    metrics["inference_time_per_sample"] = inference_time / num_samples

    print_metrics(metrics, "TimesFM (Multiple Samples)")

    # 複数サンプルのグラフ保存
    n_plot = min(4, num_samples)
    plot_multiple_predictions(
        contexts=[contexts[i] for i in range(n_plot)],
        targets=[targets[i] for i in range(n_plot)],
        predictions=[predictions[i] for i in range(n_plot)],
        model_name="TimesFM",
        n_samples=n_plot,
        save_path=str(output_path / "timesfm_multiple_predictions.png"),
        show=False
    )

    # 結果を保存
    results_dict = {
        "model": "TimesFM",
        "model_version": "google/timesfm-2.0-500m-pytorch",
        "metrics": metrics,
        "num_samples": num_samples,
        "context_length": data_loader.context_length,
        "prediction_length": data_loader.prediction_length,
        "dataset": "ETTh1",
        "target_column": "OT"
    }

    results_file = Path(output_dir) / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n結果を保存しました: {results_file}")

    return results_dict


def main():
    parser = argparse.ArgumentParser(description="TimesFM時系列予測ベンチマーク")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ETTh1.csv",
        help="データセットのパス"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=96,
        help="コンテキスト長"
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=96,
        help="予測長"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="評価するサンプル数"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="timesfm/results",
        help="結果の出力ディレクトリ"
    )

    args = parser.parse_args()

    # データローダー初期化
    print("\n" + "="*60)
    print("データセットを読み込み中...")
    print("="*60)

    data_loader = ETTh1Loader(
        data_path=args.data_path,
        target_column="OT",
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    info = data_loader.get_info()
    print(f"\nデータセット情報:")
    print(f"  総データ数: {info['total_length']}")
    print(f"  訓練データ: {info['train_length']}")
    print(f"  検証データ: {info['val_length']}")
    print(f"  テストデータ: {info['test_length']}")
    print(f"  目標変数: {info['target_column']}")
    print(f"  コンテキスト長: {info['context_length']}")
    print(f"  予測長: {info['prediction_length']}")

    # ベンチマーク実行
    run_timesfm_benchmark(
        data_loader=data_loader,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    print("TimesFM ベンチマーク完了!")
    print("="*60)


if __name__ == "__main__":
    main()
