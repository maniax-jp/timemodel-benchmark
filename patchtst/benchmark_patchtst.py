"""
PatchTST専用ベンチマークスクリプト

データの90%で学習、10%で評価を行います。
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

# PatchTSTモデルをインポート
patchtst_dir = os.path.join(project_root, 'patchtst')
sys.path.insert(0, patchtst_dir)
from models.patchtst_model import PatchTSTPredictor


def run_patchtst_benchmark(
    data_loader: ETTh1Loader,
    num_test_samples: int = 10,
    train_ratio: float = 0.9,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    output_dir: str = "results",
    use_saved_model: bool = True,
    model_save_path: str = "patchtst/saved_model"
):
    """PatchTSTベンチマークを実行"""
    print("\n" + "="*60)
    print("PatchTST ベンチマーク実行")
    print("="*60)

    # 保存されたモデルの確認
    model_exists = os.path.exists(model_save_path) and os.path.exists(os.path.join(model_save_path, "config.json"))

    if use_saved_model and model_exists:
        print(f"\n保存されたモデルを使用します: {model_save_path}")
        predictor = PatchTSTPredictor.load_model(model_save_path, device="cuda")

        # 訓練はスキップ
        history = {
            "training_time": 0,
            "final_train_loss": 0,
            "used_saved_model": True
        }

        # 評価データの準備（訓練はスキップするが評価は必要）
        print("\n[ステップ0] 評価データの準備...")
        all_contexts, all_targets = data_loader.create_samples(split="train", num_samples=None)
        np.random.seed(42)
        indices = np.random.permutation(len(all_contexts))
        all_contexts = all_contexts[indices]
        all_targets = all_targets[indices]

        split_idx = int(len(all_contexts) * train_ratio)
        eval_contexts = all_contexts[split_idx:]
        eval_targets = all_targets[split_idx:]

        print(f"評価サンプル数: {len(eval_contexts)}")
    else:
        # モデルを新規作成
        print("\n新しいモデルを作成します")
        predictor = PatchTSTPredictor(
            context_length=data_loader.context_length,
            horizon_length=data_loader.prediction_length,
            patch_length=16,
            num_hidden_layers=3,
            d_model=128,
            num_attention_heads=4,
            device="cuda"
        )

        # 全データを取得してシャッフル
        print("\n[ステップ0] データの準備...")
        print(f"訓練データ割合: {train_ratio*100:.0f}%, 評価データ割合: {(1-train_ratio)*100:.0f}%")

        # 訓練データから最大数のサンプルを作成
        all_contexts, all_targets = data_loader.create_samples(split="train", num_samples=None)
        print(f"利用可能なサンプル数: {len(all_contexts)}")

        # データをシャッフル
        np.random.seed(42)
        indices = np.random.permutation(len(all_contexts))
        all_contexts = all_contexts[indices]
        all_targets = all_targets[indices]

        # 90/10に分割
        split_idx = int(len(all_contexts) * train_ratio)
        train_contexts = all_contexts[:split_idx]
        train_targets = all_targets[:split_idx]
        eval_contexts = all_contexts[split_idx:]
        eval_targets = all_targets[split_idx:]

        print(f"訓練サンプル数: {len(train_contexts)}")
        print(f"評価サンプル数: {len(eval_contexts)}")

        # 評価用の検証データ（訓練データから一部を分離）
        val_split_idx = int(len(train_contexts) * 0.9)
        val_contexts = train_contexts[val_split_idx:]
        val_targets = train_targets[val_split_idx:]
        train_contexts = train_contexts[:val_split_idx]
        train_targets = train_targets[:val_split_idx]

        print(f"実際の訓練サンプル数: {len(train_contexts)}")
        print(f"検証サンプル数: {len(val_contexts)}")

        # モデルの訓練
        print("\n[ステップ1] モデルの訓練...")
        history = predictor.train(
            train_contexts=train_contexts,
            train_targets=train_targets,
            val_contexts=val_contexts,
            val_targets=val_targets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            output_dir=os.path.join(output_dir, "checkpoints"),
            verbose=True
        )

        # モデルを保存
        predictor.save_model(model_save_path, verbose=True)
        history["used_saved_model"] = False

    # 評価データから指定数のサンプルを選択
    if num_test_samples > len(eval_contexts):
        num_test_samples = len(eval_contexts)

    test_contexts = eval_contexts[:num_test_samples]
    test_targets = eval_targets[:num_test_samples]

    # 単一サンプルでテスト
    print(f"\n[ステップ2] 単一サンプルでテスト...")
    context_single = test_contexts[0]
    target_single = test_targets[0]

    start_time = time.time()
    prediction_single = predictor.predict_single(context_single)
    single_inference_time = time.time() - start_time

    print(f"単一サンプル推論時間: {single_inference_time:.3f}秒")

    # 単一サンプルの評価
    metrics_single = calculate_all_metrics(target_single, prediction_single)
    print_metrics(metrics_single, "PatchTST (Single Sample)")

    # グラフ保存
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_single_prediction(
        context=context_single,
        target=target_single,
        prediction=prediction_single,
        model_name="PatchTST",
        save_path=str(output_path / "patchtst_single_prediction.png"),
        show=False
    )

    # 複数サンプルでベンチマーク
    print(f"\n[ステップ3] {num_test_samples}サンプルでベンチマーク...")
    print(f"Test contexts shape: {test_contexts.shape}")
    print(f"Test targets shape: {test_targets.shape}")

    # バッチ予測
    result = predictor.benchmark(
        contexts=test_contexts,
        targets=test_targets,
        verbose=True
    )

    predictions = result["predictions"]
    inference_time = result["inference_time"]

    # 評価指標計算
    metrics = calculate_all_metrics(test_targets, predictions)
    metrics["inference_time"] = inference_time
    metrics["inference_time_per_sample"] = inference_time / num_test_samples

    print_metrics(metrics, "PatchTST (Test Samples)")

    # 複数サンプルのグラフ保存
    n_plot = min(4, num_test_samples)
    plot_multiple_predictions(
        contexts=[test_contexts[i] for i in range(n_plot)],
        targets=[test_targets[i] for i in range(n_plot)],
        predictions=[predictions[i] for i in range(n_plot)],
        model_name="PatchTST",
        n_samples=n_plot,
        save_path=str(output_path / "patchtst_multiple_predictions.png"),
        show=False
    )

    # 結果を保存
    # 保存されたモデルを使用した場合は訓練サンプル数を取得できない
    if history.get("used_saved_model"):
        train_samples_count = "N/A (used saved model)"
    else:
        train_samples_count = len(train_contexts) + len(val_contexts)

    results_dict = {
        "model": "PatchTST",
        "model_config": {
            "patch_length": 16,
            "num_hidden_layers": 3,
            "d_model": 128,
            "num_attention_heads": 4,
        },
        "training": {
            "train_samples": train_samples_count,
            "eval_samples": len(eval_contexts),
            "train_ratio": train_ratio,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_time": history["training_time"],
            "final_train_loss": history["final_train_loss"],
            "used_saved_model": history.get("used_saved_model", False),
        },
        "metrics": metrics,
        "num_test_samples": num_test_samples,
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
    parser = argparse.ArgumentParser(description="PatchTST時系列予測ベンチマーク")
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
        "--num-test-samples",
        type=int,
        default=10,
        help="テストで評価するサンプル数"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="訓練データの割合（残りが評価データ）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="訓練エポック数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="バッチサイズ"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="学習率"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stoppingの待機エポック数"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="patchtst/results",
        help="結果の出力ディレクトリ"
    )
    parser.add_argument(
        "--use-saved-model",
        action="store_true",
        default=True,
        help="保存されたモデルを使用するか"
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="patchtst/saved_model",
        help="モデルの保存/読み込みパス"
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
    run_patchtst_benchmark(
        data_loader=data_loader,
        num_test_samples=args.num_test_samples,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        use_saved_model=args.use_saved_model,
        model_save_path=args.model_save_path
    )

    print("\n" + "="*60)
    print("PatchTST ベンチマーク完了!")
    print("="*60)


if __name__ == "__main__":
    main()
