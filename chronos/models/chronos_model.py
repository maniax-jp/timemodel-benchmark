"""
Chronosモデル実装
AmazonのChronos (Time Series Foundation Model) を使用した予測
"""
import numpy as np
import torch
from chronos import ChronosPipeline
from typing import Union, List
import time


class ChronosPredictor:
    """Chronosを使用した時系列予測"""

    def __init__(
        self,
        context_length: int = 96,
        horizon_length: int = 96,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cuda",
    ):
        """
        Args:
            context_length: 入力コンテキストの長さ
            horizon_length: 予測ホライゾンの長さ
            model_name: 使用するChronosモデル
                - amazon/chronos-t5-tiny (8M params)
                - amazon/chronos-t5-mini (20M params)
                - amazon/chronos-t5-small (46M params) [デフォルト]
                - amazon/chronos-t5-base (200M params)
                - amazon/chronos-t5-large (710M params)
            device: "cuda" または "cpu"
        """
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.model_name = model_name
        self.device = device

        print(f"Chronosモデルを初期化中... (device: {device})")
        print(f"Model: {model_name}")
        print(f"Context length: {context_length}, Horizon: {horizon_length}")

        # Chronosパイプラインの初期化
        print(f"事前学習済みモデルをロード中: {model_name}")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        print("Chronosモデルの初期化が完了しました")

    def predict_single(
        self,
        context: np.ndarray,
        num_samples: int = 1,
    ) -> np.ndarray:
        """
        単一サンプルの予測

        Args:
            context: コンテキスト系列 (context_length,)
            num_samples: 生成するサンプル数(アンサンブル用)

        Returns:
            予測値 (horizon_length,)
        """
        # 入力の検証
        if len(context) != self.context_length:
            raise ValueError(
                f"Context length mismatch: expected {self.context_length}, got {len(context)}"
            )

        # contextをtensorに変換
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # 予測実行
        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=self.horizon_length,
                num_samples=num_samples,
            )

        # forecastは (batch_size, num_samples, horizon_length) の形状
        # median を取得して (horizon_length,) に変換
        forecast_np = forecast.numpy()
        prediction = np.median(forecast_np[0], axis=0)  # (horizon_length,)

        return prediction

    def predict_batch(
        self,
        contexts: np.ndarray,
        num_samples: int = 1,
    ) -> np.ndarray:
        """
        バッチ予測

        Args:
            contexts: コンテキスト系列 (batch_size, context_length)
            num_samples: 生成するサンプル数

        Returns:
            予測値 (batch_size, horizon_length)
        """
        batch_size = contexts.shape[0]

        if contexts.shape[1] != self.context_length:
            raise ValueError(
                f"Context length mismatch: expected {self.context_length}, got {contexts.shape[1]}"
            )

        # contextをtensorに変換
        context_tensor = torch.tensor(contexts, dtype=torch.float32)

        # 予測実行
        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=self.horizon_length,
                num_samples=num_samples,
            )

        # forecastは (batch_size, num_samples, horizon_length)
        # median を取得
        forecast_np = forecast.numpy()
        predictions = np.median(forecast_np, axis=1)  # (batch_size, horizon_length)

        return predictions

    def benchmark(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        num_samples: int = 1,
        verbose: bool = True
    ) -> dict:
        """
        ベンチマーク実行

        Args:
            contexts: コンテキスト系列 (n_samples, context_length)
            targets: 正解値 (n_samples, horizon_length)
            num_samples: 生成するサンプル数
            verbose: 詳細出力

        Returns:
            {
                "predictions": 予測値,
                "targets": 正解値,
                "inference_time": 推論時間(秒)
            }
        """
        n_samples = contexts.shape[0]

        if verbose:
            print(f"\nChronosベンチマーク実行中...")
            print(f"サンプル数: {n_samples}")
            print(f"Context length: {self.context_length}")
            print(f"Horizon length: {self.horizon_length}")
            print(f"Model: {self.model_name}")

        # 推論時間計測
        start_time = time.time()

        predictions = self.predict_batch(contexts, num_samples=num_samples)

        inference_time = time.time() - start_time

        if verbose:
            print(f"推論完了: {inference_time:.3f}秒")
            print(f"1サンプルあたり: {inference_time/n_samples:.3f}秒")

        return {
            "predictions": predictions,
            "targets": targets,
            "inference_time": inference_time,
        }


def test_chronos():
    """Chronosモデルの動作確認"""
    print("=== Chronos動作テスト ===\n")

    # ダミーデータで動作確認
    context_length = 96
    horizon_length = 96

    # ランダムな時系列データ生成
    np.random.seed(42)
    context = np.random.randn(context_length).astype(np.float32)

    # モデル初期化
    predictor = ChronosPredictor(
        context_length=context_length,
        horizon_length=horizon_length,
        model_name="amazon/chronos-t5-tiny",  # 最小モデルでテスト
        device="cuda"
    )

    # 予測実行
    print("\n単一サンプル予測テスト...")
    prediction = predictor.predict_single(context)

    print(f"Context shape: {context.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction sample (first 10): {prediction[:10]}")

    print("\n動作テスト完了!")


if __name__ == "__main__":
    test_chronos()
