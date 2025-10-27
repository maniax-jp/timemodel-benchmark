"""
TimesFMモデル実装
GoogleのTimesFM (Time Series Foundation Model) を使用した予測
"""
import numpy as np
import torch
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
from typing import Union, List
import time


class TimesFMPredictor:
    """TimesFMを使用した時系列予測"""

    def __init__(
        self,
        context_length: int = 96,
        horizon_length: int = 96,
        backend: str = "gpu",
    ):
        """
        Args:
            context_length: 入力コンテキストの長さ
            horizon_length: 予測ホライゾンの長さ
            backend: "gpu" または "cpu"
        """
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.backend = backend

        print(f"TimesFMモデルを初期化中... (backend: {backend})")
        print(f"Context length: {context_length}, Horizon: {horizon_length}")

        # ハイパーパラメータの設定
        hparams = TimesFmHparams(
            context_len=context_length,
            horizon_len=horizon_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=backend,
        )

        # チェックポイントの設定
        checkpoint = TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )

        # TimesFMモデルの初期化
        print("事前学習済みチェックポイントをロード中...")
        self.model = TimesFm(hparams=hparams, checkpoint=checkpoint)
        print("TimesFMモデルの初期化が完了しました")

    def predict_single(
        self,
        context: np.ndarray,
        frequency: int = 0
    ) -> np.ndarray:
        """
        単一サンプルの予測

        Args:
            context: コンテキスト系列 (context_length,)
            frequency: 周波数インデックス(0: 高頻度、1: 中頻度、2: 低頻度)

        Returns:
            予測値 (horizon_length,)
        """
        # 入力の検証
        if len(context) != self.context_length:
            raise ValueError(
                f"Context length mismatch: expected {self.context_length}, got {len(context)}"
            )

        # (1, context_length) に変形
        context_reshaped = context.reshape(1, -1)

        # 予測実行
        forecast = self.model.forecast(
            inputs=context_reshaped,
            freq=[frequency]
        )

        # forecastはタプル (point_forecast, quantile_forecast)
        # point_forecastは (1, horizon_length) の形状
        if isinstance(forecast, tuple):
            point_forecast = forecast[0]  # (1, horizon_length)
            return point_forecast[0]  # (horizon_length,)
        elif hasattr(forecast, 'point_forecast'):
            return forecast.point_forecast[0]
        else:
            return np.array(forecast).reshape(-1)

    def predict_batch(
        self,
        contexts: np.ndarray,
        frequency: int = 0
    ) -> np.ndarray:
        """
        バッチ予測

        Args:
            contexts: コンテキスト系列 (batch_size, context_length)
            frequency: 周波数インデックス

        Returns:
            予測値 (batch_size, horizon_length)
        """
        batch_size = contexts.shape[0]

        if contexts.shape[1] != self.context_length:
            raise ValueError(
                f"Context length mismatch: expected {self.context_length}, got {contexts.shape[1]}"
            )

        # 予測実行
        forecast = self.model.forecast(
            inputs=contexts,
            freq=[frequency] * batch_size
        )

        # forecastはタプル (point_forecast, quantile_forecast)
        if isinstance(forecast, tuple):
            return forecast[0]  # (batch_size, horizon_length)
        elif hasattr(forecast, 'point_forecast'):
            return forecast.point_forecast
        else:
            return np.array(forecast)

    def benchmark(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        frequency: int = 0,
        verbose: bool = True
    ) -> dict:
        """
        ベンチマーク実行

        Args:
            contexts: コンテキスト系列 (n_samples, context_length)
            targets: 正解値 (n_samples, horizon_length)
            frequency: 周波数インデックス
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
            print(f"\nTimesFMベンチマーク実行中...")
            print(f"サンプル数: {n_samples}")
            print(f"Context length: {self.context_length}")
            print(f"Horizon length: {self.horizon_length}")

        # 推論時間計測
        start_time = time.time()

        predictions = self.predict_batch(contexts, frequency=frequency)

        inference_time = time.time() - start_time

        if verbose:
            print(f"推論完了: {inference_time:.3f}秒")
            print(f"1サンプルあたり: {inference_time/n_samples:.3f}秒")

        return {
            "predictions": predictions,
            "targets": targets,
            "inference_time": inference_time,
        }


def test_timesfm():
    """TimesFMモデルの動作確認"""
    print("=== TimesFM動作テスト ===\n")

    # ダミーデータで動作確認
    context_length = 96
    horizon_length = 96

    # ランダムな時系列データ生成
    np.random.seed(42)
    context = np.random.randn(context_length).astype(np.float32)

    # モデル初期化
    predictor = TimesFMPredictor(
        context_length=context_length,
        horizon_length=horizon_length,
        backend="gpu"
    )

    # 予測実行
    print("\n単一サンプル予測テスト...")
    prediction = predictor.predict_single(context)

    print(f"Context shape: {context.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction sample (first 10): {prediction[:10]}")

    print("\n動作テスト完了!")


if __name__ == "__main__":
    test_timesfm()
