"""
評価指標モジュール
MSE, MAE, RMSE, MAPEなどの評価指標を計算
"""
import numpy as np
from typing import Dict


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE) を計算"""
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE) を計算"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE) を計算"""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (MAPE) を計算

    Args:
        y_true: 正解値
        y_pred: 予測値
        epsilon: ゼロ除算を防ぐための小さな値

    Returns:
        MAPE値 (%)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    全ての評価指標を計算

    Args:
        y_true: 正解値 (n_samples, prediction_length) または (prediction_length,)
        y_pred: 予測値 (n_samples, prediction_length) または (prediction_length,)

    Returns:
        評価指標の辞書
    """
    return {
        "mse": float(calculate_mse(y_true, y_pred)),
        "mae": float(calculate_mae(y_true, y_pred)),
        "rmse": float(calculate_rmse(y_true, y_pred)),
        "mape": float(calculate_mape(y_true, y_pred)),
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """評価指標を見やすく表示"""
    print(f"\n{'='*50}")
    print(f"{model_name} - 評価指標")
    print(f"{'='*50}")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"{'='*50}\n")
