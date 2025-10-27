"""
可視化モジュール
予測結果と正解値を重ねてプロット
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from pathlib import Path


# 日本語フォント設定(存在する場合)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# スタイル設定
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_single_prediction(
    context: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    単一サンプルの予測結果をプロット

    Args:
        context: コンテキスト系列
        target: 正解値
        prediction: 予測値
        model_name: モデル名
        save_path: 保存先のパス(Noneの場合は保存しない)
        show: プロットを表示するか
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # コンテキスト部分
    context_x = np.arange(len(context))
    ax.plot(context_x, context, 'gray', linewidth=2, alpha=0.7, label='Context')

    # 予測と正解の範囲
    pred_x = np.arange(len(context), len(context) + len(target))

    # 正解値
    ax.plot(pred_x, target, 'b-', linewidth=2, label='Ground Truth', marker='o', markersize=3)

    # 予測値
    ax.plot(pred_x, prediction, 'r--', linewidth=2, label='Prediction', marker='x', markersize=4)

    # 境界線
    ax.axvline(x=len(context), color='green', linestyle=':', linewidth=2, alpha=0.7, label='Prediction Start')

    # ラベルとタイトル
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value (Temperature)', fontsize=12)
    ax.set_title(f'{model_name} - Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_predictions(
    contexts: List[np.ndarray],
    targets: List[np.ndarray],
    predictions: List[np.ndarray],
    model_name: str = "Model",
    n_samples: int = 4,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    複数サンプルの予測結果をサブプロットで表示

    Args:
        contexts: コンテキスト系列のリスト
        targets: 正解値のリスト
        predictions: 予測値のリスト
        model_name: モデル名
        n_samples: 表示するサンプル数
        save_path: 保存先のパス
        show: プロットを表示するか
    """
    n_samples = min(n_samples, len(contexts))
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        context = contexts[i]
        target = targets[i]
        prediction = predictions[i]

        context_x = np.arange(len(context))
        pred_x = np.arange(len(context), len(context) + len(target))

        ax.plot(context_x, context, 'gray', linewidth=2, alpha=0.7, label='Context')
        ax.plot(pred_x, target, 'b-', linewidth=2, label='Ground Truth', marker='o', markersize=2)
        ax.plot(pred_x, prediction, 'r--', linewidth=2, label='Prediction', marker='x', markersize=3)
        ax.axvline(x=len(context), color='green', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'Sample {i+1}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} - Multiple Predictions', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    targets: np.ndarray,
    predictions_dict: dict,
    context: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    複数モデルの予測結果を比較

    Args:
        targets: 正解値
        predictions_dict: {model_name: predictions}の辞書
        context: コンテキスト系列(オプション)
        save_path: 保存先のパス
        show: プロットを表示するか
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # コンテキスト
    offset = 0
    if context is not None:
        context_x = np.arange(len(context))
        ax.plot(context_x, context, 'gray', linewidth=2, alpha=0.7, label='Context')
        offset = len(context)

    # 正解値
    pred_x = np.arange(offset, offset + len(targets))
    ax.plot(pred_x, targets, 'b-', linewidth=3, label='Ground Truth', marker='o', markersize=3)

    # 各モデルの予測
    markers = ['x', 's', '^', 'v', 'D']
    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        marker = markers[idx % len(markers)]
        ax.plot(
            pred_x, predictions, '--',
            linewidth=2, label=f'{model_name}',
            marker=marker, markersize=4, alpha=0.8
        )

    if context is not None:
        ax.axvline(x=offset, color='green', linestyle=':', linewidth=2, alpha=0.7)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value (Temperature)', fontsize=12)
    ax.set_title('Model Comparison - Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
