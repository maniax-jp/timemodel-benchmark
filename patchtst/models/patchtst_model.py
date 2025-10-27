"""
PatchTSTモデル実装
HuggingFace transformersのPatchTSTを使用した時系列予測

PatchTSTは学習が必要なモデルなので、データの90%で学習し10%で評価します。
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from typing import Optional, Dict
import time
import os
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """時系列データセット"""

    def __init__(self, contexts: np.ndarray, targets: np.ndarray):
        """
        Args:
            contexts: (n_samples, context_length)
            targets: (n_samples, prediction_length)
        """
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        # PatchTSTが期待する形式で直接返す
        # past_values: (context_length, num_channels)
        # future_values: (prediction_length, num_channels)
        context = torch.FloatTensor(self.contexts[idx]).unsqueeze(1)  # (context_length, 1)
        target = torch.FloatTensor(self.targets[idx]).unsqueeze(1)  # (prediction_length, 1)

        return {
            'past_values': context,
            'future_values': target
        }


class PatchTSTPredictor:
    """PatchTSTを使用した時系列予測"""

    def __init__(
        self,
        context_length: int = 96,
        horizon_length: int = 96,
        patch_length: int = 16,
        num_hidden_layers: int = 3,
        d_model: int = 128,
        num_attention_heads: int = 4,
        device: str = "cuda",
        model_path: Optional[str] = None,
    ):
        """
        Args:
            context_length: 入力コンテキストの長さ
            horizon_length: 予測ホライゾンの長さ
            patch_length: パッチの長さ
            num_hidden_layers: Transformerレイヤー数
            d_model: モデルの次元数
            num_attention_heads: アテンションヘッド数
            device: "cuda" または "cpu"
            model_path: 保存されたモデルのパス（Noneの場合は新規作成）
        """
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.device = device
        self.patch_length = patch_length
        self.num_hidden_layers = num_hidden_layers
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads

        print(f"PatchTSTモデルを初期化中... (device: {device})")
        print(f"Context length: {context_length}, Horizon: {horizon_length}")
        print(f"Patch length: {patch_length}, Layers: {num_hidden_layers}, d_model: {d_model}")

        # 保存されたモデルがあれば読み込む
        if model_path and os.path.exists(model_path):
            print(f"保存されたモデルを読み込み中: {model_path}")
            self.model = PatchTSTForPrediction.from_pretrained(model_path)
            self.config = self.model.config
            print("保存されたモデルの読み込みが完了しました")
        else:
            # PatchTSTの設定
            self.config = PatchTSTConfig(
                num_input_channels=1,  # 単変量
                context_length=context_length,
                prediction_length=horizon_length,
                patch_length=patch_length,
                patch_stride=patch_length,  # ノンオーバーラップ
                num_hidden_layers=num_hidden_layers,
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                ffn_dim=d_model * 4,
                dropout=0.1,
                head_dropout=0.1,
                pooling_type="mean",
                channel_attention=False,
                scaling="std",
                loss="mse",
                pre_norm=True,
                norm_type="batchnorm",
            )

            # モデルの初期化
            self.model = PatchTSTForPrediction(self.config)
            print("PatchTSTモデルの初期化が完了しました")

        if device == "cuda":
            self.model = self.model.cuda()

        self.model.eval()  # デフォルトは評価モード

    def train(
        self,
        train_contexts: np.ndarray,
        train_targets: np.ndarray,
        val_contexts: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        output_dir: str = "patchtst/checkpoints",
        verbose: bool = True
    ) -> Dict:
        """
        モデルの学習

        Args:
            train_contexts: 訓練データのコンテキスト (n_samples, context_length)
            train_targets: 訓練データのターゲット (n_samples, prediction_length)
            val_contexts: 検証データのコンテキスト (optional)
            val_targets: 検証データのターゲット (optional)
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            early_stopping_patience: Early stoppingの待機エポック数
            output_dir: チェックポイント保存ディレクトリ
            verbose: 詳細出力

        Returns:
            訓練履歴
        """
        if verbose:
            print(f"\n=== PatchTST Training ===")
            print(f"Training samples: {len(train_contexts)}")
            if val_contexts is not None:
                print(f"Validation samples: {len(val_contexts)}")
            print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        # データセットの作成
        train_dataset = TimeSeriesDataset(train_contexts, train_targets)
        eval_dataset = None
        if val_contexts is not None and val_targets is not None:
            eval_dataset = TimeSeriesDataset(val_contexts, val_targets)

        # 訓練設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if eval_dataset else "no",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="no",  # チェックポイント保存を無効化して高速化
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to=[],  # WandB等を使わない
            disable_tqdm=not verbose,
        )

        # Trainerの設定
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # 訓練実行
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Final train loss: {train_result.training_loss:.4f}")

        # 訓練履歴を返す
        history = {
            "training_time": training_time,
            "final_train_loss": train_result.training_loss,
        }

        return history

    def save_model(self, save_path: str, verbose: bool = True):
        """
        学習済みモデルを保存

        Args:
            save_path: 保存先のパス
            verbose: 詳細出力
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nモデルを保存中: {save_path}")

        self.model.save_pretrained(save_path)

        if verbose:
            print("モデルの保存が完了しました")

    @classmethod
    def load_model(
        cls,
        model_path: str,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        保存されたモデルを読み込み

        Args:
            model_path: モデルのパス
            device: "cuda" または "cpu"
            verbose: 詳細出力

        Returns:
            PatchTSTPredictor
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

        if verbose:
            print(f"保存されたモデルを読み込み中: {model_path}")

        # configを読み込んで設定を取得
        config = PatchTSTConfig.from_pretrained(model_path)

        predictor = cls(
            context_length=config.context_length,
            horizon_length=config.prediction_length,
            patch_length=config.patch_length,
            num_hidden_layers=config.num_hidden_layers,
            d_model=config.d_model,
            num_attention_heads=config.num_attention_heads,
            device=device,
            model_path=model_path
        )

        if verbose:
            print("モデルの読み込みが完了しました")

        return predictor

    def predict_single(self, context: np.ndarray) -> np.ndarray:
        """
        単一サンプルの予測

        Args:
            context: コンテキスト系列 (context_length,)

        Returns:
            予測値 (horizon_length,)
        """
        self.model.eval()

        # (1, context_length, 1) の形式に変換
        context_tensor = torch.FloatTensor(context).unsqueeze(0).unsqueeze(2)

        if self.device == "cuda":
            context_tensor = context_tensor.cuda()

        with torch.no_grad():
            outputs = self.model(past_values=context_tensor)
            prediction = outputs.prediction_outputs.squeeze(0).squeeze(1).cpu().numpy()

        return prediction[:self.horizon_length]

    def predict_batch(self, contexts: np.ndarray) -> np.ndarray:
        """
        バッチ予測

        Args:
            contexts: コンテキスト系列 (batch_size, context_length)

        Returns:
            予測値 (batch_size, horizon_length)
        """
        self.model.eval()

        # (batch_size, context_length, 1) の形式に変換
        context_tensor = torch.FloatTensor(contexts).unsqueeze(2)

        if self.device == "cuda":
            context_tensor = context_tensor.cuda()

        with torch.no_grad():
            outputs = self.model(past_values=context_tensor)
            predictions = outputs.prediction_outputs.squeeze(2).cpu().numpy()

        return predictions[:, :self.horizon_length]

    def benchmark(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True
    ) -> dict:
        """
        ベンチマーク実行（学習済みモデルでの評価）

        Args:
            contexts: コンテキスト系列 (n_samples, context_length)
            targets: 正解値 (n_samples, horizon_length)
            verbose: 詳細出力

        Returns:
            {
                "predictions": 予測値,
                "targets": 正解値,
                "inference_time": 推論時間(秒)
            }
        """
        self.model.eval()

        n_samples = contexts.shape[0]

        if verbose:
            print(f"\nPatchTSTベンチマーク実行中...")
            print(f"サンプル数: {n_samples}")
            print(f"Context length: {self.context_length}")
            print(f"Horizon length: {self.horizon_length}")

        # 推論時間計測
        start_time = time.time()
        predictions = self.predict_batch(contexts)
        inference_time = time.time() - start_time

        if verbose:
            print(f"推論完了: {inference_time:.3f}秒")
            print(f"1サンプルあたり: {inference_time/n_samples:.3f}秒")

        return {
            "predictions": predictions,
            "targets": targets,
            "inference_time": inference_time,
        }
