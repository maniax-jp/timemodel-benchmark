"""
データローダーモジュール
ETTh1データセットの読み込みと前処理を行う
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class ETTh1Loader:
    """ETTh1データセットのローダー"""

    def __init__(
        self,
        data_path: str = "data/ETTh1.csv",
        target_column: str = "OT",
        context_length: int = 96,
        prediction_length: int = 96,
    ):
        """
        Args:
            data_path: データセットのパス
            target_column: 予測対象の列名(デフォルト: OT - 油温度)
            context_length: 入力系列の長さ
            prediction_length: 予測系列の長さ
        """
        self.data_path = data_path
        self.target_column = target_column
        self.context_length = context_length
        self.prediction_length = prediction_length

        # データの読み込み
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

        # 訓練/検証/テストデータに分割(12ヶ月/4ヶ月/4ヶ月)
        n = len(self.df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        self.train_data = self.df[:train_end]
        self.val_data = self.df[train_end:val_end]
        self.test_data = self.df[val_end:]

    def get_target_series(self, split: str = "test") -> np.ndarray:
        """
        指定されたsplitの目標変数の時系列データを取得

        Args:
            split: "train", "val", "test"のいずれか

        Returns:
            目標変数の時系列データ
        """
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        return data[self.target_column].values

    def create_samples(
        self,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        コンテキストとターゲットのサンプルペアを作成

        Args:
            split: "train", "val", "test"のいずれか
            num_samples: 作成するサンプル数(Noneの場合は最大数)

        Returns:
            contexts: (n_samples, context_length)
            targets: (n_samples, prediction_length)
        """
        series = self.get_target_series(split)

        # 作成可能な最大サンプル数
        max_samples = len(series) - self.context_length - self.prediction_length + 1

        if num_samples is None:
            num_samples = max_samples
        else:
            num_samples = min(num_samples, max_samples)

        contexts = []
        targets = []

        # 等間隔でサンプリング
        indices = np.linspace(
            0,
            max_samples - 1,
            num_samples,
            dtype=int
        )

        for idx in indices:
            context = series[idx : idx + self.context_length]
            target = series[
                idx + self.context_length :
                idx + self.context_length + self.prediction_length
            ]
            contexts.append(context)
            targets.append(target)

        return np.array(contexts), np.array(targets)

    def get_single_sample(
        self,
        split: str = "test",
        index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        単一のサンプルを取得(テスト用)

        Args:
            split: "train", "val", "test"のいずれか
            index: サンプルのインデックス

        Returns:
            context: (context_length,)
            target: (prediction_length,)
        """
        series = self.get_target_series(split)

        context = series[index : index + self.context_length]
        target = series[
            index + self.context_length :
            index + self.context_length + self.prediction_length
        ]

        return context, target

    def get_info(self) -> dict:
        """データセット情報を取得"""
        return {
            "total_length": len(self.df),
            "train_length": len(self.train_data),
            "val_length": len(self.val_data),
            "test_length": len(self.test_data),
            "target_column": self.target_column,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "columns": list(self.df.columns),
        }
