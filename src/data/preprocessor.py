import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional


class DataPreprocessor:
    """Предобработка данных"""

    def __init__(self, handle_missing='remove', scale_features=True, encode_labels=True):
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.encode_labels = encode_labels
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoder = LabelEncoder() if encode_labels else None

    def preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Обработка пропусков
        if self.handle_missing == 'remove':
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X, y = X[mask], y[mask]

        # Кодирование меток
        if self.encode_labels:
            y = self.label_encoder.fit_transform(y)

        # Масштабирование
        if self.scale_features:
            X = self.scaler.fit_transform(X)

        return X, y