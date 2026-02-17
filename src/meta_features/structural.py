import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from typing import Dict


class StructuralFeatures:
    """Структурные мета-признаки"""

    def extract(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        features = {}

        # Энтропия классов
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        features['class_entropy'] = entropy(probs).item()

        # Взаимная информация
        try:
            mi = mutual_info_classif(X, y, random_state=42)
            features['mutual_info_mean'] = np.mean(mi)
            features['mutual_info_std'] = np.std(mi)
        except:
            features['mutual_info_mean'] = 0
            features['mutual_info_std'] = 0

        return features