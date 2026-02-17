import numpy as np
from typing import Dict, Optional


class GeneralFeatures:
    """Базовые мета-признаки"""

    def extract(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        features = {}

        n_instances = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        features['n_instances'] = n_instances
        features['n_features'] = n_features
        features['n_classes'] = n_classes
        features['dimensionality'] = n_features / n_instances if n_instances > 0 else 0

        missing = np.isnan(X).sum()
        features['missing_ratio'] = missing / (n_instances * n_features) if n_features > 0 else 0

        return features