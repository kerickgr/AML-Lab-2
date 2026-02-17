import numpy as np
from typing import Dict


class StatisticalFeatures:
    """Статистические мета-признаки"""

    def extract(self, X: np.ndarray) -> Dict[str, float]:
        features = {}

        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        features['mean_mean'] = np.mean(means)
        features['mean_std'] = np.std(means)
        features['std_mean'] = np.mean(stds)
        features['std_std'] = np.std(stds)

        skewness = self._skewness(X)
        features['skewness_mean'] = np.mean(skewness)

        kurtosis = self._kurtosis(X)
        features['kurtosis_mean'] = np.mean(kurtosis)

        # Корреляция
        if X.shape[1] > 1:
            corr = np.corrcoef(X.T)
            np.fill_diagonal(corr, 0)
            features['correlation_mean'] = np.mean(np.abs(corr))

        return features

    def _skewness(self, X):
        n = X.shape[0]
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        return np.sum(((X - mean) / std) ** 3, axis=0) / n

    def _kurtosis(self, X):
        n = X.shape[0]
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        return np.sum(((X - mean) / std) ** 4, axis=0) / n - 3