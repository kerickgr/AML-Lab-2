import numpy as np
from typing import Tuple, Dict, Optional


class NoiseGenerator:
    """Генерация шума в данных"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def add_gaussian_noise(self, X: np.ndarray, noise_level=0.1) -> np.ndarray:
        X_noisy = X.copy()
        for i in range(X.shape[1]):
            std = np.std(X[:, i])
            if std > 0:
                noise = np.random.normal(0, std * noise_level, X.shape[0])
                X_noisy[:, i] += noise
        return X_noisy

    def add_label_noise(self, y: np.ndarray, noise_ratio=0.1) -> np.ndarray:
        y_noisy = y.copy()
        n_noisy = int(len(y) * noise_ratio)
        unique = np.unique(y)

        if len(unique) <= 1:
            return y_noisy

        indices = np.random.choice(len(y), n_noisy, replace=False)
        for idx in indices:
            possible = [l for l in unique if l != y[idx]]
            if possible:
                y_noisy[idx] = np.random.choice(possible)

        return y_noisy

    def add_all_noise(self, X: np.ndarray, y: np.ndarray,
                      noise_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        if noise_config is None:
            noise_config = {'gaussian_level': 0.1, 'label_noise_ratio': 0.1}

        X_noisy = self.add_gaussian_noise(X, noise_config.get('gaussian_level', 0.1))
        y_noisy = self.add_label_noise(y, noise_config.get('label_noise_ratio', 0.1))

        return X_noisy, y_noisy