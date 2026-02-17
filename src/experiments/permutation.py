import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from typing import Dict, Tuple


class PermutationExperiment:
    """Эксперимент с перестановками для проверки инвариантности"""

    def __init__(self, extractor):
        self.extractor = extractor

    def run_experiment(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        results = {}

        # Оригинал
        results['original'] = self.extractor.extract(X, y)

        # Перемешивание строк
        X_rows, y_rows = shuffle(X, y, random_state=42)
        results['rows_shuffled'] = self.extractor.extract(X_rows, y_rows)

        # Перемешивание столбцов
        X_cols = X.copy()
        np.random.seed(42)
        np.random.shuffle(X_cols.T)
        results['cols_shuffled'] = self.extractor.extract(X_cols, y)

        # Перестановка меток
        y_perm = y.copy()
        unique = np.unique(y)
        if len(unique) > 1:
            perm = np.random.permutation(unique)
            for old, new in zip(unique, perm):
                y_perm[y == old] = new
        results['labels_permuted'] = self.extractor.extract(X, y_perm)

        return results

    def analyze_changes(self, results: Dict[str, Dict]) -> pd.DataFrame:
        changes = []
        original = results['original']

        for exp_type, exp_features in results.items():
            if exp_type == 'original':
                continue

            for feature in original:
                if feature in exp_features:
                    orig_val = original[feature]
                    exp_val = exp_features[feature]

                    if isinstance(orig_val, (int, float)):
                        changes.append({
                            'experiment': exp_type,
                            'feature': feature,
                            'original': orig_val,
                            'new': exp_val,
                            'change': exp_val - orig_val,
                            'relative_change': abs(exp_val - orig_val) / (abs(orig_val) + 1e-10)
                        })

        return pd.DataFrame(changes)