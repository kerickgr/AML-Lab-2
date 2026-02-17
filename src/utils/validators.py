import numpy as np
import pandas as pd
from typing import Tuple, List


class DataValidator:
    """Валидация данных"""

    @staticmethod
    def validate_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """Проверка датасета"""
        if X is None or y is None:
            return False, "Данные отсутствуют"

        if len(X.shape) != 2:
            return False, f"X должен быть 2D, получено {len(X.shape)}D"

        if len(y.shape) != 1:
            return False, f"y должен быть 1D, получено {len(y.shape)}D"

        if X.shape[0] != len(y):
            return False, f"Несоответствие размеров: X={X.shape[0]}, y={len(y)}"

        if X.shape[0] < 10:
            return False, f"Слишком мало samples: {X.shape[0]}"

        if len(np.unique(y)) < 2:
            return False, "Нужно минимум 2 класса"

        if np.any(np.isinf(X)):
            return False, "X содержит бесконечные значения"

        return True, "Датасет корректен"

    @staticmethod
    def validate_meta_features(meta_features: dict) -> Tuple[bool, List[str]]:
        """Проверка мета-признаков"""
        issues = []

        if not meta_features:
            issues.append("Пустой словарь мета-признаков")
            return False, issues

        required = ['n_instances', 'n_features', 'n_classes']
        for req in required:
            if req not in meta_features:
                issues.append(f"Отсутствует обязательный признак: {req}")

        for key, value in meta_features.items():
            if not isinstance(value, (int, float, np.number)):
                issues.append(f"Нечисловое значение для {key}: {type(value)}")

            if isinstance(value, float) and np.isnan(value):
                issues.append(f"NaN значение для {key}")

        return len(issues) == 0, issues