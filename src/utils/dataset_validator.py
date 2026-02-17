"""
Dataset validation utilities
"""
import numpy as np
from typing import Tuple, Optional


class DatasetValidator:
    """Валидация датасетов перед обработкой"""

    @staticmethod
    def is_valid(X: np.ndarray, y: np.ndarray,
                 min_samples: int = 10,
                 min_classes: int = 2,
                 min_samples_per_class: int = 2) -> Tuple[bool, str]:
        """
        Проверка датасета на пригодность для обучения

        Args:
            X: Матрица признаков
            y: Целевая переменная
            min_samples: Минимальное количество образцов
            min_classes: Минимальное количество классов
            min_samples_per_class: Минимальное количество образцов в каждом классе

        Returns:
            (is_valid, reason)
        """
        if X is None or y is None:
            return False, "Данные отсутствуют"

        if X.shape[0] < min_samples:
            return False, f"Слишком мало образцов: {X.shape[0]} < {min_samples}"

        unique_classes = np.unique(y)
        if len(unique_classes) < min_classes:
            return False, f"Слишком мало классов: {len(unique_classes)} < {min_classes}"

        # Проверка каждого класса
        for cls in unique_classes:
            count = np.sum(y == cls)
            if count < min_samples_per_class:
                return False, f"Класс {cls} имеет только {count} образцов (нужно {min_samples_per_class})"

        # Проверка на NaN
        if np.isnan(X).any():
            # Проверяем, можно ли удалить NaN
            mask = ~np.isnan(X).any(axis=1)
            if mask.sum() < min_samples:
                return False, "Слишком много NaN, после удаления останется мало данных"

        return True, "Датасет валиден"

    @staticmethod
    def get_dataset_info(X: np.ndarray, y: np.ndarray) -> dict:
        """Получение информации о датасете"""
        info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': {},
            'missing_values': np.isnan(X).sum(),
            'missing_ratio': np.isnan(X).sum() / X.size if X.size > 0 else 0
        }

        for cls in np.unique(y):
            info['class_distribution'][str(cls)] = int(np.sum(y == cls))

        return info
