"""
Модуль для оценки алгоритмов на датасетах
"""
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AlgorithmEvaluator:
    """Оценка алгоритмов на датасетах"""

    def __init__(self,
                 algorithms: Dict[str, Any],
                 cv_folds: int = 5,
                 test_size: float = 0.3,
                 random_state: int = 42,
                 scoring: str = 'balanced_accuracy'):
        """
        Инициализация оценщика

        Args:
            algorithms: Словарь алгоритмов
            cv_folds: Количество фолдов для кросс-валидации
            test_size: Размер тестовой выборки
            random_state: Random seed
            scoring: Метрика оценки
        """
        self.algorithms = algorithms
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring

        # Доступные метрики
        self.metrics = {
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score,
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
        }

    def evaluate_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оценка с помощью кросс-валидации с адаптивными параметрами"""
        scores = {}

        # Получаем информацию о классах
        class_counts = np.bincount(y)
        min_class_size = np.min(class_counts)
        n_classes = len(class_counts)

        # Адаптивное количество фолдов
        n_folds = min(self.cv_folds, min_class_size)
        if n_folds < 2:
            # Если данных太少, используем holdout вместо CV
            logger.debug(
                f"Слишком мало данных для CV (min_class_size={min_class_size}), используем holdout")
            return self.evaluate_holdout(X, y)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        for name, algorithm in self.algorithms.items():
            try:
                # Специальная обработка для разных алгоритмов
                X_use = X
                algo_to_use = algorithm

                if name == 'KNN':
                    # Адаптируем n_neighbors к размеру данных
                    n_neighbors = min(5, X.shape[0] - 1, min_class_size)
                    if n_neighbors < 1:
                        n_neighbors = 1
                    from sklearn.neighbors import KNeighborsClassifier
                    algo_to_use = KNeighborsClassifier(n_neighbors=n_neighbors)

                elif name == 'LogisticRegression':
                    # Масштабирование для Logistic Regression
                    scaler = StandardScaler()
                    X_use = scaler.fit_transform(X)
                    # Увеличиваем max_iter
                    from sklearn.linear_model import LogisticRegression
                    algo_to_use = LogisticRegression(max_iter=2000, random_state=self.random_state)

                cv_scores = cross_val_score(
                    algo_to_use, X_use, y,
                    cv=cv,
                    scoring=self.scoring,
                    error_score=0.0
                )
                scores[name] = np.mean(cv_scores)
                logger.debug(f"{name}: {scores[name]:.4f} (±{np.std(cv_scores):.4f})")

            except Exception as e:
                logger.warning(f"Ошибка при оценке {name}: {e}")
                scores[name] = 0.0

        return scores

    def evaluate_holdout(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оценка с разделением на train/test для маленьких датасетов"""
        scores = {}

        # Для очень маленьких датасетов используем bootstrap
        if X.shape[0] < 10:
            logger.debug(f"Очень маленький датасет ({X.shape[0]} объектов), используем bootstrap")
            return self._evaluate_bootstrap(X, y)

        # Адаптивный размер тестовой выборки
        test_size = min(0.3, 1.0 / X.shape[0])

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            for name, algorithm in self.algorithms.items():
                try:
                    # Специальная обработка для разных алгоритмов
                    X_train_use = X_train
                    X_test_use = X_test
                    algo_to_use = algorithm

                    if name == 'KNN':
                        # Адаптируем n_neighbors
                        n_neighbors = min(3, X_train.shape[0] - 1)
                        if n_neighbors < 1:
                            n_neighbors = 1
                        from sklearn.neighbors import KNeighborsClassifier
                        algo_to_use = KNeighborsClassifier(n_neighbors=n_neighbors)

                    elif name == 'LogisticRegression':
                        # Масштабирование
                        scaler = StandardScaler()
                        X_train_use = scaler.fit_transform(X_train)
                        X_test_use = scaler.transform(X_test)
                        from sklearn.linear_model import LogisticRegression
                        algo_to_use = LogisticRegression(max_iter=2000,
                                                         random_state=self.random_state)

                    algo_to_use.fit(X_train_use, y_train)
                    y_pred = algo_to_use.predict(X_test_use)

                    if self.scoring == 'balanced_accuracy':
                        scores[name] = balanced_accuracy_score(y_test, y_pred)
                    else:
                        scores[name] = accuracy_score(y_test, y_pred)

                    logger.debug(f"{name}: {scores[name]:.4f}")

                except Exception as e:
                    logger.warning(f"Ошибка в {name}: {e}")
                    scores[name] = 0.0

        except Exception as e:
            logger.error(f"Ошибка разделения данных: {e}")
            for name in self.algorithms:
                scores[name] = 0.0

        return scores

    def _evaluate_bootstrap(self, X: np.ndarray, y: np.ndarray, n_iter: int = 10) -> Dict[
        str, float]:
        """Bootstrap оценка для очень маленьких датасетов"""
        scores = {name: [] for name in self.algorithms}

        for i in range(n_iter):
            # Bootstrap выборка
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # Оценка
            for name, algorithm in self.algorithms.items():
                try:
                    # Адаптация алгоритмов
                    X_boot_use = X_boot
                    algo_to_use = algorithm

                    if name == 'KNN':
                        n_neighbors = min(3, X_boot.shape[0] - 1)
                        if n_neighbors < 1:
                            n_neighbors = 1
                        from sklearn.neighbors import KNeighborsClassifier
                        algo_to_use = KNeighborsClassifier(n_neighbors=n_neighbors)

                    elif name == 'LogisticRegression':
                        scaler = StandardScaler()
                        X_boot_use = scaler.fit_transform(X_boot)
                        from sklearn.linear_model import LogisticRegression
                        algo_to_use = LogisticRegression(max_iter=2000,
                                                         random_state=self.random_state)

                    # Используем кросс-валидацию на bootstrap выборке
                    n_folds = min(3, np.min(np.bincount(y_boot)))
                    if n_folds >= 2:
                        cv_scores = cross_val_score(algo_to_use, X_boot_use, y_boot,
                                                    cv=n_folds, scoring=self.scoring)
                        scores[name].append(np.mean(cv_scores))
                    else:
                        # Если классов слишком мало, обучаем на всей выборке и оцениваем на ней же
                        algo_to_use.fit(X_boot_use, y_boot)
                        y_pred = algo_to_use.predict(X_boot_use)
                        scores[name].append(balanced_accuracy_score(y_boot, y_pred))

                except Exception as e:
                    logger.debug(f"Ошибка в bootstrap для {name}: {e}")
                    continue

        # Усредняем результаты
        final_scores = {}
        for name in self.algorithms:
            if scores[name]:
                final_scores[name] = np.mean(scores[name])
                logger.debug(
                    f"{name} (bootstrap): {final_scores[name]:.4f} (на основе {len(scores[name])} итераций)")
            else:
                final_scores[name] = 0.0
                logger.warning(f"{name}: нет успешных bootstrap итераций")

        return final_scores

    def evaluate_all(self, X: np.ndarray, y: np.ndarray, method: str = 'cv') -> Dict[str, float]:
        """
        Оценка всех алгоритмов

        Args:
            X: Матрица признаков
            y: Целевая переменная
            method: Метод оценки ('cv' или 'holdout')

        Returns:
            Словарь с оценками алгоритмов
        """
        if method == 'cv':
            return self.evaluate_cv(X, y)
        elif method == 'holdout':
            return self.evaluate_holdout(X, y)
        else:
            raise ValueError(f"Неизвестный метод оценки: {method}")

    def get_best_algorithm(self, X: np.ndarray, y: np.ndarray, method: str = 'cv') -> Tuple[
        str, float]:
        """
        Получение лучшего алгоритма для датасета

        Args:
            X: Матрица признаков
            y: Целевая переменная
            method: Метод оценки

        Returns:
            Кортеж (имя_алгоритма, оценка)
        """
        scores = self.evaluate_all(X, y, method)
        best_name = max(scores, key=scores.get)
        return best_name, scores[best_name]

    def evaluate_multiple_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[
        str, Dict[str, float]]:
        """
        Оценка алгоритмов по нескольким метрикам

        Args:
            X: Матрица признаков
            y: Целевая переменная

        Returns:
            Словарь {алгоритм: {метрика: значение}}
        """
        results = {}

        # Используем holdout для множественных метрик
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        for name, algorithm in self.algorithms.items():
            results[name] = {}

            try:
                # Специальная обработка для разных алгоритмов
                X_train_use = X_train
                X_test_use = X_test
                algo_to_use = algorithm

                if name == 'KNN':
                    n_neighbors = min(5, X_train.shape[0] - 1)
                    if n_neighbors < 1:
                        n_neighbors = 1
                    from sklearn.neighbors import KNeighborsClassifier
                    algo_to_use = KNeighborsClassifier(n_neighbors=n_neighbors)

                elif name == 'LogisticRegression':
                    scaler = StandardScaler()
                    X_train_use = scaler.fit_transform(X_train)
                    X_test_use = scaler.transform(X_test)
                    from sklearn.linear_model import LogisticRegression
                    algo_to_use = LogisticRegression(max_iter=2000, random_state=self.random_state)

                algo_to_use.fit(X_train_use, y_train)
                y_pred = algo_to_use.predict(X_test_use)

                # Вычисляем все метрики
                for metric_name, metric_func in self.metrics.items():
                    try:
                        results[name][metric_name] = metric_func(y_test, y_pred)
                    except Exception as e:
                        logger.debug(f"Ошибка вычисления {metric_name} для {name}: {e}")
                        results[name][metric_name] = 0.0

            except Exception as e:
                logger.warning(f"Ошибка при оценке {name}: {e}")
                for metric_name in self.metrics:
                    results[name][metric_name] = 0.0

        return results
