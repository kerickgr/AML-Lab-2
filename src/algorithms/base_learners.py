from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, List, Any


class BaseLearners:
    """Базовые алгоритмы для оценки на датасетах"""

    AVAILABLE = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    def __init__(self, algorithm_names: List[str] = None):
        if algorithm_names is None:
            algorithm_names = ['LogisticRegression', 'RandomForest', 'KNN']

        self.algorithms = {}
        for name in algorithm_names:
            if name in self.AVAILABLE:
                self.algorithms[name] = self.AVAILABLE[name]

    def get_all_algorithms(self) -> Dict[str, Any]:
        return self.algorithms.copy()

    def get_algorithm_names(self) -> List[str]:
        return list(self.algorithms.keys())