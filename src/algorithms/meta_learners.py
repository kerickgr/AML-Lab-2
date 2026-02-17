from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Any


class MetaLearners:
    """Мета-алгоритмы для обучения на мета-наборе"""

    AVAILABLE = {
        'Dummy': DummyClassifier(strategy='most_frequent'),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }

    def __init__(self, algorithm_names: List[str] = None):
        if algorithm_names is None:
            algorithm_names = ['Dummy', 'LogisticRegression', 'RandomForest']

        self.algorithms = {}
        for name in algorithm_names:
            if name in self.AVAILABLE:
                self.algorithms[name] = self.AVAILABLE[name]

    def get_all_algorithms(self) -> Dict[str, Any]:
        return self.algorithms.copy()

    def get_baseline(self) -> str:
        return 'Dummy'