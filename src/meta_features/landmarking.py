import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict


class LandmarkingFeatures:
    """Ландмаркинг - производительность простых моделей"""

    def __init__(self, cv_folds=3):
        self.cv_folds = cv_folds

    def extract(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        features = {}

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        features['decision_tree_accuracy'] = self._evaluate(dt, X, y)

        # Naive Bayes
        nb = GaussianNB()
        features['naive_bayes_accuracy'] = self._evaluate(nb, X, y)

        # 1-NN
        knn = KNeighborsClassifier(n_neighbors=1)
        features['one_nn_accuracy'] = self._evaluate(knn, X, y)

        return features

    def _evaluate(self, clf, X, y):
        try:
            n_folds = min(self.cv_folds, np.min(np.bincount(y)))
            if n_folds < 2:
                return 0.0
            scores = cross_val_score(clf, X, y, cv=n_folds, scoring='balanced_accuracy')
            return np.mean(scores)
        except:
            return 0.0