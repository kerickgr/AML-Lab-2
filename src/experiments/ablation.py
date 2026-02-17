import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional


class AblationStudy:
    """Ablation study для оценки важности групп признаков"""

    def __init__(self, classifier=RandomForestClassifier(n_estimators=100, random_state=42)):
        self.classifier = classifier
        self.results = None

    def run_study(self, X: pd.DataFrame, y: pd.Series,
                  feature_groups: dict) -> pd.DataFrame:
        results = []

        # Все признаки
        all_score = self._evaluate(X, y)
        results.append({'group': 'All Features', 'accuracy': all_score})

        # Удаление каждой группы
        for group_name, features in feature_groups.items():
            present_features = [f for f in features if f in X.columns]
            if not present_features:
                continue

            X_without = X.drop(columns=present_features)
            score = self._evaluate(X_without, y)
            results.append({
                'group': f'Without {group_name}',
                'accuracy': score,
                'drop': all_score - score
            })

        # Только одна группа
        for group_name, features in feature_groups.items():
            present_features = [f for f in features if f in X.columns]
            if not present_features:
                continue

            X_only = X[present_features]
            score = self._evaluate(X_only, y)
            results.append({
                'group': f'Only {group_name}',
                'accuracy': score
            })

        self.results = pd.DataFrame(results)
        return self.results

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        X_clean = X.fillna(X.mean())
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(self.classifier, X_clean, y, cv=cv,
                                     scoring='balanced_accuracy')
            return np.mean(scores)
        except:
            return 0.0