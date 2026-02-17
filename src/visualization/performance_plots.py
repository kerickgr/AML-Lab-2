import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from pathlib import Path


class PerformancePlotter:
    """Визуализация производительности алгоритмов"""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None

    def plot_algorithm_comparison(self, results_df: pd.DataFrame):
        """Сравнение алгоритмов"""
        score_cols = [c for c in results_df.columns if c.startswith('score_')]

        if len(score_cols) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Box plot
        data = []
        for col in score_cols:
            for score in results_df[col].dropna():
                data.append({
                    'Algorithm': col.replace('score_', ''),
                    'Score': score
                })

        perf_df = pd.DataFrame(data)
        sns.boxplot(data=perf_df, x='Algorithm', y='Score', ax=axes[0, 0])
        axes[0, 0].set_title('Распределение производительности')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Лучший алгоритм
        if 'best_algorithm' in results_df.columns:
            counts = results_df['best_algorithm'].value_counts()
            axes[0, 1].pie(counts.values, labels=counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Доля датасетов, где алгоритм лучший')

        # 3. Сравнение попарно
        if len(score_cols) >= 2:
            algo1 = score_cols[0].replace('score_', '')
            algo2 = score_cols[1].replace('score_', '')

            axes[1, 0].scatter(results_df[score_cols[0]],
                               results_df[score_cols[1]],
                               alpha=0.6)
            axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[1, 0].set_xlabel(algo1)
            axes[1, 0].set_ylabel(algo2)
            axes[1, 0].set_title(f'{algo1} vs {algo2}')

        # 4. Тепловая карта
        perf_matrix = results_df[score_cols].T
        sns.heatmap(perf_matrix, annot=False, cmap='YlOrRd',
                    cbar_kws={'label': 'Performance'}, ax=axes[1, 1])
        axes[1, 1].set_xlabel('Датасет')
        axes[1, 1].set_ylabel('Алгоритм')
        axes[1, 1].set_title('Тепловая карта производительности')

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20):
        """Важность признаков"""
        if importance_df.empty:
            return

        if 'importance' in importance_df.columns:
            top_features = importance_df.nlargest(top_n, 'importance')
            imp_col = 'importance'
        else:
            top_features = importance_df.iloc[:, 0].nlargest(top_n).reset_index()
            top_features.columns = ['feature', 'importance']
            imp_col = 'importance'

        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.3)))

        # Горизонтальная столбчатая диаграмма
        y_pos = range(len(top_features))
        axes[0].barh(y_pos, top_features[imp_col], color='steelblue', alpha=0.7)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_xlabel('Важность')
        axes[0].set_title(f'Топ-{top_n} наиболее важных признаков')
        axes[0].invert_yaxis()

        # Накопительная важность
        cumulative = importance_df.sort_values('importance', ascending=False)['importance'].cumsum()
        cumulative = cumulative / cumulative.iloc[-1]

        axes[1].plot(range(1, len(cumulative) + 1), cumulative, 'bo-')
        axes[1].axhline(y=0.8, color='red', linestyle='--', label='80%')
        axes[1].axvline(x=top_n, color='green', linestyle='--', label=f'Top {top_n}')
        axes[1].set_xlabel('Количество признаков')
        axes[1].set_ylabel('Накопительная важность')
        axes[1].set_title('Накопительная важность признаков')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()