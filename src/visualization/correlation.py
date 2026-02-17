import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from pathlib import Path


class CorrelationPlotter:
    """Визуализация корреляционных матриц"""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None

    def plot_correlation_matrix(self, df: pd.DataFrame,
                                figsize: Tuple[int, int] = (14, 12),
                                method: str = 'pearson',
                                annot: bool = True):
        """Построение корреляционной матрицы"""
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.shape[1] < 2:
            print("Недостаточно признаков")
            return

        corr = df_numeric.corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=figsize)
        sns.heatmap(corr, mask=mask, annot=annot, fmt='.2f',
                    cmap='RdBu_r', center=0, square=True,
                    cbar_kws={"shrink": 0.8})
        plt.title(f'{method.capitalize()} корреляционная матрица')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / f'correlation_{method}.png',
                        dpi=150, bbox_inches='tight')
        plt.show()

    def plot_high_correlations(self, df: pd.DataFrame,
                               threshold: float = 0.8):
        """Визуализация высоких корреляций"""
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.shape[1] < 2:
            return

        corr = df_numeric.corr()

        # Поиск пар с высокой корреляцией
        high_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > threshold:
                    high_pairs.append({
                        'feature1': corr.columns[i],
                        'feature2': corr.columns[j],
                        'correlation': corr.iloc[i, j]
                    })

        if not high_pairs:
            print(f"Нет корреляций выше {threshold}")
            return

        high_df = pd.DataFrame(high_pairs)
        high_df = high_df.sort_values('correlation', key=abs, ascending=False)

        plt.figure(figsize=(10, max(6, len(high_df) * 0.3)))
        colors = ['red' if x > 0 else 'blue' for x in high_df['correlation']]

        y_pos = range(len(high_df))
        plt.barh(y_pos, high_df['correlation'], color=colors, alpha=0.7)
        plt.yticks(y_pos, [f"{row['feature1']}\n{row['feature2']}"
                           for _, row in high_df.iterrows()])
        plt.xlabel('Корреляция')
        plt.title(f'Пары признаков с |корреляцией| > {threshold}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / f'high_correlations_{threshold}.png',
                        dpi=150, bbox_inches='tight')
        plt.show()