"""Module pour le calcul de métriques de performance des générateurs"""

from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

def calculate_entropy(data: NDArray, bins: int = 10) -> float:
    """Calcule l'entropie de Shannon de la distribution"""
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_correlation(data: NDArray, lag: int = 1) -> float:
    """Calcule l'autocorrélation avec un décalage donné"""
    return np.corrcoef(data[:-lag], data[lag:])[0,1]

def calculate_generator_metrics(data: NDArray) -> Dict:
    """Calcule diverses métriques de qualité pour un générateur"""
    return {
        'entropy': calculate_entropy(data),
        'correlation_lag1': calculate_correlation(data, 1),
        'correlation_lag2': calculate_correlation(data, 2),
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data)
    }
