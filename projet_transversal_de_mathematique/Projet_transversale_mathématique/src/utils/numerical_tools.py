import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable

def solve_system(func: Callable,
                initial_state: np.ndarray,
                t_span: Tuple[float, float],
                E: float,
                **kwargs) -> dict:
    """Résout numériquement le système d'EDO."""
    try:
        sol = solve_ivp(func, t_span, initial_state, args=(E,), **kwargs)
        return {
            'success': True,
            'solution': sol,
            'message': 'Success'
        }
    except Exception as e:
        return {
            'success': False,
            'solution': None,
            'message': str(e)
        }