from dataclasses import dataclass
import numpy as np
from .constants import *

@dataclass
class RangelandSystem:
    """Système de dynamique des pâturages."""

    Rg: float = RG
    Rw: float = RW
    a: float = A
    b: float = B
    c: float = C

    def equation(self, t: float, state: np.ndarray, E: float) -> np.ndarray:
        """Équations différentielles du système."""
        g, w = state

        if E == 0 and g == 0:
            dgdt = self.Rg * g * (1 - g - self.b * w)
            dwdt = self.Rw * w * (1 - w - self.c * g)
        else:
            dgdt = self.Rg * g * (1 - g - self.b * w * (E + g)/(self.a * E + g))
            dwdt = self.Rw * w * (1 - w - self.c * g * (self.a * E + g)/(E + g))

        return np.array([dgdt, dwdt])