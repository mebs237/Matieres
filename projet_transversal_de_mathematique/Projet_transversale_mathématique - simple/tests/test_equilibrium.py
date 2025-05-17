import pytest
import numpy as np
from src0.models.equilibrium import RangelandSystem

def test_system_initialization():
    system = RangelandSystem()
    assert system.Rg == 0.27
    assert system.Rw == 0.4

def test_equation():
    system = RangelandSystem()
    state = np.array([0.5, 0.5])
    result = system.equation(0, state, E=0.1)
    assert isinstance(result, np.ndarray)
    assert len(result) == 2