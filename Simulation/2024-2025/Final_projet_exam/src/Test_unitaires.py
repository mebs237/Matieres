import pytest
import numpy as np
from Analysing import analyse_sequence, evaluate_generator , AnalysisResult , EvaluationResult
from Generators import PythonGenerator, OurGenerator
from Tests import Chi2Test, KSTest

@pytest.fixture
def random_sequence():
    return np.random.random(1000)

@pytest.fixture
def uniform_sequence():
    return np.linspace(0, 1, 1000)

def test_analyse_sequence_basic(random_sequence):
    """Test basique de analyse_sequence"""
    result = analyse_sequence(random_sequence)
    assert isinstance(result, AnalysisResult)
    assert 0 <= result.stats['mean_p_value'] <= 1
    assert 0 <= result.stats['accept_ratio'] <= 1

def test_analyse_sequence_uniform(uniform_sequence):
    """Test avec une séquence uniforme parfaite"""
    result = analyse_sequence(uniform_sequence, parallel=False)
    # Une séquence parfaitement uniforme devrait avoir des p-values élevées
    assert result.stats['mean_p_value'] > 0.5
    assert result.stats['accept_ratio'] > 0.9

def test_evaluate_generator():
    """Test basique de evaluate_generator"""
    gen = PythonGenerator()
    result = evaluate_generator(gen, n_repeat=5, seq_len=1000, parallel=False)
    assert isinstance(result, EvaluationResult)
    assert result.generator_name == "Python_Generator"

def test_parallel_analysis(random_sequence):
    """Test que le mode parallèle fonctionne"""
    result_parallel = analyse_sequence(random_sequence, parallel=True)
    result_serial = analyse_sequence(random_sequence, parallel=False)
    # Les résultats devraient être similaires
    assert np.isclose(result_parallel.stats['mean_p_value'],
                     result_serial.stats['mean_p_value'],
                     rtol=0.1)