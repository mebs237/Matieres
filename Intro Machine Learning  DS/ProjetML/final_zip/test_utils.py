import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import os

# Import des fonctions à tester
from utilitaires import (low_entropy_cols, to_missing_values, quasi_constant_cols,low_information_cols, apply_pca, mutual_info_norm, mutual_info_matrix, get_dependant_pairs, remove_redundant_cols)

# Fixtures pour les données de test
@pytest.fixture
def sample_df():
    """Crée un DataFrame de test avec différents types de colonnes"""
    return pd.DataFrame({
        'constant_num': [1] * 100,
        'low_var_num': [1] * 95 + [2] * 5,
        'normal_num': np.random.normal(0, 1, 100),
        'categorical': ['A'] * 80 + ['B'] * 20,
        'many_missing': [np.nan] * 80 + [1] * 20,
        'normal_cat': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.normal(0, 1, 100)
    })

@pytest.fixture
def numerical_df():
    """Crée un DataFrame avec uniquement des colonnes numériques"""
    return pd.DataFrame({
        'x1': np.random.normal(0, 1, 100),
        'x2': np.random.normal(0, 1, 100),
        'target': np.random.normal(0, 1, 100)
    })

# Tests pour low_entropy_cols
def test_low_entropy_cols(sample_df):
    """Test de la fonction low_entropy_cols"""
    result = low_entropy_cols(sample_df, threshold=0.5)
    assert 'constant_num' in result
    assert 'low_var_num' in result
    assert 'normal_num' not in result

def test_low_entropy_cols_empty_df():
    """Test avec un DataFrame vide"""
    empty_df = pd.DataFrame()
    result = low_entropy_cols(empty_df)
    assert result == []

# Tests pour to_missing_values
def test_to_missing_values(sample_df):
    """Test de la fonction to_missing_values"""
    result = to_missing_values(sample_df, threshold=0.7)
    assert 'many_missing' in result
    assert 'normal_num' not in result

def test_to_missing_values_no_missing():
    """Test avec un DataFrame sans valeurs manquantes"""
    df = pd.DataFrame({'col': range(10)})
    result = to_missing_values(df)
    assert result == []

# Tests pour quasi_constant_cols
def test_quasi_constant_cols(sample_df):
    """Test de la fonction quasi_constant_cols"""
    result = quasi_constant_cols(sample_df, threshold=0.05)
    assert 'constant_num' in result
    assert 'normal_num' not in result

# Tests pour mutual_info_norm
def test_mutal_info_norm(sample_df):
    """Test de la fonction mutual_info_norm"""
    # Test avec deux variables numériques
    mi = mutual_info_norm(sample_df['normal_num'], sample_df['normal_num'])
    assert mi == pytest.approx(1.0, rel=1e-12)

    # Test avec une variable catégorielle et une numérique
    mi = mutual_info_norm(sample_df['categorical'], sample_df['normal_num'])
    assert 0 <= mi <= 1

# Tests pour mutual_info_matrix
def test_mutal_info_matrix(numerical_df):
    """Test de la fonction mutual_info_matrix"""
    result = mutual_info_matrix(numerical_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)  # Pour un DataFrame avec 3 colonnes
    #assert (result.diagonal() == 1.0).all()  # La diagonale doit être 1
    assert (result >= 0).all().all()  # Toutes les valeurs doivent être >= 0
    assert (result <= 1).all().all()  # Toutes les valeurs doivent être <= 1

# Tests pour apply_pca
def test_apply_pca(numerical_df):
    """Test de la fonction apply_pca"""
    result_df, pca = apply_pca(numerical_df,
                              columns=['x1', 'x2'],
                              n_components=2,
                              verbose=False)

    assert 'PC1' in result_df.columns
    assert 'PC2' in result_df.columns
    assert 'target' in result_df.columns
    assert pca.n_components_ == 2

def test_apply_pca_invalid_input():
    """Test de apply_pca avec des entrées invalides"""
    df = pd.DataFrame({'a': ['x', 'y']})
    with pytest.raises(ValueError):
        apply_pca(df, columns=['a'], handle_categorical='drop')

# Tests pour remove_redundant_cols
def test_remove_redundant_cols(sample_df):
    """Test de la fonction remove_redundant_cols"""
    keep, drop = remove_redundant_cols(sample_df,
                                     target='target',
                                     threshold=0.8,
                                     verbose=False)

    assert isinstance(keep, list)
    assert isinstance(drop, list)
    assert set(keep).union(set(drop)) == set(sample_df.columns)
    assert 'target' in keep  # La cible doit toujours être conservée

# Tests pour get_dependant_pairs
def test_get_dependant_pairs(sample_df):
    """Test de la fonction get_dependant_pairs"""
    pairs = get_dependant_pairs(sample_df, threshold=0.5, verbose=False)
    assert isinstance(pairs, list)
    assert all(len(pair) == 3 for pair in pairs)  # Chaque paire doit avoir 3 éléments
    assert all(0 <= score <= 1 for _, _, score in pairs)  # Les scores doivent être entre 0 et 1

if __name__ == '__main__':
    pytest.main([__file__])