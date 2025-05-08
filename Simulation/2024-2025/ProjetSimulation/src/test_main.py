from main import group_digits , read_decimal_file
import numpy as np

def test_read_decimal():
    """Test de la fonction read_decimal_file"""
    seq = read_decimal_file('data/e2M.txt')
    print(f"Nombre de chiffres lus : {len(seq)}")
    print(f"Premiers chiffres : {seq}")

# Exemple d'utilisation
def test_group_digits():
    """Test de la fonction group_digits"""
    # Test avec différents cas
    test_cases = [
        (np.array([1,2,3,4,5,6,7]), 3),
        (np.array([1,2,3,4,5]), 2),
        (np.array([9,8,7,6,5,4,3,2,1]), 4)
    ]

    for arr, k in test_cases:
        result = group_digits(arr, k)
        print(f"\nEntrée: {arr}")
        print(f"k = {k}")
        print(f"Résultat: {result}")
