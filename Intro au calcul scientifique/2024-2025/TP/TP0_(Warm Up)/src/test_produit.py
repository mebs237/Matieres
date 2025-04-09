# test_produit.py
import pytest
from produit import produit, add, power_set
from typeguard import check_type

#------------------------------------------------------------------

class TestProduit:
    """Tests pour la fonction produit()."""

    # cas normaux et limites
    @pytest.mark.parametrize("x, y, expected", [
        ([1, 2], ['a', 'b'], [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]),
        ([], [1, 2], []),   # x est vide
        ([1, 2], [], []),   # y est vide
        ([], [], []),       # x et y sont vides
        (['x'], [3, 4], [('x', 3), ('x', 4)]),
    ])
    def test_valid_inputs(self, x, y, expected):
        result = produit(x, y)
        # Vérification du contenu ET des types
        assert result == expected
        assert check_type(result , list[tuple]) == result
        """
        assert isinstance(result, list)
        assert all(isinstance(pair, tuple) for pair in result)
        """
        #Chaque élément est un tuple

    # cas d'erreurs
    @pytest.mark.parametrize("x, y", [
        (5, [1, 2]),         # x n'est pas une liste
        ([1, 2], 3.14),      # y n'est pas une liste
        (None, ["a"]),       # x est None
        ({"a":1 } , [1,2]),   # x est un dictionnaire (non liste)
        (["a",],"error"),      # y est une chaîne de caractère

    ])
    def test_type_invalid_inputs1(self, x, y):
        with pytest.raises(TypeError):
            produit(x, y)


#--------------------------------------------------

class TestAdd:
    """Tests pour la fonction add()."""

    # cas normaux et limites
    @pytest.mark.parametrize("old, a, expected", [
        ([[1]], 2, [[1], [1, 2]]),
        ([[]], 1, [[], [1]]),
        ([[], [2]], 'a', [[], [2], ['a'], [2, 'a']]), # 'a' n'est pas du même type que les éléments des sous listes de old
    ])
    def test_valid_inputs(self, old, a, expected):
        result = add(old, a)
        # Vérification du contenu ET des types
        assert result == expected
        assert check_type(result , list[list]) == result
        """
        assert isinstance(result, list)
        assert all(isinstance(sublist, list) for sublist in result)  """# Tous des listes

    # cas d'erreurs
    @pytest.mark.parametrize("old, a", [
        (5, 2),      # old n'est pas une liste
        ([1, 2], 3), # old contient des non-listes → erreur lors de e + [a]
       #("invalid" , 4 ) # old est une chaîne (Interprétée comme liste de caractères)
        ([[1], "not_a_list"], 2),  # Sous-liste invalide
    ])
    def test_invalid_structures(self, old, a):
        with pytest.raises(TypeError):
            add(old, a)



#------------------------------------------------------------------


class TestPowerSet:
    """Tests pour la fonction power_set()."""

    @pytest.mark.parametrize("x, expected", [
        ([], [[]]),
        ([1, 2], [[], [1], [2], [1, 2]]),
        ([1, 2, 3], [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]),
    ])
    def test_power_set_valid_inputs(self, x, expected):
        result = power_set(x)
        # Vérification du contenu (ordre ignoré) ET des types
        assert sorted(map(sorted, result)) == sorted(map(sorted, expected))
        assert check_type(result,list[list]) == result
        #assert isinstance(result, list)
        #assert all(isinstance(subset, list) for subset in result)  # Tous des listes

    @pytest.mark.parametrize("x", [
        5,                     # x est un entier
        None,                  # x est None
        {"a": 1 , "b": 2},     # x est un dictionnaire
        "invalid_iterable",    # x est une chaîne (itérable mais non traité comme liste)
    ])
    def test_power_set_invalid_inputs(self, x):
        with pytest.raises(TypeError):
            power_set(x)


if __name__=='__main__':

    pytest.main([__file__])