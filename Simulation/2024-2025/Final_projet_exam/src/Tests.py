""" Module des Tests """

from abc import ABC , abstractmethod
from math import factorial
from functools import cache
from typing import Literal , List,Dict,Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.stats import chi2

# Les fonctions comunes à plusieurs tests
@cache
def stirling(n:int,k:int)->int:
    """
    Méthode pour calculer le nombre de stirling de deuxième espèce.
    """
    assert n >= 0 and k >= 0 , "n et k doivent être plus grand que 0."

    if n == k:
        return 1

    elif n == 0 or k == 0:
        return 0

    return k * stirling(n - 1, k) + stirling(n - 1, k - 1)
@cache
def input_type(data:NDArray)->Literal["integers","decimals"]:
    """
        Méthode pour déterminer le type de données
    """
    return "integers" if isinstance(data[0],int) else "decimals"

class Test(ABC):
    """
    Classe abstraite définissant un test d'hypothèse.
    """
    @abstractmethod
    def __str__(self)->str:
        """
        Méthode  permettant  de récupérer le nom du test.
        """

    @abstractmethod
    def test(self, data:NDArray,alpha:float=0.05,info:bool=False) -> float|Dict:
        """
            teste  la séquence de nombres data

            Args:
                data : suite de nombre à tester
                alpha : seuil de signification (pourcentage d'erreur permit)
                info : si True retourne des infos supplementaires sur le test effectué

            Returns:
                _ : si info retourne un Dictionnaire contenant la stat seuil et observée Kᵣ , Kₐ , p-valeur ,
        """


class Chi2Test(Test):
    """
        classe qui impléménte le test de χ²

        Args:
            k:nombre de bins pour la catégorisation des nombres de data
    """
    def __init__(self, k:int=10):
        self.k = k

    def __str__(self):
        return "χ² Test "

    @staticmethod
    def group_low_frequencies(observed:NDArray, expected:NDArray, threshold=5)->Tuple[NDArray,NDArray]:
        """
            Regroupe les catégories adjacentes avec des fréquences attendues inférieures au seuil = threshold jusqu'a ce que 80% des fréquences attendues soient supérieures au seuil
        """
        pass
        return observed, expected

    #group_low_frequencies n'est impléménté que dans cette classe car n'est utlisé que dans le cadre d'un test de chi2

    @staticmethod
    def chi2_statistic(observed:NDArray, expected:NDArray)->float:
        """
            Calcule la statistique du chi carré
        """

        observed, expected = Chi2Test.group_low_frequencies(observed, expected)
        return np.sum((observed - expected) ** 2 / expected) , observed.size - 1

    def test(self, data:NDArray,alpha:float=0.05,info:bool=False) -> float|Dict:

        N = len(data)
        if input_type(data) == "integers":
            observed = np.bincount(data,minlength=np.max(data)+1)

        #les fréquences observées pour k bins
        observed , _ = np.histogram(data,bins=self.k)

        # les fréquences attendues pour k bins
        expected = np.full(self.k , N/self.k)


        #statistique du chi carré et degré de liberté
        chi2_stat , df = Chi2Test.chi2_statistic(observed, expected)

        #p-valeur
        p_value = float(chi2.cdf(chi2_stat, df))
        # Valeur/statistique critique
        Ka = float(chi2.ppf(1-alpha,df))

        return {
            "K": chi2_stat,
            "Ka": Ka,
            "p_value": p_value,
            "df": df
        }if info else p_value


class GapTest(Test):
    """
        classe qui représente le test de Gap

        Args:
            a : borne inférieur de l'intervalle pour le marquage
            b : borne supérieure de l'intervalle pour le marquage
            g : fonction de calcul du paramètre p de la loi gémetrique
    """
    def __init__(self, a=0.1,b=0.3):
        self.a = a
        self.b = b
        self.p = b-a
    def __str__(self):
        return "Gap Test "


    def test(self, data:NDArray,alpha:float=0.05,info:bool=False) -> float|Dict:

        #selection des indices des nombres à marquer
        mark = (data >= self.a) & (data <= self.b)
        index = np.where(mark)[0]

        # Calculer les gaps entre les indices
        gaps = np.diff(index) - 1
        if len(gaps) > 0:
            max_gap = np.max(gaps)
            observed =  np.bincount(gaps , minlength=max_gap + 1)
        else:
            raise ValueError(f"il n' y a pas de gaps  sur  rapport à l'intervalle [{self.a} , {self.b}[ ")



        max_gap = np.max(gaps)
        expected = (gaps.size * self.p) * ( (1 - self.p) ** np.arange(max_gap + 1))

        # statistique du Chi carré , et déré de liberté
        chi2_stat  , df = Chi2Test.chi2_statistic(observed, expected)

        #p-valeur
        p_value = float(chi2.cdf(chi2_stat, df))

        # Valeur/statistique critique
        Ka = float(chi2.ppf(1-alpha,df))

        return {
            "K": chi2_stat,
            "Ka": Ka,
            "p_value": p_value,
            "df": df
        }if info else p_value
