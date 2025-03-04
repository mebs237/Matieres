""" Module des Tests """

from abc import ABC , abstractmethod
import numpy as np
from scipy.stats import chi2


class Test(ABC):
    """
    Classe abstraite représentant un test d'hypothèse.
    """
    @abstractmethod
    def test(self, data:np.ndarray) -> dict:
        """
            teste  la séquence de nombres data

            Args:
                data : suite de nombre à tester
                alpha : seuil

            Returns:
                dic : un dictionnaire contenant la statistique χ² , p-valeur
        """


class Chi2Test(Test):
    """classe qui represente le test de χ²

        Args:
            k:nombre de bins pour la catégorisation des nombres de data
    """
    def __init__(self, k:int=10):
        self.k = k

    def __str__(self):
        return "χ² Test "

    @staticmethod
    def group_low_frequencies(observed:np.ndarray, expected:np.ndarray, threshold=5):
        """
            Regroupe les catégories avec des fréquences attendues inférieures au seuil = threshold
        """

        # Indices des catégories à regrouper
        low_freq_indices = np.where(expected < threshold)

        # Regroupement des catégories en une seule
        if low_freq_indices.size > 0:
            observed[low_freq_indices[0]] = observed[low_freq_indices].sum()
            expected[low_freq_indices[0]] = expected[low_freq_indices].sum()
            observed = np.delete(observed, low_freq_indices[1:])
            expected = np.delete(expected, low_freq_indices[1:])

        return observed, expected

    @staticmethod
    def chi2_statistic(observed:np.ndarray, expected:np.ndarray,threshold = 2 )->float:
        """
            calcule la statistique chi2 en la corrigeant dans le cas de faibles fréquences théoriques
        """
        observed , expected = Chi2Test.group_low_frequencies(observed,expected, threshold)
        chi2_stat = np.sum((observed - expected) ** 2 / expected)


        return float(chi2_stat)

    def test(self,data:np.ndarray)->dict:

        N = len(data)
        #les fréquences observées pour k bins
        observed , _ = np.histogram(data,bins=self.k)

        # les fréquences attendues pour k bins
        expected = np.full(self.k , N/self.k)

        #statistique du chi carré
        chi2_stat= self.chi2_statistic(observed,expected)

        # calcul de la p-valeur et du dégré de liberté

        df = observed.size - 1

        p_value = float(chi2.sf(chi2_stat, df))

        return {
            "chi2_stat": chi2_stat,
            "p_value": p_value
        }


class GapTest(Test):
    """
        classe qui représente le test de Gap

        Args:
            a : borne inférieur de l'intervalle pour le marquage
            b : borne supérieure de l'intervalle pour le marquage
            g : fonction de calcul du paramètre p de la loi gémetrique
    """
    def __init__(self, a=0.1,b=0.2):
        self.a = a
        self.b = b
        self.norm = False
    def __str__(self):
        return "Gap Test "


    def test(self, data : np.ndarray,alpha=0.05)->dict:

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

        if isinstance(data[0],int): # pour savoir si c'est une séquence de nombre entier >=1 ou décimaux
            p = (self.b-self.a+1)/np.max(data)
        else :
            p = self.b-self.a

        max_gap = np.max(gaps)
        expected = (gaps.size * p) * ( (1 - p) ** np.arange(max_gap + 1))

        # statistique du Chi carré
        chi2_stat = Chi2Test.chi2_statistic(observed, expected)

        # p-valeur et degré de liberté

        df = len(observed) - 1

        p_value = chi2.sf(chi2_stat, df)


        return {
            "chi2_stat": chi2_stat,
            "p_value": p_value,
        }
