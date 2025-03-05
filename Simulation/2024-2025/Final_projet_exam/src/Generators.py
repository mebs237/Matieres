#Generators
"""
Module des Générateurs"""
#     Args:


from abc import ABC, abstractmethod
import numpy as np




class Generators(ABC):
    """
    Classe abstraite définissant ce qu'est un générateur.
    """

    @abstractmethod
    def next(self) -> float:
        """
        Génére le prochain nombre pseudo-aléatoire.
        """

    def generate_n(self,n:int)->np.ndarray:
        """
        Génère n nombres pseudo aléatoires

        Args
        ---
            n (int) : quantité (nombre) de nombres à générer
        """
        return np.array([self.next() for _ in range(n)])

class PythonGenerator(Generators):
    """
    Classe permettant d'utiliser le générateur par défaut de Python en tant que Generators.
    """
    def __str__(self) -> str:
        return "PythonGenerators"

    def next(self) -> float:
        return np.random.random()


class ExpGenerator(Generators):
    """
    Classe qui implemente notre générateur pseudo-aléatoire d'une loi uniforme [0; 1[ utilisant une sequence 'decimals' de chiffres

    """
    # (a,c,m) paramètres du (Générateur de congruenciel linéaire) de Numerical Recipes

    a = np.int64(1664525)
    c = np.int64(1013904223)
    m = np.int64(2**32)

    """ un autre choix de paramètres
    m = np.int64(2**64 -1 )  plus grand entier sur 64 bits
    a = np.int64(16807)
    c = np.int64(0)

    un autre choix , celui du merseene twister
    m = np.int64(2**19937 -1)
    """

    def __str__(self) -> str:

        return "PiGenerator"

    def __init__(self, decimals : np.ndarray,lg : int =10)->None:
        """
        Args
        ---
            decimals (ndarray) : Séquence de chiffres
            lg (int) : taille du blocs de décimal à sélectionné vaut  7 par défaut
        """

        self.decimals = decimals
        self.lendecimals = len(decimals)
        # initialisation aléatoire de l'index
        self.index  = np.random.randint(0, self.lendecimals - lg)
        self.power_of_10 = np.array([10**i for i in range(lg)])
        self.lg = lg

    def next(self) -> float:

        # selection des le blocs de décimales  pour initialiser x0
        # indices des décimales à selectionner
        indices = np.arange(self.index,self.index + self.lg)%self.lendecimals
        selected =  self.decimals[indices]
        # x0 = selected
        x0 = np.dot(selected , self.power_of_10[::-1])

        # calcul du nombre pseudo aléatoire
        x = (self.a*x0 + self.c )%(self.m)
        u = float(x)/(self.m)

        # mise à jour de l'index pour le prochain appel
        self.index = (self.index + np.random.randint(1,self.lendecimals-self.lg))%self.lendecimals

        return u

