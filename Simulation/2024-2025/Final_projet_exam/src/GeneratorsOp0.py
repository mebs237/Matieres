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

class EGenerator(Generators):
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

        return "EGenerator"

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

class ImprovedExpGenerator(Generators):

    def __init__(self, decimals: list[int]) -> None:
        self.decimals = decimals
        self.len_decimals = len(decimals)
        self.index = np.random.randint(0, self.len_decimals - 1)

    def next(self) -> float:
        # Utiliser 10 décimales consécutives
        num = 0.0
        power = 0.1
        for i in range(10):
            self.index = (self.index + 1) % self.len_decimals
            num += self.decimals[self.index] * power
            power /= 10
        return num

class Im2Generator(Generators):
    # Paramètres pour deux LCG différents
    a1, c1, m1 = np.int64(1664525), np.int64(1013904223), np.int64(2**32)
    a2, c2, m2 = np.int64(16807), np.int64(0), np.int64(2**64 - 1)

    def __init__(self, decimals: np.ndarray, lg: int = 7):
        self.decimals = decimals
        self.lg = lg
        self.len_decimals = len(decimals)
        self.index = np.random.randint(0, self.len_decimals - lg)
        self.power_of_10 = np.array([10**i for i in range(lg)])
        # États initiaux des deux LCG
        self.x1 = self._compute_x0()
        self.x2 = self._compute_x0()

    def _compute_x0(self):
        indices = np.arange(self.index, self.index + self.lg) % self.len_decimals
        bloc = self.decimals[indices]
        return np.dot(bloc, self.power_of_10[::-1])

    def next(self) -> float:
        # Génération avec deux LCG
        self.x1 = (self.a1 * self.x1 + self.c1) % self.m1
        self.x2 = (self.a2 * self.x2 + self.c2) % self.m2
        combined = (self.x1 ^ self.x2) & 0xFFFFFFFF
        # Post-traitement XORShift
        combined ^= (combined << 21) & 0xFFFFFFFF
        combined ^= (combined >> 35)
        combined ^= (combined << 4)
        # Mise à jour aléatoire de l'index tous les 1000 appels
        if np.random.randint(0, 1000) == 0:
            self.index = np.random.randint(0, self.len_decimals - self.lg)
        return combined / 0xFFFFFFFF

class SimplePiGenerator:
    """
    Générateur de nombres aléatoires [0, 1[ utilisant les décimales de π.
    - Simple à comprendre : utilise directement les décimales de π
    - Efficace : pas de calculs complexes
    - Aléatoire : démarre à un index aléatoire
    """

    def __init__(self, decimals: np.ndarray, chunk_size: int = 10):
        self.decimals = decimals          # Ex: [3,1,4,1,5,9,...]
        self.chunk_size = chunk_size      # Nombre de décimales utilisées par nombre
        self.index = np.random.randint(0, len(decimals) - chunk_size)  # Départ aléatoire

    def next(self) -> float:
        # 1. Prendre un bloc de décimales
        end = self.index + self.chunk_size
        chunk = self.decimals[self.index : end]

        # 2. Convertir en nombre entre 0 et 1 (ex: [1,4,1,5] → 0.1415)
        number = 0.0
        for i, digit in enumerate(chunk, 1):
            number += digit * (10 ** -i)  # 1 → 0.1, 4 → 0.04, etc.

        # 3. Avancer l'index (reboucler si nécessaire)
        self.index = (end) % (len(self.decimals) - self.chunk_size)

        return number

generator = Im2Generator(decimals=np.array([3,1,4,1,5,9,2,6]))
print(generator.generate_n(5))  # Output: 0.12345678...
