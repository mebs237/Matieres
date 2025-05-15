"""
    modules pour la  Visualisation  de la dynamique d'un système Rangeland
"""
from dataclasses import dataclass , field
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from rangeland import RangelandSystem, EquilibriumType, DefaultSystem
from tools import plot_vector_field_near_equilibria

@dataclass
class RangelandVisualizer:
    """classe pour l'affichage de différents graphes d'un système Rangeland"""

    system:RangelandSystem = field(default_factory=DefaultSystem)

    def plot_trajectory(self, initial_conditions: list, *args):
        """Trace les trajectoires"""
        # Votre code de trajectory_plot...



    def plot_phase_portrait(self, *args):
        """Trace le portrait de phase complet"""

        # Fusion de vos fonctions plot_separatrix et plot_equilibria
    def plot_equilibria(self,*args):
        """
        trace les point d'équilibre de la courbe

        """


    def plot_vector_field(self, grid_size:int=20):
        """
        Trace un champ de vecteurs F(u)

        Arguments:
            F: fonction de champ de vecteurs, prend un tableau de coordonnées et retourne les vecteurs
            grid_size: nombre de points dans la grille (par défaut 20)
        """
        def F(u):
            return self.system.equation(0,u)
        x_min, x_max = (0,1)
        y_min, y_max = (0,1)
        X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        U, V = F([X, Y])

        plt.figure(figsize=(8, 6))
        plt.quiver(X, Y, U, V, color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Champ de vecteurs F(u)')
        plt.show()

    def plot_separatrix(self , *args):
        """
        Trace la séparatrice (si elle existe) et les équilibres vers lesquels toutes les trajectoires sont censées converger.
        """
        # Récupérer les points de la courbe séparatrice
        w , g = self.system.separatrix_points(*args)

        # Récupérer les équilibres et leur classification
        equilibria  = self.system.get_classified_equilibria(*args)

        # Créer la figure
        plt.figure(figsize=(10, 8))

        # Tracer la courbe séparatrice si elle existe
        if len(w) > 0:
            plt.plot( g,w, '--', color='blue', label='Courbe séparatrice')

        # Tracer les équilibres avec des couleurs différentes selon leur stabilité
        for eq  in equilibria:
            nature , coord = eq.nature , eq.coord
            if nature == EquilibriumType.STABLE:
                plt.plot(coord[0], coord[1], 'go', markersize=16, label='Équilibre stable' , color='blue')
            elif nature == 'instable':
                plt.plot(coord[0], coord[1], 'ro', markersize=16, label='instable',color='red')
            else:  # selle
                plt.plot(coord[0], coord[1], 'bo', markersize=16, label='selle',color='yellow')

        # Ajouter une grille et des légendes
        plt.grid(True)
        plt.xlabel('herbe grass g')
        plt.ylabel('herbe weed  w')
        plt.title(f'Courbe séparatrice et équilibres pour  = {args}')

        # Éviter les doublons dans la légende
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # Définir les limites du graphique
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Afficher le graphique
        plt.show()

