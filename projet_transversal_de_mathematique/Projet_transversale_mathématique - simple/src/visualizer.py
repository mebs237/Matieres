"""
    modules pour la  Visualisation  d'un système Rangeland
"""

import matplotlib.pyplot as plt
import numpy as np
from .rangeland import RangelandSystem

class RangelandVisualizer:
    """classe pour l'affichage de différents graphes d'un système Rangeland"""

    def __init__(self, system: RangelandSystem):
        self.system = system

    def plot_trajectory(self, initial_conditions: list, E: float):
        """Trace les trajectoires"""
        # Votre code de trajectory_plot...



    def plot_phase_portrait(self, *args):
        """Trace le portrait de phase complet"""

        # Fusion de vos fonctions plot_separatrix et plot_equilibria
    def plot_equilibria(self,*args):
        """
        trace les point d'équilibre de la courbe
        """
    def plot_separatrix(self , args):
        """
        Trace la courbe séparatrice (si elle existe) et les équilibres vers lesquels
        toutes les trajectoires sont censées converger.

        Args:
            E: Paramètre d'exploitation ou environnemental
        """
        # Récupérer les points de la courbe séparatrice
        w , g = self.system.separatrix_points(args)

        # Récupérer les équilibres et leur classification
        equilibria  = self.system.get_classified_equilibria(E)

        # Créer la figure
        plt.figure(figsize=(10, 8))

        # Tracer la courbe séparatrice si elle existe
        if len(w) > 0:
            plt.plot( g,w, '--', color='blue', label='Courbe séparatrice')

        # Tracer les équilibres avec des couleurs différentes selon leur stabilité
        for eq  in equilibria:
            typ , val = eq.type , eq.val
            if typ == 'stable':
                plt.plot(val[0], val[1], 'go', markersize=16, label='Équilibre stable' , color='blue')
            elif typ == 'instable':
                plt.plot(val[0], val[1], 'ro', markersize=16, label='instable',color='red')
            else:  # selle
                plt.plot(val[0], val[1], 'bo', markersize=16, label='selle',color='yellow')

        # Ajouter une grille et des légendes
        plt.grid(True)
        plt.xlabel('herbe grass g')
        plt.ylabel('herbe weed  w')
        plt.title(f'Courbe séparatrice et équilibres pour E = {E}')

        # Éviter les doublons dans la légende
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # Définir les limites du graphique
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Afficher le graphique
        plt.show()

