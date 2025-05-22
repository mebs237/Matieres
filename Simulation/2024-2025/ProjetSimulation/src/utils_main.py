"""
    Point d'entrée principal pour exécuter les tests statistiques sur les générateurs de nombres pseudo-aléatoires.
"""
# IMPORTATION DES MODULES
from numpy.typing import NDArray
from numpy import array




def read_decimal_file(file_path: str) -> NDArray:
    """
    Lit un fichier contenant les décimales et extrait tous les chiffres.

    Args:
        file_path (str): Chemin vers le fichier à lire.

    Returns:
        NDArray: Tableau contenant tous les chiffres.
    """
    try:
        # Lecture et nettoyage du fichier
        with open(file_path, 'r', encoding='utf-8') as file:
            content = ''.join(file.readlines())
            # Supprimer les espaces, retours à la ligne et commentaires
            content = ''.join(c for c in content if c.isdigit() or c == '.')

        # Trouver le point décimal et extraire tous les chiffres après
        if '.' in content:
            _, digits = content.split('.', 1)
            return array([int(d) for d in digits], dtype=int)
        else:
            return array([], dtype=int)

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
        return array([])
    except ValueError as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return array([])


def group_digits(array_1: NDArray, k: int) -> NDArray:
    """
    Groupe les chiffres d'un tableau en nombres de k chiffres consécutifs.

    Parameters
    ----------
    array_1 : NDArray
        Tableau de chiffres à grouper
    k : int
        Nombre de chiffres consécutifs par groupe

    Returns
    -------
    NDArray
        Tableau contenant les nombres formés par k chiffres consécutifs

    Examples
    --------
    >>> group_digits(np.array([1,2,3,4,5,6,7]), 3)
    array([123, 456, 7])
    >>> group_digits(np.array([1,2,3,4,5]), 2)
    array([12, 34, 5])
    """
    if k <= 0:
        raise ValueError("k doit être strictement positif")

    n = len(array_1)
    # Initialiser le tableau résultat
    result = []

    # Traiter les groupes complets de k chiffres
    i = 0
    while i <= n - k:
        # Convertir k chiffres en un seul nombre
        number = sum(array_1[i + j] * (10 ** (k - j - 1)) for j in range(k))
        result.append(number)
        i += k

    # Traiter les chiffres restants s'il y en a
    if i < n:
        remaining_digits = n - i
        number = sum(array_1[i + j] * (10 ** (remaining_digits - j - 1))
                    for j in range(remaining_digits))
        result.append(number)

    return array(result)



def plot_p_values_evolution(self):
    """
    Affiche l'évolution de la moyenne des p-values par itération.
    """
    df = self.hist_df
    grouped = df.groupby("Iteration")["p_value"].mean().reset_index()

    fig = px.line(grouped, x="Iteration", y="p_value",
                  title=f"Évolution des p-values moyennes par itération ({self.name})",
                  markers=True)
    fig.update_yaxes(title_text="p_value moyenne")
    fig.show()


def plot_p_values_evolution(self):
    """
    Affiche l'évolution de la moyenne des p-values en fonction de la position,
    pour chaque granularité (taille de fenêtre).
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=len(self.window_sizes), cols=1,
                        shared_xaxes=True,
                        subplot_titles=[f"Granularité {g}" for g in self.window_sizes])

    for idx, g in enumerate(sorted(self.window_sizes)):
        sub_df = self.hist_df[self.hist_df["window_size"] == g]
        grouped = sub_df.groupby("window_start")["p_value"].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=grouped["window_start"], y=grouped["p_value"],
                       mode="lines+markers", name=f"g={g}"),
            row=idx + 1, col=1
        )

    fig.update_layout(height=300 * len(self.window_sizes),
                      title_text=f"Évolution des p-values moyennes pour {self.name}",
                      showlegend=False)
    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="p_value moyenne")
    fig.show()
