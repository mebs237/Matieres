from functools import wraps
from typeguard import check_type

def type_check3(func:callable)->callable:
    """
    Décorateur permettant de vérifier les types des arguments passés à une fonction
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Vérification des annotations de type
        annotations = func.__annotations__
        for arg_name, arg_value in zip(func.__code__.co_varnames, args):
            if arg_name in annotations:
                expected_type = annotations[arg_name]
                try :
                    check_type(arg_value , expected_type)
                except  :
                    raise TypeError(f" '{arg_name}' must be type {str(expected_type)}")

        return func(*args, **kwargs)
    return wrapper


@type_check3
def produit[X,Y](x:list[X],y:list[Y])->list[tuple[X,Y]]:
    """
    fait le produit catesien de deux ensembles
    """
    return [(xi,yi) for xi in x for yi in y]


@type_check3
def add[T](old:list[list[T]],a:T)->list[list[T]]:
    """
    Génère un ensemble de sous-ensembles en ajoutant l'élément a à chaque sous-ensemble dans old puis étendant old avec ces nouveaux sous-ensembles

    Parameters
    ---
    old (list[list[T]]) : ensemble des parties  d'un certain ensemble x
    a (T) :  élément à ajouter

    Returns
    ---
    _  (list[list[T]]) : ensemble des parties de l'ensemble x avec l'élément a ajouté
    """
    new_subsets : list[list[T]] = [ e + [a] for e in old ]

    return sorted(old + new_subsets , key=len)


@type_check3
def power_set[X]( x:list[X] )->list[list[X]]:
    """
    Génère l'ensemble des parties 2ˣ  de l'ensemble x

    Parameters
    ---
        x : liste représentant l'ensemble initial

    Returns
    ---
        _ : liste dont les éléments sont des listes qui représentent les sous ensembles de x

    """
    if len(x)==0 : # l'ensemble x est vide
        return [[]]
    res : list[list[X]] = [[]]
    for e in x :
        res = add(res,e)

    return res

# version récursive de power_set

@type_check3
def power_set_rec1[X](x:list[X])->list[list[X]]:
    """
    Génère l'ensemble des parties 2ˣ  de l'ensemble x
    """
    if len(x)==0 : # l'ensemble x est vide
        return [[]]
    return add(power_set_rec1(x[:-1]),x[-1])

def power_set_rec2[X](x:list[X])->list[list[X]]:
    """
    Génère l'ensemble des parties 2ˣ  de l'ensemble x
    """
    if len(x)==0:
        return [[]]
    #partition sans le premier élément
    partial_parts = power_set_rec2(x[1:])
    # inclure le 1er élément dans chaque partition
    full_parts = [[x[0]] + part for part in partial_parts]
    return full_parts
