from functools import wraps
from typeguard import check_type

def type_check3(func):
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






