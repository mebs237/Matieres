from pandas import set_option
from seaborn import set_palette
from matplotlib.pyplot import style , rcParams

import matplotlib.ticker as mtick
from tqdm import tqdm
from sklearn.compose import ColumnTransformer , make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import defaultdict
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV


# @title sélection du meilleur modèle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from typing import Dict, Tuple , List
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV


# @title fonction de score et evaluation
from scipy import stats
from scipy.stats import uniform, randint,  entropy
from sklearn.metrics import root_mean_squared_error, r2_score ,mean_absolute_error , mean_squared_error , mutual_info_score


# @title initialisation des modèles
from sklearn.linear_model import ElasticNet , Lasso
from xgboost import XGBRegressor


import os
import sys
import joblib
import json
import warnings
import glob
import re
from IPython.display import Markdown, display
from os.path import join


#Pour ignorer les warnings
warnings.filterwarnings('ignore')

# configuration des graphiques

style.use('seaborn-v0_8-whitegrid')
rcParams['figure.figsize'] = (12, 8)
set_palette('Set2')


# Pour une meilleure lisibilité dans le notebook


set_option('display.max_columns', None)
set_option('display.max_rows', None)
set_option('display.float_format', '{:.3f}'.format)
