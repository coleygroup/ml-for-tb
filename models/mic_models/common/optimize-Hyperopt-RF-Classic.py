# 1. Imports
#--------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy import genfromtxt

import joblib
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

import graphviz as graphviz



# 2. Global variables
#--------------------------------------------------------------------------------------------
#models:  crf, wrf, rfbs, rfu, rfee
data_path = 'ignore/data/'
model_path = 'ignore/model/'
RANDOM_STATE = 42

taacf = True
if taacf:
    suffix = '_TS' 
else:
    suffix = '_M' 



# 3. LOAD DATA
#--------------------------------------------------------------------------------------------
# use the same training set used in mlsmr baseline to compare ressults
x_train = pd.read_csv(f'{data_path}x_train{suffix}.csv', index_col=0) 
y_train = genfromtxt(f'{data_path}y_train{suffix}.csv', delimiter=' ')

#remove inhibition and SMILES columns
x_train = x_train.drop(['SMILES', 'Inhibition'], axis=1)



# 4.  Set hyperparameters to search through
#--------------------------------------------------------------------------------------------
space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 25, 750, 10)),                 # num of trees:
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 10, 500, 10)),         # min num of samples required at each leaf node
    'max_depth': scope.int(hp.quniform('max_depth', 5, 750, 10)),                        # max num of levels in tree
    'min_samples_split': scope.int(hp.quniform ('min_samples_split', 10, 1000, 10)),     # min mum of samples required to split a node
    'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),            # maximum number of features Random Forest is allowed to try in individual tree
    'bootstrap': hp.choice('bootstrap', [True, False]),                                  # Method of selecting samples for training each tree
    'criterion': hp.choice('criterion', ['entropy', 'gini']),
    'random_state': RANDOM_STATE,
    'n_jobs': -1
    }



# 5. define objective function
#--------------------------------------------------------------------------------------------
def objective(space, x_train, y_train, cv, scoring_type):
    clf = RandomForestClassifier(**space)
    score = cross_val_score(clf, x_train, y_train, cv = cv, scoring = scoring_type).mean()

    # We aim to maximize scoring metric passed, therefore we return it as a negative value
    return {'loss': -score, 'loss_variance': np.var(-score, ddof=1), 'status': STATUS_OK }



# 6.  Hyperopt - Random Forest - Manual Class weight
#--------------------------------------------------------------------------------------------
#scoring = ['balanced_accuracy', 'f1']
scoring = 'f1'
splits = 5
repeats = 10
filename = f'classic_hyperopt_{scoring}'

#set up trials object to store optimization details
trials = Trials()

# used the formula in this article to estimate max_evals
# https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html
# 4 hp.quniform:  10(4) = 40  or  20(4) = 80 
# 3 hp.choice:  15(4 * 2 * 2) = 240
# suggested range:  280 - 320
max_evals = 300


# n_splits:  # of folds
# n_repeats:  # of times cross-validator is repeated
rstratified_kfold = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=RANDOM_STATE)

# set fmin to call custom objective function with additional params
fmin_objective = partial(objective, x_train=x_train, y_train=y_train, cv=rstratified_kfold, scoring_type=scoring)
best = fmin(fn = fmin_objective,
            space =space,
            algo = tpe.suggest,
            max_evals = max_evals,
            trials = trials)


# store trials info
path_trials = f'{model_path}{filename}{suffix}.trl'
joblib.dump(trials, path_trials)

# store best parameters
with open(f'{model_path}{filename}{suffix}.best','w') as data: 
    data.write(str(best))