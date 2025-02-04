# 1. Imports
#--------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy import genfromtxt
import joblib


from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from imblearn.ensemble import EasyEnsembleClassifier



# 2. Global variables
#--------------------------------------------------------------------------------------------
#models:  crf, wrf, rfbs, rfu, rfee
data_path = 'ignore/data/'
model_path = 'ignore/model/'
RANDOM_STATE = 42
filename = f'rfee_'

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



# 4.  SET HYPERPARAMTERS TO SEARCH THROUGH
#--------------------------------------------------------------------------------------------
# np.linspace = returns evenly spaced numbers over a specified interval:  interval = (x2-x1)/(n-1))
# https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html

n_estimators = np.linspace(175, 500, 14, dtype=int)                             # # of AdaBoost learners in the ensemble.                                                       
sampling_strategy = ['majority', 'not minority', 'not majority', 'all', 'auto'] # Sampling information to sample the data set.
replacement = [True, False]                                                     # Whether or not to sample randomly with replacement or not

#set grid space
grid_searchcv = {'n_estimators': n_estimators,
               'sampling_strategy': sampling_strategy,
               'replacement': replacement}



# 6.  RandomizeSearchCV - Random Forest 
#--------------------------------------------------------------------------------------------
# instantiate a new random forest classifier 
rfcRS = EasyEnsembleClassifier()
scoring = 'balanced_accuracy'
#scoring = 'f1'

cv = 10
iter = 20
splits = 5
repeats = 10
filename = f'{filename}{scoring}'

# n_splits:  # of folds
# n_repeats:  # of times cross-validator is repeated
rstratified_kfold = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=RANDOM_STATE)

# Instantiate random grid search model, for best params to look through
rscv = RandomizedSearchCV(estimator=rfcRS,
                           param_distributions=grid_searchcv,
                           n_iter=iter,
                           scoring=scoring,
                           cv=rstratified_kfold,  
                           n_jobs=-1, verbose=3,
                           random_state=RANDOM_STATE)

# perform hyperparameter optimization
rscv.fit(x_train, y_train)

# save randomized search cv
file_rscv = f'{model_path}{filename}{suffix}.rscv'
joblib.dump(rscv, file_rscv)
joblib.dump(rscv, f'{filename}{suffix}.rscv')


# get best params
best_params = rscv.best_estimator_.get_params()

#create new model and train w/best parameters
rfClassifier = EasyEnsembleClassifier(**best_params)
rfClassifier.fit(x_train, y_train)

#store model
file_bestmodel = f'{model_path}{filename}{suffix}.mdl'
joblib.dump(rfClassifier, file_bestmodel)
joblib.dump(rfClassifier, f'{filename}{suffix}.mdl')