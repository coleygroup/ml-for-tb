# 1. Imports
#--------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy import genfromtxt
import joblib


from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier



# 2. Global variables
#--------------------------------------------------------------------------------------------
#models:  crf, wrf, rfbs, rfu, rfee
data_path = 'ignore/data/'
model_path = 'ignore/model/'
RANDOM_STATE = 42
filename = f'rfu_'

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
# https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

n_estimators = np.linspace(175, 500, 14, dtype=int)         # num of trees                                                       
min_samples_leaf = np.linspace(50, 300, 11, dtype=int)      # min num of samples required at each leaf node
min_samples_split = np.linspace(2, 30, 28, dtype=int)       # min mum of samples required to split a node
max_depth = np.linspace(1, 500, 20, dtype=int)              # max num of levels in tree
max_features = [None, 'sqrt', 'log2']                       # maximum number of features Random Forest is allowed to try in individual tree
bootstrap = [True, False]                                   # Method of selecting samples for training each tree
sampling_strategy= np.linspace(0.025, 1, 25, dtype=float)    # ratio of the # of minority class to majority class after resampling
replacement = [True, False]                                 # wheter to sample randomly w/replacement
class_weight= [{0:1, 1: w} for w in list(range(15, 50))]  # class weights for inactive/active, increment = 1
criterion=['gini', 'entropy']                               # Criterion


#set grid space
grid_searchcv = {'n_estimators': n_estimators,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features,
    'bootstrap': bootstrap,
    'sampling_strategy': sampling_strategy,
    'replacement': replacement,
    'class_weight': class_weight,
    'criterion': criterion}



# 5.  RandomizeSearchCV - Random Forest - baseline
#--------------------------------------------------------------------------------------------
# instantiate a new random forest classifier 
rfcRS = BalancedRandomForestClassifier()
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
rfClassifier = BalancedRandomForestClassifier(**best_params)
rfClassifier.fit(x_train, y_train)

#store model
file_bestmodel = f'{model_path}{filename}{suffix}.mdl'
joblib.dump(rfClassifier, file_bestmodel)
joblib.dump(rfClassifier, f'{filename}{suffix}.mdl')