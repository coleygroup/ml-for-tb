import pandas as pd
import numpy as np
from numpy import genfromtxt

import joblib
import os
import pathlib

#common training code
#to import common file
import traintestsplit_common as splits



# 1.  Global variables and Constants
#---------------------------------------------------------------------------------------------
#models taacf:  wrf, crf
#models mlsmr:  wrf, rfu
RANDOMSTATE = 42
model = 'wrf'
metric = 'f1'

isTAACF = True

if isTAACF:
    dataset = 'TAACF-SRIKinase'
    suffix = '_TS'
else:
    dataset = 'MLSMR'
    suffix = '_M'

data_path = 'ignore/data/'
model_path = f'ignore/model/'

split_size_start = 0.40
split_size_increment = 0.05
cv = 10



# 2.  load data
#---------------------------------------------------------------------------------------------
#continue to use the same split for baseline performance analysis, training, and future hyperparameter optimization
X_train = pd.read_csv(f'{data_path}X_train{suffix}.csv', index_col=0) 
y_train = genfromtxt(f'{data_path}y_train{suffix}.csv', delimiter=' ')

if isTAACF:
    x_cv = pd.read_csv(f'{data_path}x_cv{suffix}.csv', index_col=0) 
    y_cv = genfromtxt(f'{data_path}y_cv{suffix}.csv', delimiter=' ')

x_test = pd.read_csv(f'{data_path}x_test{suffix}.csv', index_col=0) 
y_test = genfromtxt(f'{data_path}y_test{suffix}.csv', delimiter=' ')


#remove inhibition and SMILES columns
X_train = X_train.drop(['SMILES', 'Inhibition'], axis=1)
if isTAACF:
    x_cv = x_cv.drop(['SMILES', 'Inhibition'], axis=1)
x_test = x_test.drop(['SMILES', 'Inhibition'], axis=1)

#add label columns
X_train['Activity'] = y_train.tolist()
if isTAACF:
    x_cv['Activity'] = y_cv.tolist()
x_test['Activity'] = y_test.tolist()

if isTAACF:
    #concatenate train and cv
    dataset_list = [X_train, x_cv]  
    X_train_cv = pd.concat(dataset_list)



# 3.  get saved model
#---------------------------------------------------------------------------------------------
filename = f'{model}_{metric}'
file_model = f'{model_path}{filename}{suffix}.mdl'
rfClassifier = splits.get_savedmodel(file_model)



# 4.  score train splits
#---------------------------------------------------------------------------------------------
filename = f'{filename}_trainsplits{suffix}'
if isTAACF:
    dfResults = splits.score_trainsplits(X_train_cv, rfClassifier, 'Activity', split_size_start, split_size_increment, metric, cv=cv, random_state=RANDOMSTATE, filename=f'{filename}.csv')
else:
    dfResults = splits.score_trainsplits(X_train, rfClassifier, 'Activity', split_size_start, split_size_increment, metric, cv=cv, random_state=RANDOMSTATE, filename=f'{filename}.csv')
dfResults.to_csv(f'{filename}_final.csv')