# 1.  Imports
#--------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy import genfromtxt

import joblib
import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

from itertools import product

#common training code
#to import common file
import modelquantification_common as quantify




def no_information_rate(targets, predictions, loss_fn):
    combinations = np.array(list(product(targets, predictions)))
    return loss_fn(combinations[:, 0], combinations[:, 1])


# 2.  Global variables
#--------------------------------------------------------------------------------------------
#models taacf:  wrf, crf
#models mlsmr:  wrf, rfu
RANDOMSTATE = 42
model = 'wrf'
metric = 'f1'


isTAACF = True
if isTAACF:
    dataset = 'TAACF-SRIKinase'
    suffix = '_TS'    
    #filename = f'rf_undersampling_RSCV_f1'
    #filename = f'rf_bootstrap_RSCV_f1'
else:
    dataset = 'MLSMR'
    suffix = '_M'
    #filename = f'rf_undersampling_RSCV_f1'
    #filename = f'rf_bootstrap_RSCV_f1'

data_path = 'ignore/data/'
model_path = f'ignore/model/'
filename = f'{model}_{metric}'

#data_path_base = f'../model evaluation/ignore/data/{dataset}/'
#model_path = f'../model evaluation/ignore/model/{dataset}/'
#results_path = f'../model quantification/ignore/data/{dataset}/'

cv = 10



# 3.  load data
#---------------------------------------------------------------------------------------------
# MLSMR uses a 80/20 split
#continue to use the same split for baseline performance analysis, training, and future hyperparameter optimization
X_train = pd.read_csv(f'{data_path}X_train{suffix}.csv', index_col=0) 
y_train = genfromtxt(f'{data_path}y_train{suffix}.csv', delimiter=' ')
x_test = pd.read_csv(f'{data_path}x_test{suffix}.csv', index_col=0) 
y_test = genfromtxt(f'{data_path}y_test{suffix}.csv', delimiter=' ')


if isTAACF:
    # TAACF-SRIKinase uses a 40-30-30 split
    x_cv = pd.read_csv(f'{data_path}x_cv{suffix}.csv', index_col=0) 
    y_cv = genfromtxt(f'{data_path}y_cv{suffix}.csv', delimiter=' ')

    # smiles & activity are needed for mol identification
    # TAACF:  concatenate train and cv for bootstrapping
    ''''''
    dataset_list = [X_train, x_cv]  
    X_632plus = pd.concat(dataset_list)
    y_632plus = np.concatenate((y_train, y_cv))
    X_boot = pd.concat(dataset_list)
    y_boot = np.concatenate((y_train, y_cv))
    X_boot['Activity'] = y_boot.tolist()

    #remove inhibition and SMILES columns
    X_train = X_train.drop(['SMILES', 'Inhibition'], axis=1)
    x_cv = x_cv.drop(['SMILES', 'Inhibition'], axis=1)
    x_test = x_test.drop(['SMILES', 'Inhibition'], axis=1)
    X_632plus = X_632plus.drop(['SMILES', 'Inhibition'], axis=1)
else:
    # MLSMR:  no need to concatenate 
    X_632plus = X_train
    y_632plus = y_train
    X_boot = X_train
    y_boot = y_train
    X_boot['Activity'] = y_boot.tolist()

    #remove inhibition and SMILES columns
    X_train = X_train.drop(['SMILES', 'Inhibition'], axis=1)
    x_test = x_test.drop(['SMILES', 'Inhibition'], axis=1)
    X_632plus = X_632plus.drop(['SMILES', 'Inhibition'], axis=1)



# 4.  set bootstrap variables
#---------------------------------------------------------------------------------------------
#TAACF:  use 70% of samples to train
#MLSMR:  use 80% of samples to train
train_size = round(len(X_boot) * .7)

# df to save scores
dfBootstrapResults = pd.DataFrame(columns=['method', 'dataset', 'ds0', 'ds1', 'train0', 'train1', 'test0', 'test1', 'metric', 'round', 'score', 'metric_bootTest', 'metric_bootTrain', 'metric_632', 'metric_test'])

# set the bootstrap rounds to perform
bootstrap_start = 1
bootstrap_end = 1001
bootstrap_range = np.arange(bootstrap_start, bootstrap_end+1)

#set the weight
weight = 0.632



# 5.  get saved model
#---------------------------------------------------------------------------------------------
file_model = f'{model_path}{filename}{suffix}.mdl'
rfClassifier = quantify.get_savedmodel(file_model)
rfClassifier.n_jobs = 50



# 6.  score bootstrap models
#---------------------------------------------------------------------------------------------
for i in bootstrap_range:
    #get bootstrap train sample and drop label column
    X_train_boot, y_train_boot = resample(X_boot, y_boot, replace=True, n_samples=train_size, stratify=y_boot, random_state=RANDOMSTATE)
    X_train_boot = X_train_boot.drop(['Activity'], axis=1)

    #get unique SMILES assigned to train sample
    x_train_boot_SMILES = X_train_boot.SMILES.unique().tolist()

    #get molecules not used in train sample, these will be used for evaluation
    x_test_boot = X_boot[~X_boot.SMILES.isin(x_train_boot_SMILES)]

    #create labels
    y_test_boot = x_test_boot['Activity']
    x_test_boot = x_test_boot.drop(['Activity'], axis=1)

    #get imbalanced ratio counts
    activity, counts = np.unique(y_boot, return_counts=True)
    ds0 = counts[0]
    ds1 = counts[1]
    activity, counts = np.unique(y_train_boot, return_counts=True)
    train0 = counts[0]
    train1 = counts[1]
    activity, counts = np.unique(y_test_boot, return_counts=True)
    test0 = counts[0]
    test1 = counts[1]

    #remove inhibition and SMILES columns
    X_train_boot = X_train_boot.drop(['SMILES', 'Inhibition'], axis=1)
    x_test_boot = x_test_boot.drop(['SMILES', 'Inhibition'], axis=1)


    # train and score model
    rfClassifier.fit(X_train_boot, y_train_boot)
    score_model = rfClassifier.score(x_test_boot, y_test_boot)

    # get f1 score for test set
    y_test_pred = rfClassifier.predict(x_test)
    if metric == 'f1':
        metric_test = f1_score(y_test, y_test_pred)
    elif metric == 'balanced_accuracy':
        metric_test = balanced_accuracy_score(y_test, y_test_pred)

    # predict training accuracy on the whole training set, see .632 boostrap paper
    # in Eq (6.12) in Estimating the Error Rate of a Prediction Rule: Improvement on Cross-Validation"
    # by B. Efron, 1983, https://doi.org/10.2307/2288636
    
    # get f1 score on bootstrap test set
    y_test_boot_pred = rfClassifier.predict(x_test_boot)    
    if metric == 'f1':
        metric_boot_test = f1_score(y_test_boot, y_test_boot_pred)
    elif metric == 'balanced_accuracy':
        metric_boot_test = balanced_accuracy_score(y_test_boot, y_test_boot_pred)

    # get f1 score on bootstrap training set
    y_train_boot_pred = rfClassifier.predict(X_train_boot)
    if metric == 'f1':
        metric_boot_train = f1_score(y_train_boot, y_train_boot_pred)
    elif metric == 'balanced_accuracy':
        metric_boot_train = balanced_accuracy_score(y_train_boot, y_train_boot_pred)

    #calculate weighted f1 score 
    metric_632 = (weight * metric_boot_train) + ((1.0 - weight) * metric_boot_test)

    #632+
    '''
    # computationally expensive, did not run
    if metric == 'f1':
        gamma = no_information_rate(y_632plus, rfClassifier.predict(X_632plus), f1_score)
    elif metric == 'balanced_accuracy':
        gamma = no_information_rate(y_632plus, rfClassifier.predict(X_632plus), balanced_accuracy_score)
    R = (metric_boot_test - metric_boot_train) / (gamma - metric_boot_train)
    weight = 0.632 / (1 - 0.368 * R)
    metric_632plus = (weight * metric_boot_train) + ((1.0 - weight) * metric_boot_test)
    '''

    # save results  
    dfBootstrapResults.loc[len(dfBootstrapResults.index)] = ['bootstrap', 'train/cv', ds0, ds1, train0, train1, test0, test1, metric, i, score_model, metric_boot_test, metric_boot_train, metric_632, metric_test]
    dfBootstrapResults.to_csv(f'{data_path}{filename}{suffix}.csv')


