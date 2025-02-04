import pandas as pd
import numpy as np
from numpy import genfromtxt

import joblib
import os
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import seaborn as sns
import cv2
from IPython.display import display, Image
import dataframe_image as df_image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate

import pandas as pd
import numpy as np
from numpy import genfromtxt

import PIL
from PIL import ImageFont, ImageDraw 
import seaborn as sns
import matplotlib.lines as mplines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import plotly.figure_factory as ff

import pathlib
import math
from cmath import isnan
import itertools
import re
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV, cross_val_score
import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.decomposition import PCA
from functools import partial

import umap
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier

import graphviz as graphviz
from IPython.display import Image
from subprocess import call
import pydotplus
import collections
from collections import Counter





# styles
#-----------------------------------------------------------------------------------------------------
# Set CSS properties for th elements in dataframe
cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'tr:hover td',
    'props': 'background-color: #FAF4B7; font-weight: bold;'
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal; text-align:left;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'font-size: 14px; padding: 5px;'
}
caption_css = {
    'selector': 'caption',
    'props': 'font-size: 16px; padding: 20px; font-weight: bold; color: #922B21;'
}
index_names2 = {
    'selector': '.row_heading',
    'props': 'font-weight:bold; text-align:left;'
}
index_names3 = {
    'selector': '.row_heading',
    'props': 'text-align:left;'
}



def add_imageborder(image_filename, color = [255,255,255], width=5):
    img = cv2.imread(image_filename)
    top, bottom, left, right = [width]*4
    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite(image_filename, img_with_border)



def format_color_groups(df, groupby):
    colors = ['white', '#f5eeee']
    
    x = df.copy()
    factors = list(x[groupby].unique())
    #factors = df.index.get_level_values(0).unique()

    i = 0
    for factor in factors:
        style = f'background-color: {colors[i]}; color: #1B2631'
        x.loc[x[groupby] == factor, :] = style
        
        i = not i
    return x



def format_column_leftalign(df, column):
    if df.name == column:
        return ['text-align: left; color: #1B2631; font-weight: bold'] * len(df)
    return [''] * len(df)



def format_df(df, caption, alignRight=[], hide=[]):
    #apply styles to df
    df_styled = df.style.set_caption(caption)\
        .set_table_styles([index_names3, headers, caption_css])\
        .set_properties(subset=alignRight, **{'text-align': 'right'})\
        .hide(axis='columns', subset=hide)

    return df_styled




# training
# ---------------------------------------------------------------------------------------------------------------
def create_trainsplits(dfFeatures, dfLabel, stratifyColumn, testsize1, testsize2, randomstate, filenames): 
    # add label to  features
    data_features_label = pd.concat([dfLabel, dfFeatures], axis = 1)

    # split data into train and validation/test combo
    X_train, x_test = train_test_split(data_features_label, stratify=data_features_label[stratifyColumn], test_size=testsize1, random_state=randomstate)
    
    #get labels & delete them from train
    # np.ravel:  return a contiguous flattened array
    y_train = np.ravel(pd.DataFrame(X_train['Activity']))
    del X_train['Activity']
    
    if testsize2 > 0:
        # cross validation set and test set
        # splits the validation/test combo
        x_cv, x_test = train_test_split(x_test, stratify=x_test[stratifyColumn], test_size = testsize2, random_state=randomstate)

        #get labels & delete them from cv/test dataset
        y_cv = np.ravel(pd.DataFrame(x_cv['Activity']))
        del x_cv['Activity']

    #get labels & delete them from cv/test dataset
    y_test = np.ravel(pd.DataFrame(x_test['Activity']))
    del x_test ['Activity']      

    # save train and y files
    X_train.to_csv(filenames['x_train'], index=False)
    np.savetxt(filenames['y_train'], y_train, delimiter=" ")
    x_test.to_csv(filenames['x_test'], index=False)
    np.savetxt(filenames['y_test'], y_test, delimiter=" ")      

    if testsize2 > 0:
        # save cv files
        x_cv.to_csv(filenames['x_cv'], index=False)
        np.savetxt(filenames['y_cv'], y_cv, delimiter=" ")   

        return X_train, y_train, x_cv, y_cv, x_test, y_test
    else:
        return X_train, y_train, x_test, y_test
    


def get_trainsplits_filenames(data_path, suffix='', includeCV=False):
    x_train = f'{data_path}x_train{suffix}.csv'
    y_train = f'{data_path}y_train{suffix}.csv'
    x_cv = f'{data_path}x_cv{suffix}.csv'
    y_cv = f'{data_path}y_cv{suffix}.csv'
    x_test = f'{data_path}x_test{suffix}.csv'
    y_test = f'{data_path}y_test{suffix}.csv'

    if includeCV:
        filenames_splitdata = {
            'x_train': x_train,
            'y_train': y_train,
            'x_cv': x_cv,
            'y_cv': y_cv,
            'x_test': x_test,
            'y_test': y_test
            }
    else:
        filenames_splitdata = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
            }

    return filenames_splitdata 



def get_trainsplits(filenames, col_NonDescriptors=[]): 
    # retrieve saved splits for baseline performance analysis, training, and future hyperparameter optimization
    x_train = pd.read_csv(filenames['x_train'], index_col=0)
    y_train = genfromtxt(filenames['y_train'], delimiter=' ')
    x_test = pd.read_csv(filenames['x_test'], index_col=0)
    y_test = genfromtxt(filenames['y_test'], delimiter=' ')

    #remove inhibition and SMILES columns
    x_train = x_train.drop(col_NonDescriptors, axis=1)
    x_test = x_test.drop(col_NonDescriptors, axis=1)

    if 'x_cv' in filenames and 'y_cv' in filenames:
        x_cv = pd.read_csv(filenames['x_cv'], index_col=0) 
        y_cv = genfromtxt(filenames['y_cv'], delimiter=' ')
        x_cv = x_cv.drop(col_NonDescriptors, axis=1)
        return x_train, y_train, x_cv, y_cv, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test
    


def train_models(models, classifiers, train, titles, groups, model_parameters, path_model, path_data, path_image
    , baseline, suffix, data_training, dfScores, predict_split='x_test', predict=True, predictCvTest=False): 

    filename_baseline = ''
    classifier_baseline = ''
    if baseline:
        filename_baseline = '_baseline'
        classifier_baseline = 'Baseline'

    if predict_split == 'x_test':
        predict_on = 'test'
    else:
        predict_on = 'cv'

    predict_dataset = ''
    if predictCvTest:
        if predict_on == 'cv':
            predict_dataset = 'CV'
        else:
            predict_dataset = 'Test'



    for model, classifier_type, train_model, title, group, model_params in zip(models, classifiers, train, titles, groups, model_parameters):
        # set all variables
        #-----------------------------------------------------------------------
        image_path = f'{path_image}{model}\\'
        
        file_model = f'{path_model}{model}{filename_baseline}{suffix}.mdl'
        file_pred = f'{path_data}{model}/{model}{filename_baseline}_y{predict_on}pred{suffix}.csv'
        file_probs = f'{path_data}{model}/{model}{filename_baseline}_y{predict_on}probs{suffix}.csv'

        classifier = f'{model.upper()} {classifier_baseline} {predict_dataset}'
        if predict_on == 'cv':
            filename_metrics = f'{model}{filename_baseline}_{predict_on}{suffix}'  
        else:
            filename_metrics = f'{model}{filename_baseline}{suffix}'      

        # train & make/get predictions
        #-----------------------------------------------------------------------
        if train_model:
            # train model
            if classifier_type == 'RandomForest':
                rfClassifier = train_RF(data_training["x_train"], data_training["y_train"], model_params)
            elif classifier_type == 'BalancedRandomForest':
                rfClassifier = BalancedRandomForestClassifier(**model_params)
                rfClassifier.fit(data_training["x_train"], data_training["y_train"])
            elif classifier_type == 'EasyEnsemble':
                rfClassifier = EasyEnsembleClassifier(**model_params)
                rfClassifier.fit(data_training["x_train"], data_training["y_train"])
        
            # store model
            joblib.dump(rfClassifier, file_model)
        else:
            # get saved model
            model_file = os.path.abspath(file_model)
            rfClassifier = joblib.load(model_file)


        if predict:
            # Make predictions and probabilities for prediction set
            y_pred, y_probs = predict_RF(rfClassifier, data_training[predict_split])

            #store results
            np.savetxt(file_pred, y_pred, delimiter=" ")
            np.savetxt(file_probs, y_probs, delimiter=" ")
        else:
            # if predictions are already made... get stored predictions
            y_pred = genfromtxt(f'{file_pred}', delimiter=' ')
            y_probs = genfromtxt(f'{file_probs}', delimiter=' ')


        # get metrics
        #-----------------------------------------------------------------------
        path_data_model = f'{path_data}{model}/'
        params_metrics = {'classifier': classifier,
            'group': group,
            'y_true': data_training[f"y_{predict_on}"], 
            'y_pred': y_pred, 
            'y_probs': y_probs, 
            #'baseline': baseline,
            'dfScores': dfScores,
            'image_path': image_path,
            'data_path': path_data_model, 
            #'suffix': suffix,
            'filename': filename_metrics,
            'title': title,         
            'bestth_AUC': np.nan,
            'bestth_PrecRecall': np.nan,
            'CreateGraphs': True,
            'DisplayGraphs': False,
            'predict_on': predict_on}

        # get all classification metrics and stores results in dfScores
        results = get_performance_metrics(**params_metrics)
        dfScores = results[2]

    return dfScores



def train_RF(x_train, y_train, params):
    #instantiate random forest    
    rfClassifier = RandomForestClassifier(**params)

    #train model
    rfClassifier.fit(x_train, np.ravel(y_train))

    return rfClassifier


def predict_RF(rfClassifier, x):
    # Make predictions
    y_pred = rfClassifier.predict(x)

    # get probabilities for cross validation set
    # keep probabilities for the positive outcome only
    y_probs = rfClassifier.predict_proba(x)[:,1]

    return y_pred, y_probs


def get_savedmodel(file_model, file_pred, file_probs):
    # get stored model
    model = os.path.abspath(file_model)
    rfClassifier = joblib.load(model)

    # get stored predictions/probabilities
    y_pred = genfromtxt(f'{file_pred}', delimiter=' ')
    y_probs = genfromtxt(f'{file_probs}', delimiter=' ')

    return rfClassifier, y_pred, y_probs





# metrics
# ---------------------------------------------------------------------------------------------------------------
def get_performance_metrics(classifier, group, y_true, y_pred, y_probs, dfScores
    , data_path, image_path='', filename='', title='' 
    , thresholds=['AUC Th', 'PR Th'], bestth_AUC=np.nan, bestth_PrecRecall=np.nan
    , CreateGraphs=True, DisplayGraphs=True, type='', predict_on='test'):


    # set variables
    global IMAGE_PATH
    global DATA_PATH
    global THRESHOLDS

    IMAGE_PATH = image_path
    DATA_PATH = data_path
    THRESHOLDS = thresholds

    filename = re.sub('[\W_]+', '_', str(filename))   
    classifier = format_Classifier(classifier) 

    # set parameters to pass to function
    # not applying threshold, even if passed in, for comparison
    params = {'classifier': classifier,
        'group': group,
        'y_true': y_true, 
        'y_pred': y_pred, 
        'y_probs': y_probs, 
        'dfScores': dfScores,
        'filename': filename,
        'th_lst': thresholds,
        'bestth_AUC': bestth_AUC,
        'bestth_PrecRecall': bestth_PrecRecall,
        'title': title,
        'CreateGraphs': CreateGraphs,
        'DisplayGraphs': DisplayGraphs} 
       
    # call function and get results
    results = create_performance_metrics(**params)
    auc_roc = results[3]
    auc_precision_recall = results[4]

    # if the thresholds are not passed in, use the ones calculated above otherwise we use the thresholds passed in
    if np.isnan(bestth_AUC):
        bestth_AUC = results[5]

    if np.isnan(bestth_PrecRecall):
        bestth_PrecRecall = results[6]

    dfMetrics = results[7]
    dfScores = results[8]

    best = {thresholds[0]: bestth_AUC, thresholds[1]: bestth_PrecRecall}

    pred = []

    # create confusion matrix, roc curve and metrics based on best threshold
    for idx, th_type in enumerate(thresholds):
        # get thrshold
        th_best = best[th_type]

        if th_type == 'AUC Th':
            th_auc = bestth_AUC
            th_pr = np.nan
        else:
            th_auc = np.nan
            th_pr = bestth_PrecRecall

        if np.isnan(th_best) == False:
            # get probabilities >= best threshold, and convert to boolean, to serve as the new y_pred
            y_pred_th = (y_probs > th_best).astype(bool)
            pred.append(y_pred_th)

            classifier_th = classifier + ' ' + th_type
            params = {'classifier': classifier_th, 
            'group': group,
            'y_true': y_true, 
            'y_pred': y_pred_th,
            'y_probs': y_probs,  
            'auc_roc': auc_roc,
            'auc_precision_recall': auc_precision_recall,
            'th_best': th_best,
            'th_type': th_type,
            'thROC': th_auc,
            'thPrecRecall': th_pr,            
            'dfMetrics': dfMetrics,
            'dfScores': dfScores, 
            'title': title,
            'filename': filename,         
            'CreateGraphs': CreateGraphs,            
            'DisplayGraphs': DisplayGraphs}
    
            dfMetrics, dfScores = create_performance_metrics_th(**params)

    return bestth_AUC, bestth_PrecRecall, dfScores, pred



def format_Classifier(classifier):
    classifier = re.sub(r'balanced_accuracy','Balanced Accuracy', classifier)
    classifier = re.sub(r'average_precision','Average Precision', classifier)
    classifier = re.sub(r'roc_auc_weighted','ROC AUC Weighted', classifier)
    classifier = re.sub(r'roc_auc','ROC AUC', classifier)
    classifier = re.sub(r'pr_auc','PR AUC', classifier)    
    classifier = re.sub(r'neg_brier_score','Brier Score', classifier)
    classifier = re.sub(r'neg_log_loss','Log Loss', classifier)
    classifier = re.sub(r'f1_weighted','F1 Weighted', classifier)
    classifier = re.sub(r'f1','F1', classifier)
    classifier = re.sub(r'cohen_kappa_score', "Cohens Kappa", classifier)
    classifier = re.sub(r'hinge_loss','Hinge Loss', classifier)
    classifier = re.sub(r'matthews_corrcoef','Matthews Corr', classifier)

    return classifier



def store_results(df, classifier, group, accuracy_balanced, specificity, sensitivity, precision, recall, f1
    , matthew, youden, auc, auc_pr, cohen_kappa, thROC, thPrecRecall, tp, fp, tn, fn):
    #df to keep scores of models
    if df.empty:
        df = pd.DataFrame(columns=['Group', 'Balanced Acc', 'Specificity', 'Sensitivity'
            , 'Precision', 'Recall', 'F1', 'MCC', 'Youden', 'Kappa'
            , 'AUC', 'AUC TH', 'PR AUC', 'PR TH', 'TP', 'FN', 'TN', 'FP'])

    #print(classifier)
    df.loc[classifier] = [group, accuracy_balanced, specificity, sensitivity
        , precision, recall, f1, matthew, youden, cohen_kappa, auc, thROC
        , auc_pr, thPrecRecall
        , round(int(tp)), round(int(fn)), round(int(tn)), round(int(fp))]

    df["FN"] = df["FN"].astype(int)
    df["TP"] = df["TP"].astype(int)
    df["TN"] = df["TN"].astype(int)
    df["FP"] = df["FP"].astype(int)

    return df



def create_performance_metrics(classifier, group, y_true, y_pred, y_probs, dfScores, 
    th_lst, bestth_AUC=np.nan, bestth_PrecRecall=np.nan,
    title='', filename='', CreateGraphs=True, DisplayGraphs=True):

    dfMetrics = pd.DataFrame()
    column = 'No TH'


    curve_type = 'AUC'
    #-------------------------------------------------------------------------------------------
    #get roc curve and the area under the curve
    fpr_lst, tpr_lst, thresholds = metrics.roc_curve(y_true, y_probs)
    roc_auc = metrics.auc(fpr_lst, tpr_lst)

    # if the thresholds are not passed in, use the ones calculated above otherwise we use the thresholds passed in 
    if np.isnan(bestth_AUC):
        #get best threshold
        J = tpr_lst - fpr_lst
        ix = np.nanargmax(J)
        bestth_AUC = thresholds[ix]
    else:
        ix = thresholds.index(bestth_AUC)
   
    # save data and plot the curve
    save_roc_data(classifier, fpr_lst, tpr_lst, curve_type, filename)
    if CreateGraphs:
        x_label = 'False Positive Rate'
        y_label = 'True Positive Rate'
        plot_auc_curve(fpr_lst, x_label, tpr_lst, y_label, ix, roc_auc, title, filename, best_threshold=bestth_AUC, show=DisplayGraphs)



    curve_type = 'AUCPR'
    #-------------------------------------------------------------------------------------------
    # get precision/recall lists
    precision_lst, recall_lst, thresholds = metrics.precision_recall_curve(y_true, y_probs)  
    save_roc_data(classifier, precision_lst, recall_lst, curve_type, filename=filename)

    # convert to f score
    with np.errstate(divide='ignore',invalid='ignore'):
        fscore = (2 * (precision_lst * recall_lst)) / (precision_lst + recall_lst)
    np.seterr(divide='warn', invalid='warn')

    # locate the index of the largest f score, to get the best_threshold
    ix = np.nanargmax(fscore)

    # if the thresholds are not passed in, use the ones calculated above otherwise we use the thresholds passed in 
    if np.isnan(bestth_PrecRecall):        
        bestth_PrecRecall = thresholds[ix]

    # get precision recall metrics
    auc_precision_recall = metrics.auc(recall_lst, precision_lst)
    precision, recall, f1 = metrics_precision_recall(y_true, y_pred)

    if CreateGraphs:
      no_skill = len(y_pred[y_pred==1]) / len(y_pred)
      plot_recall_curve(no_skill, precision_lst, recall_lst, auc_precision_recall, ix, title, filename, best_threshold=bestth_PrecRecall, show=DisplayGraphs)



    # get metrics 
    #-------------------------------------------------------------------------------------------
    tn, fp, fn, tp, specificity, sensitivity, balanced_acc, youdens_j = metrics_confusion_matrix(y_true, y_pred, CreateGraphs, title, filename, DisplayGraphs)
    cohen_kappa, mcc = metrics_additional(y_true, y_pred) 



    # classification report 
    #-------------------------------------------------------------------------------------------
    if CreateGraphs:
        # zero_division=1, used to remove divide by 0 warning for F1 score
        classification_report = metrics.classification_report(y_true, y_pred, zero_division=1)

        # create classification report image, don't show yet, need to create metrics table and append to image
        plot_classification_report(classification_report, filename, title=title, show=False)

        # store metrics, but don't show.  we are appending that image under the classification report
        params = {'df': dfMetrics, 
                'column': column, 
                'balanced_acc': balanced_acc, 
                'specificity': specificity,
                'sensitivity': sensitivity,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'MCC': mcc,
                'J': youdens_j,
                'roc_auc': roc_auc,
                'auc_pr': auc_precision_recall,
                'cohen_kappa': cohen_kappa,
                'title': title,
                'filename': filename,
                'show': False,
                }
        dfMetrics = plot_metrics(**params)

        # combine the classification report and metrics into one image and show
        plot_classificationRprt_and_metrics(title, filename, sidebyside=False, show=DisplayGraphs)



    #store all results
    #-------------------------------------------------------------------------------------------
    params = {'df': dfScores, 
        'classifier': classifier, 
        'group': group, 
        'accuracy_balanced': balanced_acc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matthew': mcc,
        'youden': youdens_j,
        'auc': roc_auc,
        'auc_pr': auc_precision_recall,
        'cohen_kappa': cohen_kappa,
        'thROC': 0.5,
        'thPrecRecall': 0.5,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
        }
    dfScores = store_results(**params)

    return precision, recall, f1, roc_auc, auc_precision_recall, bestth_AUC, bestth_PrecRecall, dfMetrics, dfScores



def metrics_confusion_matrix(y_true, y_pred, CreateGraphs, title, filename, DisplayGraphs):
    matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp)    #aka True Negative Rate (TNR)
    sensitivity = tp / (tp + fn)    #aka Recall, True Positive Rate (TPR)
    false_positive_rate = fp / (fp + tn) # aka, 1-Specificity
    balanced_acc = (specificity + sensitivity) / 2
    youdens_j = sensitivity - false_positive_rate

    if CreateGraphs:
        plot_confusionmatrix(matrix, title, filename, show=DisplayGraphs)

    return tn, fp, fn, tp, specificity, sensitivity, balanced_acc, youdens_j



def metrics_precision_recall(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1



def metrics_additional(y_true, y_pred):
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    return cohen_kappa, mcc



def create_performance_metrics_th(classifier, group, y_true, y_pred, y_probs,  
      auc_roc, auc_precision_recall, th_best, th_type, thROC, thPrecRecall,
      dfMetrics, dfScores, title, filename, CreateGraphs, DisplayGraphs):   

    # get thrshold, convert it to string to add to graph titles
    th = str(round(th_best, 4))
    

    # get metrics 
    #-------------------------------------------------------------------------------------------
    tn, fp, fn, tp, specificity, sensitivity, balanced_acc, youdens_j = metrics_confusion_matrix(y_true, y_pred, CreateGraphs, title, filename, DisplayGraphs)
    cohen_kappa, mcc = metrics_additional(y_true, y_pred) 
    precision, recall, f1 = metrics_precision_recall(y_true, y_pred)


    # classification report 
    #-------------------------------------------------------------------------------------------
    if CreateGraphs:
        # zero_division=1, used to remove divide by 0 warning for F1 score
        classification_report = metrics.classification_report(y_true, y_pred, zero_division=1)

        # create classification report image, don't show yet, need to create metrics table and append to image
        plot_classification_report(classification_report, filename, title=title, ApplyThreshold=True, show=False, th=th, th_type=th_type)

        # store metrics for threshold applied, but don't show.  we are appending that image under the classification report
        # different logic applied than training notebooks
        # in training we want to apply the th calculated from the model we are currently training
        # in scoring we want to apply a th from a different model all together, thus we empty dfMetrics 
        dfMetrics = pd.DataFrame()
        params = {'df': dfMetrics, 
            'column': f'{th_type}={th}', 
            'balanced_acc': balanced_acc, 
            'specificity': specificity,
            'sensitivity': sensitivity,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'MCC': mcc,
            'J': youdens_j,
            'roc_auc': auc_roc,
            'auc_pr': auc_precision_recall,
            'cohen_kappa': cohen_kappa,
            'title': title,
            'filename': filename,
            'th_type': th_type,
            'show': False,
            }
        dfMetrics = plot_metrics(**params)

        # combine the classification report and metrics into one image and show
        plot_classificationRprt_and_metrics(title, filename, sidebyside=False, show=DisplayGraphs, th_type=th_type)


    #store all results
    #-------------------------------------------------------------------------------------------
    params = {'df': dfScores, 
        'classifier': classifier, 
        'group': group, 
        'accuracy_balanced': balanced_acc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matthew': mcc,
        'youden': youdens_j,
        'auc': auc_roc,
        'auc_pr': auc_precision_recall,
        'cohen_kappa': cohen_kappa,
        'thROC': thROC,
        'thPrecRecall': thPrecRecall,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
        }

    dfScores = store_results(**params)
    
    return dfMetrics, dfScores





# ROC & Precision Recall Curve
#-----------------------------------------------------------------------------------------------------
def save_roc_data(classifier, list1, list2, curve_type, filename):    
    #df to keep scores of models
    if type == 'AUCPR':
        df = pd.DataFrame(columns=['Precision', 'Recall'])
    else:
        df = pd.DataFrame(columns=['FPR', 'TPR'])
    
    df.loc[classifier] = [list1, list2]

    directory = 'AUC'
    prefix1 = 'FPR'
    prefix2 = 'TPR'
    if curve_type == 'AUCPR':
        directory = 'PrecisionRecall'
        prefix1 = 'Precision'
        prefix2 = 'Recall'

    np.savetxt(f'{DATA_PATH}{directory}/{prefix1}_{filename}.csv', list1, delimiter=" ")
    np.savetxt(f'{DATA_PATH}{directory}/{prefix2}_{filename}.csv', list2, delimiter=" ")
    


def plot_auc_curve(fpr, x_label, tpr, y_label, ix, roc_auc, title, filename='', best_threshold=np.nan, show=True):
    #plot ROC curve
    legend_best = f'Threshold: {round(best_threshold, 3)}'
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')

    plt.plot([0,1],[0,1], linestyle=':', color='#EB5353', label = 'No skill')    
    plt.plot(fpr, tpr, color = '#0F2C67', label = 'AUC: %0.2f'% roc_auc,)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color = '#F7AA00', label=legend_best, s=150)
    plt.xlabel(x_label, fontsize=14, labelpad=20)
    plt.ylabel(y_label, fontsize=14, labelpad=20)
    plt.legend(loc = 'lower right')
        
    title = f'{title}\nROC Curve'
    plt.title(title, fontsize=12, y=1.05, pad=10, fontweight='bold', color='#363062') 

    #save image, add border and display
    image = ''
    if len(filename) > 0:
        image = f'{IMAGE_PATH}{filename}_ROCAUC.jpg'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        add_imageborder(image)

    plt.legend(loc = 'lower right', fontsize="9")    

    if show:
        plt.show()

    plt.close()

    return image



def plot_recall_curve(no_skill, precision, recall, auc_precision_recall, ix, title, filename, best_threshold=np.nan, show=True):
    #adapted from https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/   
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')
    legend_best = f'TH: {round(best_threshold, 3)}'    
    plt.title(f'{title}\nPrecision Recall', fontsize=16, y=1.05, pad=10, fontweight='bold', color='#363062')
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, label = 'AUC-PR = %0.2f'% auc_precision_recall)
    plt.scatter(recall[ix], precision[ix], marker='o', color='#008000', label=legend_best, s=100)
    plt.legend(loc = 'lower right')
    plt.ylabel('Precision', fontsize=14, labelpad=20)
    plt.xlabel('Recall', fontsize=14, labelpad=20)

    #save image, add border and display
    image = f'{IMAGE_PATH}{filename}_PrecisionRecall.jpg'      
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    add_imageborder(image)

    if show:
        display(Image(image))
    
    plt.close()





# confusion matrix
#-----------------------------------------------------------------------------------------------------
def plot_confusionmatrix(matrix, title, filename, th_applied=False, show=False, width=0, th='', th_type=''):    
    # create plot
    fig = plt.figure()
    fig.patch.set_facecolor('#FFFFFF')
    cfm_plot = sns.heatmap(matrix, cmap='Blues', annot=True, fmt='.0f'
        , annot_kws = {'size':16}, xticklabels = ["Inactive", "Active"] , yticklabels = ["Inactive", "Active"]
        , facecolor='white')
    cfm_plot.set_xlabel('Predicted',fontsize=16, labelpad=28, color='#4D4C7D')
    cfm_plot.set_ylabel('True',fontsize=16, labelpad=28, color='#4D4C7D')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    #save w/diffferent name depending if threshold has been applied
    if th_applied:
        title = f'{title}\nConfusion Matrix - {th_type} = {th}'
    else:
        title = f'{title}\nConfusion Matrix'
    
    plt.title(title, fontsize=16, pad=20, fontweight='bold', color='#363062')

    type = ''
    if len(th_type) > 0:
        type = '_' + th_type.replace(' ', '')

    # save plot to image and display    
    image = f'{IMAGE_PATH}{filename}_ConfusionMatrix{type}.jpg'    
    cfm_plot.figure.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    
    if width > 0:
        add_imageborder(image)

    if show:
        display(Image(image))

    plt.close()





# classification report
#-----------------------------------------------------------------------------------------------------
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    
    #god knows what this is
    pc.update_scalarmappable()
    ax = pc.axes

    #Use zip BELOW IN PYTHON 3
    #zip:  creates an iterator that will aggregate elements from two or more iterables
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        
        #sets font color
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)

        #if given class counts, add a gray background        
        if value > 1:
            ax.add_patch(mpatch.Rectangle((3, 0), 200, 2, color='gainsboro'))
            ax.text(x, y, round(int(value), 0), ha="center", va="center", color='black', fontsize=15, fontweight='bold')            
        #otherwise, regular heatmap
        else: 
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, fontsize=15, fontweight='bold', **kw)



def plot_classification_report(classification_report, filename, number_of_classes=2, title='', ApplyThreshold=False, show=True, th='', th_type=''):
    '''
    Plot scikit-learn classification report.
    Extension based on
        - https://stackoverflow.com/a/31689645/395857 
        - https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    '''
    lines = classification_report.split('\n')
    
    '''
    get plot info
    '''
    #drop initial lines
    lines_plot = lines[2:]

    classes = []
    AUC = []
    support = []
    class_names = []
    for line in lines_plot[: number_of_classes]:
        t = list(filter(None, line.strip().split('  ')))
        if len(t) < 4: continue
        classes.append(t[0])

        #v = [float(x) for x in t[1: len(t) - 1]]
        v = [float(x) for x in t[1: len(t)]]
        AUC.append(v)
        support.append(int(t[-1]))
        
        class_label = str(t[0])
        if (class_label == '0') | (class_label == '0.0'):
            name = 'Inactive'
        else:
            name = 'Active'
        class_names.append(name)

    xticklabels = ['Precision', 'Recall', 'F1-score', 'Class Total']
    yticklabels = ['{0}'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    correct_orientation = True
    AUC = np.array(AUC)

    title = f'{title}\nClassification Report'
    thsuffix = ''  
    if ApplyThreshold:
        title = f'{title}\n{th_type} threshold = {th}'   
        thsuffix = '_' + th_type.replace(' ', '')    


    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''
    # initialize plot
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor('#FFFFFF')   

    #error fix for matplotlib:  https://github.com/matplotlib/matplotlib/issues/21723
    plt.rcParams['axes.grid'] = False
    ax.grid(False)

    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdYlGn', vmin=0.0, vmax=1.0)

    ax.set_title(f'{title}', fontsize = 17, pad=55, fontweight='bold', color='#363062')

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False, fontsize=14, color='#293462')
    ax.set_yticklabels(yticklabels, minor=False, fontsize=14, color='#293462')

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    #ax1 = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    # Add color bar
    # creates axes on the right side of ax with a width of 5%
    # and the padding between cax and ax at 0.05 inch.    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(c, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    #save image and add border 
    image = f'{IMAGE_PATH}{filename}_ClassificationReport{thsuffix}.jpg'
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white', dpi=200)
    add_imageborder(image)
    
    if show:
        display(Image(image))

    plt.close()



def plot_metrics(df, column, balanced_acc, specificity, sensitivity
    , precision, recall, f1, MCC, J, roc_auc, auc_pr, cohen_kappa
    , title, filename, th_type='', show=True):
    #df to keep scores of models
    if df.empty:
        df = pd.DataFrame(
            index=['Balanced Accuracy', 'Specificity', 'Sensitivity'
                , 'Precision', 'Recall', 'F1', 'MCC', 'Youden''s J', 'AUC', 'AUC-PR'
                , 'Cohen Kappa']
            )

    #set column values
    column_values = [balanced_acc, specificity, sensitivity, precision, recall, f1
        , MCC, J, roc_auc, auc_pr, cohen_kappa]

    #set column name
    df[column] = column_values
    df[column] = df[column].round(decimals = 3)

    #filename suffix denoting threshold 
    if len(th_type) > 0:
        th_type = '_' + th_type.replace(' ', '')

    df_filename = f'{IMAGE_PATH}{filename}_Metrics{th_type}.jpg'    
    df = df.style.set_table_styles([index_names, headers, caption_css]).format({column: '{:,.3f}'})

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_image.export(df, df_filename, table_conversion='matplotlib')

    #add border to image
    add_imageborder(df_filename)

    if show:
        display(Image(df_filename))

    return df



def plot_classificationRprt_and_metrics(title, filename, ApplyThreshold=False, th_type='', sidebyside=True, show=True):
    #bug with RGB colors:  https://stackoverflow.com/questions/50630825/matplotlib-imshow-distorting-colors
    
    #filename suffix denoting threshold 
    if len(th_type) > 0:
        th_type = '_' + th_type.replace(' ', '')

    #build filenames
    file_metrics = f'{IMAGE_PATH}{filename}_Metrics{th_type}.jpg'
    file_classification = f'{IMAGE_PATH}{filename}_ClassificationReport{th_type}.jpg'

    #read images
    image_metrics = cv2.imread(file_metrics)
    image_classification = cv2.imread(file_classification)

    if sidebyside:
        # side by side
        #----------------------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.75, 1]}, figsize=(10, 12))
        ax1.imshow(image_classification[...,::-1])
        ax1.axis('off')
        ax2.imshow(image_metrics)
        ax2.axis('off')
        plt.axis('off')
        fig.tight_layout()
    else:
        # one above the other
        #----------------------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [.75, 1]}, figsize=(12, 8))
        ax1.imshow(image_classification[...,::-1])
        ax1.axis('off')
        ax2.imshow(image_metrics)
        ax2.axis('off')
        plt.axis('off')
        fig.tight_layout()     
    
    #save image and add border
    image = f'{IMAGE_PATH}{filename}_ClassificationReportMetrics{th_type}.jpg'   
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')   

    if show:
        display(Image(image))

    plt.close()





# model comparisons
#-----------------------------------------------------------------------------------------------------
def dataset_counts(x_train, y_train, x_test, y_test, x_cv=pd.DataFrame(), y_cv=pd.DataFrame()):
    train_ttl = x_train.shape[0]
    train_active = np.count_nonzero(y_train == 1)
    train_inactive = np.count_nonzero(y_train == 0)
    perc_train = "{:.4%}".format((train_active/train_ttl))

    test_ttl = x_test.shape[0]
    test_active = np.count_nonzero(y_test == 1)
    test_inactive = np.count_nonzero(y_test == 0)
    perc_test = "{:.4%}".format((test_active/test_ttl))

    if x_cv.empty: 
        cv_ttl = 0
        cv_active = 0
        cv_inactive = 0
        perc_cv = 0
    else:
        cv_ttl = x_cv.shape[0]
        cv_active = np.count_nonzero(y_cv == 1)
        cv_inactive = np.count_nonzero(y_cv == 0)
        perc_cv = "{:.4%}".format((cv_active/cv_ttl))

    ttl_molecules = train_ttl + test_ttl + cv_ttl
    ttl_actives = train_active + test_active + cv_active
    ttl_inactives = train_inactive + test_inactive + cv_inactive
    perc_all = "{:.4%}".format((ttl_actives/ttl_inactives))

    if x_cv.empty:
        table = [
            ["Molecules", train_ttl, test_ttl, ttl_molecules]
            ,["Actives", train_active, test_active, ttl_actives]
            , ["Inactives", train_inactive, test_inactive, ttl_inactives]
            , ["% Active", perc_train, perc_test, perc_all]]
        print(tabulate(table, headers=["Count","Training Dataset", "Test Dataset", "Total"]))
    else:
        table = [
            ["Molecules", train_ttl, cv_ttl, test_ttl, ttl_molecules]
            ,["Actives", train_active, cv_active, test_active, ttl_actives]
            , ["Inactives", train_inactive, cv_inactive, test_inactive, ttl_inactives]
            , ["% Active", perc_train, perc_cv, perc_test, perc_all]]
        print(tabulate(table, headers=["Count","Training Dataset", "CV Dataset", "Test Dataset", "Total"]))



def highlight_max(data, colors, top):
    attr = 'font-weight:bold; color: {};'.format(colors[top])

    if type(top) == int:
        top100 = data.nlargest(n=100, keep='last').values
        topvalues = np.sort(np.unique(top100)[-3 : ])

        if top == 1:
            is_max = data == data.max()
            return [attr if v else '' for v in is_max]
        elif top == 2:   
            try:
                if topvalues.size == 1:
                    is_max = np.isnan(data)
                    return ['' if v else '' for v in is_max] 
                else:
                    max = topvalues[::-1][1]               
                    is_max = data == max
                    return [attr if v else '' for v in is_max]  

            except IndexError:
                return ['']
        elif top == 3:
            try:
                if topvalues.size < 3:
                    is_max = np.isnan(data)
                    return ['' if v else '' for v in is_max] 
                else:
                    max = topvalues[::-1][2] 
                    is_max = data == max
                    return [attr if v else '' for v in is_max]
            except IndexError:
                return ['']
        
        return ['']        



def highlight_min(data, colors, top):
    attr = 'font-weight:bold; color: {}'.format(colors[top])

    if type(top) == int:
        top100 = data.nsmallest(n=100, keep='last').values
        topvalues = np.sort(np.unique(top100)[-3 : ])

        if top == 1:
            is_min = data == data.min()
            return [attr if v else '' for v in is_min]
        elif top == 2:   
            try: 
                if topvalues.size == 1:
                    is_min = np.isnan(data)
                    return ['' if v else '' for v in is_min] 
                else:             
                    min = np.sort(np.unique(top100))[1]
                    is_min = data == min         
                    return [attr if v else '' for v in is_min]       
            except IndexError:
                return ['']
        elif top == 3:
            try:
                if topvalues.size < 3:
                    is_min = np.isnan(data)
                    return ['' if v else '' for v in is_min] 
                else:   
                    min = np.sort(np.unique(top100))[2] 
                    #print(f'data {data.name}')
                    is_min = data == min
                    return [attr if v else '' for v in is_min]
            except IndexError:
                return ['']
        
        return ['']        
    else:
        return ['']



def highlight_background(data, color):
    attr = 'background-color: {}'.format(color)
    return [attr if c else '' for c in data]



def store_results(df, classifier, group, accuracy_balanced, specificity, sensitivity, precision, recall, f1
    , matthew, youden, auc, auc_pr, cohen_kappa, thROC, thPrecRecall, tp, fp, tn, fn):
    #df to keep scores of models
    if df.empty:
        df = pd.DataFrame(columns=['Group', 'Balanced Acc', 'Specificity', 'Sensitivity'
            , 'Precision', 'Recall', 'F1', 'MCC', 'Youden', 'Kappa'
            , 'AUC', 'AUC TH', 'PR AUC', 'PR TH', 'TP', 'FN', 'TN', 'FP'])

    #print(classifier)
    df.loc[classifier] = [group, accuracy_balanced, specificity, sensitivity
        , precision, recall, f1, matthew, youden, cohen_kappa, auc, thROC
        , auc_pr, thPrecRecall, round(int(tp)), round(int(fn)), round(int(tn)), round(int(fp))]

    df["FN"] = df["FN"].astype(int)
    df["TP"] = df["TP"].astype(int)
    df["TN"] = df["TN"].astype(int)
    df["FP"] = df["FP"].astype(int)

    return df



def get_ClassifierScores(df, caption='', imgfilename='', groupby=True, hide=[], top=1, return_style=True, keep_index=False, first_column=''):
    # reset integer columns, so they dont' show up w/decimals
    df["FN"] = df["FN"].astype(int)
    df["TP"] = df["TP"].astype(int)
    df["TN"] = df["TN"].astype(int)
    df["FP"] = df["FP"].astype(int)    

    if keep_index == False:
        # copy index to a columnm for sorting   
        if not ('Classifier' in df.columns):
            df.insert(1, 'Classifier', df.index)    
        df = df.sort_values(by = ['Group', 'Classifier'], ascending = [True, True])

        # remove index
        df = df.reset_index()
        df = df.drop('index', axis=1)
        df = df.sort_values(by = ['Group', 'Classifier'], ascending = [True, True])
        first_column = 'Classifier'
    

    colors = {1:'#B22727', 2:'#e97728', 3:'#80558C'}
    color_first = '#B22727'
    color_second = '#e97728'
    color_third = '#80558C'
    max_columns = ['Balanced Acc', 'Specificity', 'Sensitivity', 'Precision', 'Recall', 'F1', 'MCC', 'Youden', 'Kappa', 'AUC', 'PR AUC', 'TP', 'TN'] 
    min_columns = ['FN', 'FP']

    top1 = top2 = top3 = 1
    if top == 2:
        top2 = 2
    if top == 3:
        top2 = 2
        top3 = 3


    if len(hide) > 0:         
        df_styled = df.style.apply(format_color_groups, groupby='Group', axis=None)\
            .apply(format_column_leftalign, column=first_column)\
            .apply(highlight_max, colors=colors, top=top1, axis=0, subset = max_columns)\
            .apply(highlight_max, colors=colors, top=top2, axis=0, subset = max_columns)\
            .apply(highlight_max, colors=colors, top=top3, axis=0, subset = max_columns)\
            .apply(highlight_min, colors=colors, top=top1, axis=0, subset = min_columns)\
            .apply(highlight_min, colors=colors, top=top2, axis=0, subset = min_columns)\
            .apply(highlight_min, colors=colors, top=top3, axis=0, subset = min_columns)\
            .hide(axis='columns', subset=hide)\
            .hide(axis='index')\
            .set_caption(caption)\
            .set_table_styles([cell_hover, headers, caption_css])\
            .format(None, na_rep='')\
            .format({
                'Balanced Acc': '{:,.4f}',
                'Specificity': '{:,.4f}',
                'Sensitivity': '{:,.4f}',
                'Precision': '{:,.4f}',
                'Recall': '{:,.4f}',
                'MCC': '{:,.4f}',
                'F1': '{:,.4f}',
                'Youden': '{:,.4f}',
                'Kappa': '{:,.4f}',
                'AUC': '{:,.4f}',
                'AUC TH': '{:,.4f}',
                'PR AUC': '{:,.4f}',
                'PR TH': '{:,.4f}',
                'TH ROC': '{:,.4f}',
                'TH Recall': '{:,.4f}',
                }, precision=4, na_rep='')
    else:
        df_styled = df.style.apply(format_color_groups, groupby='Group', axis=None)\
            .apply(format_column_leftalign, column=first_column)\
            .apply(highlight_max, colors=colors, top=top1, axis=0, subset = max_columns)\
            .apply(highlight_max, colors=colors, top=top2, axis=0, subset = max_columns)\
            .apply(highlight_max, colors=colors, top=top3, axis=0, subset = max_columns)\
            .apply(highlight_min, colors=colors, top=top1, axis=0, subset = min_columns)\
            .apply(highlight_min, colors=colors, top=top2, axis=0, subset = min_columns)\
            .apply(highlight_min, colors=colors, top=top3, axis=0, subset = min_columns)\
            .hide(axis='index')\
            .set_caption(caption)\
            .set_table_styles([cell_hover, headers, caption_css])\
            .format(None, na_rep='')\
            .format({
                'Balanced Acc': '{:,.4f}',
                'Specificity': '{:,.4f}',
                'Sensitivity': '{:,.4f}',
                'Precision': '{:,.4f}',
                'Recall': '{:,.4f}',
                'MCC': '{:,.4f}',
                'F1': '{:,.4f}',
                'Youden': '{:,.4f}',
                'Kappa': '{:,.4f}',
                'AUC': '{:,.4f}',
                'AUC TH': '{:,.4f}',
                'PR AUC': '{:,.4f}',
                'PR TH': '{:,.4f}',
                'TH ROC': '{:,.4f}',
                'TH Recall': '{:,.4f}',
            }, precision=4, na_rep='')


    if len(imgfilename) > 0:     
        df_filename = f'{imgfilename}.jpg'
        df_image.export(df_styled, df_filename)

        # add border to image
        add_imageborder(df_filename, width=20)

    if return_style:
        return df_styled
    else:
        return df



def get_auc_data(path, models, xy, predict='', suffix='', baseline=False, metrics=[], scoring_suffix=''):
    suffix_baseline = ''
    if baseline:
        suffix_baseline = '_baseline'

    # create new dataframe
    df = pd.DataFrame(columns=[xy[0], xy[1]])

    if len(predict) > 0:
        predict = f'_{predict}'
    
    # iterate through the list of classifiers
    for model in models:
        if metrics:
            for metric in metrics:
                # get classifier's precision & recall
                x = genfromtxt(f'{path.replace("@model", model)}{xy[0]}_{model}_{metric}{predict}{suffix}{scoring_suffix}.csv', delimiter=' ')
                y = genfromtxt(f'{path.replace("@model", model)}{xy[1]}_{model}_{metric}{predict}{suffix}{scoring_suffix}.csv', delimiter=' ')

                index_name = f'{model} {metric}'\
                    .replace('balanced_accuracy', 'Bal Acc')\
                    .replace('f1', 'F1')

                # append row
                df.loc[index_name] = [x, y]
        else:
            # get classifier's precision & recall
            x = genfromtxt(f'{path.replace("@model", model)}{xy[0]}_{model}{suffix_baseline}{predict}{suffix}{scoring_suffix}.csv', delimiter=' ')
            y = genfromtxt(f'{path.replace("@model", model)}{xy[1]}_{model}{suffix_baseline}{predict}{suffix}{scoring_suffix}.csv', delimiter=' ')

            df.loc[model] = [x, y]

    return df



def get_auc_dataOpt_multidatasets(classifiers, xy, datasets={}):
    # create new dataframe
    df = pd.DataFrame(columns=[xy[0], xy[1]])

    for data in classifiers:
        for path, clsfrs in data.items():  
            # iterate through the list of classifiers
            for classifier in clsfrs:
                # get classifier's precision & recall
                x = genfromtxt(f'{path}{xy[0]}_{classifier}.csv', delimiter=' ')
                y = genfromtxt(f'{path}{xy[1]}_{classifier}.csv', delimiter=' ')

                # append row
                index_name = classifier.replace('wrf_', 'Weighted RF ')\
                    .replace('crf_', 'Classic RF ')\
                    .replace('rfu_', 'RF w/Undersampling ')\
                    .replace('balanced_accuracy', 'Bal Acc ')\
                    .replace('f1', 'F1 ')

                for key, value in datasets.items():
                    if key in index_name:
                        index_name = index_name.replace(key, '')
                        index_name = value + index_name  
                index_name = index_name.replace('_', '')

                df.loc[index_name] = [x, y]

    return df


def get_auc_dataOpt(classifiers, xy, datasets={}, predict='', suffix='', scoring_suffix=''):
    # create new dataframe
    df = pd.DataFrame(columns=[xy[0], xy[1]])

    if len(predict) > 0:
        predict = f'_{predict}'

    for data in classifiers:
        for path, clsfrs in data.items():  
            # iterate through the list of classifiers
            for classifier in clsfrs:
                # get classifier's precision & recall
                x = genfromtxt(f'{path}{xy[0]}_{classifier}{predict}{suffix}{scoring_suffix}.csv', delimiter=' ')
                y = genfromtxt(f'{path}{xy[1]}_{classifier}{predict}{suffix}{scoring_suffix}.csv', delimiter=' ')

                # append row
                index_name = classifier.replace('rf_', '')\
                    .replace('classic', 'Classic')\
                    .replace('classweight', 'Class Weight')\
                    .replace('_RSCV_', ' RSCV ')\
                    .replace('_hyperopt_', ' HO ')\
                    .replace('balanced_accuracy', 'Bal Acc ')\
                    .replace('f1', 'F1 ')

                for key, value in datasets.items():
                    if key in index_name:
                        index_name.replace(key, '')
                        index_name = value + index_name                

                df.loc[index_name] = [x, y]

    return df



def plot_auc_curve_comparison(df, title, imgfilename='', subtitle='', footnote = '', xy = (0.82, -0.2)):
    color=[
        '#277BC0', '#FFB200', '#F675A8', '#D61C4E',
        '#1F4690', '#EF5B0C', '#17becf', '#31087B',
        '#554994', '#277BC0', '#B1D7B4', '#A66CFF',
        '#F29393', '#CDF0EA', '#54BAB9', '#FF87CA'
        ]

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#FFFFFF')
    
    if len(subtitle) > 0:
        subtitle = '\n' + subtitle
    
    plt.title(f'{title}{subtitle}', fontsize=16, y=1.02, pad=10, fontweight='bold', color='#363062')
    
    #iterates throw each row, and plots the results for that row
    idx = 0
    for index, row in df.iterrows():
        classifier = index.replace('crf', 'CRF')\
            .replace('wrf', 'WRF')\
            .replace('rfee', 'RFEE')\
            .replace('rfbs', 'RFBS')\
            .replace('rfu', 'RFU')
        
        fpr = row["FPR"]
        tpr = row["TPR"]

        plt.plot(fpr, tpr, label = f'{classifier}', color=color[idx])
        idx += 1

    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate', fontsize=14, labelpad=20)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=20)

    if len(footnote) > 0:
        plt.annotate(footnote,
            xy = xy,
            xycoords='axes fraction',
            ha='right',
            va='center',
            fontsize=9)
    
    #save image, add border and display
    if len(imgfilename) > 0:
        image = f'{imgfilename}.png'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.35, facecolor='white')
        add_imageborder(image)

    plt.show()
    plt.close()



def plot_recall_curve_comparison(df, title, imgfilename='', subtitle='', color=[], footnote='', xy = (0.82, -0.2)):
    if len(color) == 0:
        color=[
        '#277BC0', '#FFB200', '#F675A8', '#D61C4E',
        '#1F4690', '#EF5B0C', '#17becf', '#31087B',
        '#554994', '#277BC0', '#B1D7B4', '#A66CFF',
        '#F29393', '#CDF0EA', '#54BAB9', '#FF87CA'
        ]
    
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#FFFFFF')

    if len(subtitle) > 0:
        subtitle = '\n' + subtitle
    
    plt.title(f'{title}{subtitle}', fontsize=16, y=1.02, pad=10, fontweight='bold', color='#363062')

    #import ast
    #iterates throw each row, and plots the results for that row
    idx = 0
    for index, row in df.iterrows():
        #print(index)
        classifier = index.replace('crf', 'CRF')\
            .replace('wrf', 'WRF')\
            .replace('rfee', 'RFEE')\
            .replace('rfbs', 'RFBS')\
            .replace('rfu', 'RFU')
        
        precision = row["Precision"]
        recall = row["Recall"]

        plt.plot(recall, precision, label = f'{classifier}', color=color[idx])
        idx += 1

    plt.legend(loc = 'upper right')
    plt.ylabel('Precision', fontsize=14, labelpad=20)
    plt.xlabel('Recall', fontsize=14, labelpad=20)

    if len(footnote) > 0:
        plt.annotate(footnote,
            xy = xy,
            xycoords='axes fraction',
            ha='right',
            va='center',
            fontsize=9)

    #save image, add border and display
    if len(imgfilename) > 0:
        image = f'{imgfilename}.png'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        add_imageborder(image)

    plt.show()
    plt.close()





# scoring
# ---------------------------------------------------------------------------------------------------------------
def get_feature_importances(rfClassifier, columns, title, filename, threshold, topFeatureCount=10, show=False, replaceBackslash=True):
    feature_importances = pd.DataFrame({'feature': list(columns),
                   'importance': rfClassifier.feature_importances_}).\
                    sort_values('importance', ascending = False)

    norm_fi = plot_feature_importances(feature_importances, title, filename, topFeatureCount, threshold, show, replaceBackslash)

    return feature_importances


def plot_feature_importances(df, title, filename, n = 10, threshold = None, show=True, replaceBackslash=True):
    # adapted from:  https://www.kaggle.com/code/willkoehrsen/a-complete-introduction-and-walkthrough/notebook
    
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    #plt.style.use('fivethirtyeight')
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 14
    
    # Bar plot of n most important features
    df.loc[:n-1, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (6, 7),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 16, labelpad=20); plt.ylabel(''); 
    plt.title(f'{title}\n{n} Most Important Features', size=18, fontweight='bold', color='#363062')
    plt.gca().invert_yaxis()

    #save image, add border and display
    image = f'{IMAGE_PATH}{filename}_Feature.jpg'      
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    add_imageborder(image)

    if show:
        display(Image(image))
    plt.close()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16, labelpad=20)
        plt.ylabel('Cumulative Importance', size = 16, labelpad=20); 
        plt.title(f'{title}\nCumulative Feature Importance', size=16, pad=10, fontweight='bold', color='#363062');
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        legend = ('{} features = {:.0f}% importance.'.format(importance_index + 1, 100 * threshold))
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = ':', colors = 'red', label=legend)
        plt.legend(loc = 'lower right')

        #save image, add border and display
        if replaceBackslash:
            filename = re.sub('[\W_]+', '_', str(filename))
        image = f'{IMAGE_PATH}{filename}_FeatureCumm.jpg'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        add_imageborder(image)
        
        if show:
            display(Image(image))

        plt.close()
    
    return df



def save_ClassifierFeatures(df, dfFeatureImportance, topFeatureCount, columnName, filename, suffix, show=False):
    #df to keep scores of models
    if df.empty:
        # set index
        columns = []
        for ix in range(topFeatureCount):
            columns.append(f'{ix + 1}')

        df = pd.DataFrame(
            index=columns
            )
    #set column values
    dfFeatureImportance = dfFeatureImportance.sort_values('importance', ascending = False).reset_index(drop = True)
    dfFeatureImportance = dfFeatureImportance.loc[:topFeatureCount-1, :]
    column_values = dfFeatureImportance['feature'].tolist()

    #set column name
    df[columnName] = column_values

    #apply styles to df and save as image
    filename = re.sub('[\W_]+', '_', str(filename))
    df_filename = f'{IMAGE_PATH}{filename}_TopFeatures.jpg'
    df.style.set_caption('RF Feature Importance')\
             .set_table_styles([index_names, headers, caption_css])

    df_image.export(df, df_filename, table_conversion='matplotlib')

    #add border to image
    add_imageborder(df_filename)

    df.to_csv(f'{DATA_PATH}TopFeatures{suffix}.csv')

    if show:
        display(Image(df_filename))

    return df



def plot_StyledDataFrame(df, title, imgfilename):
    imgfilename = f'{imgfilename}.png'
    
    #apply styles to df and save as image    
    df_styled = df.style.set_caption(title)\
            .set_table_styles([index_names3, headers, caption_css])
    df_image.export(df_styled, imgfilename, table_conversion='matplotlib')
    
    #add border to image
    border_width = 20
    add_imageborder(imgfilename, width=border_width)

    #display(Image(imgfilename))
    return df_styled



def save_bestparamsRF(df, rsCV, method, model, scoring, cv, n_iter, objIsModel=False, filename=''): 
    if objIsModel:
        score = math.nan
        best_params_ = rsCV.get_params()
    else:
        score = rsCV.best_score_
        best_params_ = rsCV.best_params_
    n_estimators = math.nan
    min_samples_split = math.nan
    min_samples_leaf = math.nan
    max_features = math.nan
    max_depth = math.nan
    criterion = 'n/a'
    class_weight = 'n/a'
    bootstrap = True
    max_terminal_nodes = 'n/a'
    max_samples= 'n/a'


    if 'n_estimators' in best_params_:
        n_estimators = best_params_['n_estimators']
    if 'min_samples_split' in best_params_:   
        min_samples_split = best_params_['min_samples_split']
    if 'min_samples_leaf' in best_params_:
        min_samples_leaf = best_params_['min_samples_leaf']
    if 'max_features' in best_params_:
        max_features = best_params_['max_features']
    if 'max_depth' in best_params_:
        max_depth = best_params_['max_depth']
    if 'criterion' in best_params_:
        criterion = best_params_['criterion']
    if 'class_weight' in best_params_:
        class_weight = best_params_['class_weight']
    if 'bootstrap' in best_params_:
        bootstrap = best_params_['bootstrap']
    if 'max_terminal_nodes' in best_params_:
        max_terminal_nodes = best_params_['max_terminal_nodes']
    if 'max_samples' in best_params_:
        max_samples = best_params_['max_samples']
        

    new_column_values = [method, scoring, cv, n_iter, score, n_estimators, min_samples_split, min_samples_leaf
        , max_features, max_depth, criterion, class_weight, bootstrap, max_terminal_nodes, max_samples]

    if df.empty:
        df = pd.DataFrame(
            index=['method', 'scoring', 'cv', 'n_iter', 'score', 'n_estimators', 
            'min_samples_split', 'min_samples_leaf', 'max_features', 'max_depth', 'criterion', 
            'class_weight', 'bootstrap', 'max_terminal_nodes', 'max_samples']
            )
        #df.index.name="Best Params"

    df[model] = new_column_values
    
    if len(filename) > 0:
        #apply styles to df and save as image
        df_filename = f'{IMAGE_PATH}RF_BestParams'.replace(" ", "") + '.png'
        df_styled = df.style.set_caption('RF Best Parameters')\
                .set_table_styles([index_names, headers, caption_css])
        df_image.export(df_styled, df_filename)

        #add border to image
        add_imageborder(df_filename)
        display(Image(df_filename))

    return df


def plot_DecisionTree(rfClassifier, columns, title, filename, show=True):
    image_dot = f'{IMAGE_PATH}{filename}_DecisonTree.dot'
    image_png = f'{IMAGE_PATH}{filename}_DecisonTree.png'
    
    #need to modify later to pick an estimator to visualize, hard coded to 0 right now
    estimator = rfClassifier.estimators_[0]
    tree.export_graphviz(estimator, out_file=image_dot, feature_names = columns, class_names = True,
        rounded = True, proportion = False, precision = 2, filled = True)

    graph = pydotplus.graph_from_dot_file(image_dot)
    colors = ('#F2F2F2', '#FFE5E1')

    # Convert to png
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    
    #save image, add border and display
    graph.write_png(image_png)
    add_imageborder(image_png)

    if show:
        return Image(filename = image_png)



def score_models(models, metrics, predict, saveFeatureImportance, saveHyperparameters, saveDecisionTree
    , titles, groups, path_model, path_data, path_image
    , suffix, data_scoring, dfScores, dfFeatureImportance=pd.DataFrame(), dfHyperparameters=pd.DataFrame(), predictCvTest=False
    , cv = 'RSKfold: 10Repeats; 5Splits', featureImportance_th = 0.90, topFeatureCount=10, scoring_suffix=''): 

    # set variables
    global IMAGE_PATH
    global DATA_PATH

    IMAGE_PATH = path_image
    DATA_PATH = path_data

    # determine if predicting on cv/test or just test
    if predictCvTest:
        predictions = ['cv', 'test']
    else:
        predictions = ['test']    

    # loop through datasets to predict on
    for predict_on in predictions:
        if predict_on == 'cv':
            predict_dataset = 'CV'
            predict_split = 'x_cv'
        else:
            predict_dataset = ''
            predict_split = 'x_test'

        # loop through metrics the model was optimized for
        for metric in metrics: 

            # loop through the models
            for model, make_predictions, title, group, get_featureImportance, get_hyperparameters, get_decisiontree \
                in zip(models, predict, titles, groups, saveFeatureImportance, saveHyperparameters, saveDecisionTree):

                # set all variables
                #-----------------------------------------------------------------------
                file_model = f'{path_model}{model}_{metric}{suffix}.mdl'
                file_pred = f'{path_data}{model}/{model}_{metric}_y{predict_on}pred{suffix}{scoring_suffix}.csv'
                file_probs = f'{path_data}{model}/{model}_{metric}_y{predict_on}probs{suffix}{scoring_suffix}.csv'   

                image_path = f'{path_image}{model}\\'  

                classifier = f'{model.upper()} {metric} {predict_dataset}'
                if predict_on == 'cv':
                    filename_metrics = f'{model}_{metric}_{predict_on}{suffix}'  
                else:
                    filename_metrics = f'{model}_{metric}{suffix}{scoring_suffix}'      


                # get saved model
                #-----------------------------------------------------------------------
                model_file = os.path.abspath(file_model)
                rfClassifier = joblib.load(model_file)


                # make/get predictions
                #-----------------------------------------------------------------------
                x = data_scoring[predict_split]
                if make_predictions:
                    # Make predictions and probabilities for prediction set
                    y_pred, y_probs = predict_RF(rfClassifier, x)

                    #store results
                    np.savetxt(file_pred, y_pred, delimiter=" ")
                    np.savetxt(file_probs, y_probs, delimiter=" ")
                else:
                    # if predictions are already made... get stored predictions
                    y_pred = genfromtxt(f'{file_pred}', delimiter=' ')
                    y_probs = genfromtxt(f'{file_probs}', delimiter=' ')

                
                # get metrics
                #-----------------------------------------------------------------------
                path_data_model = f'{path_data}{model}/'
                params_metrics = {'classifier': classifier,
                    'group': group,
                    'y_true': data_scoring[f"y_{predict_on}"], 
                    'y_pred': y_pred, 
                    'y_probs': y_probs, 
                    #'baseline': baseline,
                    'dfScores': dfScores,
                    'image_path': image_path,
                    'data_path': path_data_model, 
                    #'suffix': suffix,
                    'filename': filename_metrics,
                    'title': title,         
                    'bestth_AUC': np.nan,
                    'bestth_PrecRecall': np.nan,
                    'CreateGraphs': True,
                    'DisplayGraphs': False,
                    'predict_on': predict_on}

                # get all classification metrics and stores results in dfScores
                results = get_performance_metrics(**params_metrics)
                dfScores = results[2]

                
                # for metrics on test set only...
                if predict_split == 'x_test':
                    # get feature importance
                    #-----------------------------------------------------------------------
                    #instantiate RandomizedSearchCV results
                    file_rscv = f'{path_model}{model}_{metric}{suffix}.rscv'
                    obj_rscv = os.path.abspath(file_rscv)
                    rscv = joblib.load(obj_rscv)
                    scoring = rscv.scoring
                    n_iter = rscv.n_iter

                    if get_featureImportance: 
                        best_params = rscv.best_estimator_.get_params()
                        feature_importances = get_feature_importances(rfClassifier, x.columns, title, filename_metrics, featureImportance_th, topFeatureCount, show=False)
                        dfFeatureImportance = save_ClassifierFeatures(dfFeatureImportance, feature_importances, topFeatureCount, classifier, filename_metrics, suffix, show=False)
                        dfFeatureImportance.sort_index(axis=1, inplace=True)          

                    #save best params
                    #-----------------------------------------------------------------------
                    if get_hyperparameters: 
                        dfHyperparameters = save_bestparamsRF(dfHyperparameters, rscv, 'RandomSearchCV', classifier, scoring, cv, n_iter)
                        dfHyperparameters.sort_index(axis=1, inplace=True)
                    
                    #save decision tree
                    #-----------------------------------------------------------------------
                    if get_decisiontree: 
                        plot_DecisionTree(rfClassifier, x.columns, title, filename_metrics, False)

    return dfScores, dfFeatureImportance, dfHyperparameters



def scale_data(data_score, data_train, file_scalar):
    # get the StandardScaler used when training the model
    scaler_std = joblib.load(file_scalar) 

    # use it to scale new data, for only the columns used during training
    # output of the scaler is a NumPy array
    data_score_scaled_arr = scaler_std.transform(data_score[data_train.columns])

    #convert array back to DataFrame & add column names
    data_score_scaled = pd.DataFrame(data_score_scaled_arr, columns=data_train.columns)
    
    return data_score_scaled



def get_model_performance(data_path, filename, datasets, models, metrics, datasets_scored, datasets_scored_suffix, title_index = 'Dataset — Model'):
    scores = []

    for dataset in datasets:
        dataset_self = datasets_scored[dataset][0]
        suffix_self = datasets_scored_suffix[dataset][0]
        
        dataset_other = datasets_scored[dataset][1]
        suffix_other = datasets_scored_suffix[dataset][1]

        metric = metrics[dataset]


        # get scores for model scoring itself
        # -----------------------------------------------------------------------------------------------------------------------------------------------
        dfScores_self = pd.read_csv(f'{data_path}{filename}{suffix_self}_All.csv', index_col=0)
        dfScores_self = dfScores_self[dfScores_self.index.str.contains('|'.join(metric)) & ~dfScores_self.index.str.contains('CV')].sort_index()
        dfScores_self.columns.name = title_index

        # rename index
        index_names = {
            'WRF F1 ': f'{dataset_self} — {dataset} WRF'
            , 'WRF F1  PR Th': f'{dataset_self} — {dataset} WRF PR Th'
            , 'WRF F1  AUC Th': f'{dataset_self} — {dataset} WRF ROC Th'
            , 'CRF F1 ': f'{dataset_self} — {dataset} CRF'
            , 'CRF F1  PR Th': f'{dataset_self} — {dataset} CRF PR Th'
            , 'CRF F1  AUC Th': f'{dataset_self} — {dataset} CRF ROC Th'
            , 'RFU F1 ': f'{dataset_self} — {dataset} RFU'
            , 'RFU F1  PR Th': f'{dataset_self} — {dataset} RFU PR Th'
            , 'RFU F1  AUC Th': f'{dataset_self} — {dataset} RFU ROC Th'
            }
        dfScores_self = dfScores_self.rename(index=index_names)
        dfScores_self.Group = f'{dataset} ' +  dfScores_self.Group
        
        scores.append(dfScores_self)


        
        # get scores for model scoring another dataset
        # -----------------------------------------------------------------------------------------------------------------------------------------------
        dfScores_other = pd.read_csv(f'{data_path}{filename}{suffix_self}_scoring{suffix_other}.csv', index_col=0)
        dfScores_other = dfScores_other[dfScores_other.index.str.contains('|'.join(metric)) & ~dfScores_other.index.str.contains('CV')].sort_index()
        dfScores_other.columns.name = title_index

        # rename index
        index_names = {
            'WRF F1 ': f'{dataset_other} — {dataset} WRF'
            , 'WRF F1  PR Th': f'{dataset_other} — {dataset} WRF PR Th'
            , 'WRF F1  AUC Th': f'{dataset_other} — {dataset} WRF ROC Th'
            , 'CRF F1 ': f'{dataset_other} — {dataset} CRF'
            , 'CRF F1  PR Th': f'{dataset_other} — {dataset} CRF PR Th'
            , 'CRF F1  AUC Th': f'{dataset_other} — {dataset} CRF ROC Th'
            , 'RFU F1 ': f'{dataset_other} — {dataset} RFU'
            , 'RFU F1  PR Th': f'{dataset_other} — {dataset} RFU PR Th'
            , 'RFU F1  AUC Th': f'{dataset_other} — {dataset} RFU ROC Th'
            }
        dfScores_other = dfScores_other.rename(index=index_names)
        dfScores_other.Group = f'{dataset} ' +  dfScores_other.Group
        
        scores.append(dfScores_other)


    return scores






# Prediction in Spatial Structures Graphs
#-----------------------------------------------------------------------------------------------------
def label_results(truth_list, pred_list):
    # adapted from:  https://github.com/PatWalters/workshop/blob/master/predictive_models/3_running_a_predictive_model.ipynb
    # label results as true positive (TP), false positive (FP), true negative (TN), and false negative (FN).
    label_list = [["TN","FN"],["FP","TP"]]
    res = [] 

    pred_list = [int(i) for i in pred_list]
    truth_list = [int(i) for i in truth_list]

    for truth, pred in zip(pred_list, truth_list):
        res.append(label_list[truth][pred])
    return res



def plot_predictions_inspatialstructure(dfResults, drType, filename, path_image, title, subtitle, label_column, removeticks=True):
    classes = ['TN', 'FP', 'FN', 'TP']
    class_colours = {0:'#EAE3D2', 1:'#F9D923', 2:'indianred', 3:'#3EC70B'}
    filename = re.sub('[\W_]+', '_', filename)

    # instantiate plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # create legend
    recs = []

    # query dataset for each data point by label, and plot
    dfLabel = dfResults.loc[dfResults[f'{label_column}'] == 'TN']
    plt.scatter(
                dfLabel[f'{drType} 1'],
                dfLabel[f'{drType} 2'],
                c=[class_colours[0]],
                s=25,
                zorder=2
                )
    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[0])) 

    dfLabel = dfResults.loc[dfResults[f'{label_column}'] == 'FP']
    plt.scatter(
                dfLabel[f'{drType} 1'],
                dfLabel[f'{drType} 2'],
                c=[class_colours[1]],
                s=25,
                zorder=2
                )
    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[1])) 
    
    dfLabel = dfResults.loc[dfResults[f'{label_column}'] == 'FN']
    plt.scatter(
                dfLabel[f'{drType} 1'],
                dfLabel[f'{drType} 2'],
                c=[class_colours[2]],
                s=25,
                zorder=2
                )
    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[2]))

    dfLabel = dfResults.loc[dfResults[f'{label_column}'] == 'TP']
    plt.scatter(
                dfLabel[f'{drType} 1'],
                dfLabel[f'{drType} 2'],
                c=[class_colours[3]],
                s=25,
                zorder=2
                )
    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[3])) 


    # remove ticks from x and y axis
    if removeticks:
        plt.setp(ax, xticks=[], yticks=[]) 

    # create legend
    plt.legend(recs, classes, loc='upper right', prop={'size': 12})

    # add title    
    plt.title(title, y=1.05, pad=10, fontsize=16, fontweight='bold', color='#363062')
    plt.suptitle(subtitle, y=.92, fontsize=14, fontweight='bold', color='#363062')

    # add footnote
    label_counts = dfResults[label_column].value_counts()
    tn = fp = fn = tp = 0
    if 'TN' in label_counts:
        tn = label_counts['TN']
    if 'FP' in label_counts:
        fp = label_counts['FP']
    if 'FN' in label_counts:
        fn = label_counts['FN']
    if 'TP' in label_counts:
        tp = label_counts['TP']
    footnote = f'TN: {tn}    FP: {fp}    FN: {fn}    TP: {tp}'

    plt.annotate(footnote,
            xy = (0.75, 0.02),
            xycoords='axes fraction',
            ha='right',
            va="center",
            fontsize=12)

    # save image, add border and display
    image = f'{path_image}{drType}Predictions_{filename}.jpg'      
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()

    # add_imageborder(image)
    display(Image(image))


    
def plot_predictions_tsne(x, SMILES, y, y_pred, y_probs, bestth_roc, bestth_PrecRecall
    , pc_num, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=None
    , removeticks=True, title='', subtitle='', filename='', path_image='', path_model='', path_data='', getSavedModel=False):

    filename = re.sub('[\W_]+', '_', filename)
    drType = 'tSNE'

    if getSavedModel:
        #retrieve saved model/results
        path = os.path.abspath(f'{path_model}{filename}.mdl')
        tsne_results = joblib.load(path)
        dfResults = pd.read_csv(f'{path_data}{filename}.csv', index_col=0)                 
    else:
        #instantiate tsne
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)
        
        #run pca
        pca = PCA(n_components=pc_num)
        pca_result = pca.fit_transform(x)

        #add smiles, activity probability and prediction (based on threshold or no threshold)
        label = y.tolist()
        predictions = y_pred.tolist()

        dfResults = SMILES
        dfResults['Activity'] = label
        dfResults['Probability'] = y_probs.tolist()
        dfResults['Prediction'] = predictions
        
        # get TP, FP, TN, FN labels
        prediction_labels = label_results(label, predictions)
        dfResults['Label'] = prediction_labels

        y_pred_thauc = (y_probs > bestth_roc).astype(bool)
        dfResults['Prediction_THAUC'] = y_pred_thauc.tolist()
        prediction_labels = label_results(label, y_pred_thauc.tolist())
        dfResults['Label_THAUC'] = prediction_labels

        y_pred_thpr = (y_probs > bestth_PrecRecall).astype(bool)
        dfResults['Prediction_THPR'] = y_pred_thpr.tolist()
        prediction_labels = label_results(label, y_pred_thpr.tolist())
        dfResults['Label_THPR'] = prediction_labels
        #Counter(prediction_labels)

        #store the pca results
        columns = []
        for x in range(1, pca_result.shape[1] + 1):
            columns.append(f'PC_{x}')                    
        
        pca_results = pd.DataFrame(pca_result,columns=columns)
        pca_results.index = dfResults.index              
        dfResults = pd.concat([dfResults, pca_results], axis = 1)

        #use pca output to seed tsne
        tsne_results = tsne.fit_transform(pca_result)

        #save model and results            
        joblib.dump(tsne_results, f'{path_model}{filename}.mdl')

        #store the tsne results
        dfResults[f'{drType} 1'] = tsne_results[:,0].tolist()
        dfResults[f'{drType} 2'] = tsne_results[:,1].tolist()
        dfResults.to_csv(f'{path_data}{filename}.csv')

    
    label_column = 'Label'
    plot_predictions_inspatialstructure(dfResults, drType, filename, path_image, title, subtitle, label_column)
    
    label_column = 'Label_THAUC'
    filename_graph = f'{filename}_THAUC'
    subtitle_graph = f'{subtitle} & AUC Th {round(bestth_roc, 2)}'
    plot_predictions_inspatialstructure(dfResults, drType, filename_graph, path_image, title, subtitle_graph, label_column)

    label_column = 'Label_THPR'
    filename_graph = f'{filename}_THPR'
    subtitle_graph = f'{subtitle} & PR Th {round(bestth_PrecRecall, 2)}'
    plot_predictions_inspatialstructure(dfResults, drType, filename_graph, path_image, title, subtitle_graph, label_column) 



def plot_model_predictions_tsne(dataset, suffix, models, metrics, thresholds_auc, thresholds_pr
    , x, x_smiles, y, pc_num, data_path, model_path, image_path 
    , predict_on='test', n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=None
    , removeticks=True, title='', subtitle='', getSavedModel=False):
 
    for model, metric in zip(models, metrics):
        path_data = f'{data_path}{model}/'
        path_model = model_path
        path_image = f'{image_path}{model}\\'

        # get stored predictions
        file_pred = f'{path_data}{model}_{metric}_y{predict_on}pred{suffix}.csv'
        file_probs = f'{path_data}{model}_{metric}_y{predict_on}probs{suffix}.csv'

        y_pred = genfromtxt(f'{file_pred}', delimiter=' ')
        y_probs = genfromtxt(f'{file_probs}', delimiter=' ')    

        if predict_on == 'cv':
            filename = f'{model}_{metric}_tsne_{predict_on}{suffix}'  
        else:
            filename = f'{model}_{metric}_tsne{suffix}' 

        th_auc = thresholds_auc[model]
        th_pr = thresholds_pr[model]
        title = f'{dataset}: {model.upper()} {metric.capitalize()}'
        
        params = {'x': x,
                'SMILES': x_smiles,
                'y': y, 
                'y_pred': y_pred, 
                'y_probs': y_probs,   
                'bestth_roc': th_auc, 
                'bestth_PrecRecall': th_pr,
                'pc_num': pc_num,
                'perplexity': perplexity,       
                'removeticks': removeticks,
                'title': title,
                'subtitle': subtitle,
                'filename': filename,                
                'path_data' : path_data,
                'path_model' : path_model,
                'path_image' : path_image,
                'getSavedModel': getSavedModel}

        # plot predictions
        plot_predictions_tsne(**params)



def plot_predictions_umap(x, SMILES, y, y_pred, y_probs, bestth_roc, bestth_PrecRecall
    , n_neighbors=15, min_dist=0.1, metrics='euclidian', removeticks=True, title='', subtitle=''
    , filename='',  path_image='', path_model='', path_data='', getSavedModel=False):

    filename = re.sub('[\W_]+', '_', filename)
    drType = 'UMAP'

    if getSavedModel:
        #retrieve saved model/results
        path = os.path.abspath(f'{path_model}{filename}.mdl')
        reducer = joblib.load(path)
        dfResults = pd.read_csv(f'{path_data}{filename}.csv', index_col=0)                
    else:
        #instantiate umap
        reducer = umap.UMAP()
        
        #run umap
        umap_data = reducer.fit_transform(x)

        #save model and results
        joblib.dump(reducer, f'{path_model}{filename}.mdl')
        np.savetxt(f'{path_data}{filename}.csv', umap_data, delimiter=" ")

        #add smiles, activitym probability and prediction (based on threshold or no threshold)
        label = y.tolist()
        predictions = y_pred.tolist()

        dfResults = SMILES
        dfResults['Activity'] = y.tolist()
        dfResults['Probability'] = y_probs.tolist()
        dfResults['Prediction'] = y_pred.tolist()
        
        # get TP, FP, TN, FN labels
        prediction_labels = label_results(label, predictions)
        dfResults['Label'] = prediction_labels

        y_pred_thauc = (y_probs > bestth_roc).astype(bool)
        dfResults['Prediction_THAUC'] = y_pred_thauc.tolist()
        prediction_labels = label_results(label, y_pred_thauc.tolist())
        dfResults['Label_THAUC'] = prediction_labels

        y_pred_thpr = (y_probs > bestth_PrecRecall).astype(bool)
        dfResults['Prediction_THPR'] = y_pred_thpr.tolist()
        prediction_labels = label_results(label, y_pred_thpr.tolist())
        dfResults['Label_THPR'] = prediction_labels
        #Counter(prediction_labels)

        #store the umap results
        dfResults[f'{drType} 1'] = umap_data[:,0].tolist()
        dfResults[f'{drType} 2'] = umap_data[:,1].tolist()
        dfResults.to_csv(f'{path_data}{filename}.csv')


    label_column = 'Label'
    plot_predictions_inspatialstructure(dfResults, drType, filename, path_image, title, subtitle, label_column)
    
    label_column = 'Label_THAUC'
    filename_graph = f'{filename}_THAUC'
    subtitle_graph = f'{subtitle} & AUC Th {round(bestth_roc, 2)}'
    plot_predictions_inspatialstructure(dfResults, drType, filename_graph, path_image, title, subtitle_graph, label_column)

    label_column = 'Label_THPR'
    filename_graph = f'{filename}_THPR'
    subtitle_graph = f'{subtitle} & PR Th {round(bestth_PrecRecall, 2)}'
    plot_predictions_inspatialstructure(dfResults, drType, filename_graph, path_image, title, subtitle_graph, label_column) 



def plot_model_predictions_umap(dataset, suffix, models, metrics, thresholds_auc, thresholds_pr
    , x, x_smiles, y, data_path, model_path, image_path 
    , n_neighbors, min_dist, metrics_umap='euclidian', predict_on='test' 
    , removeticks=True, title='', subtitle='', getSavedModel=False):
 
    for model, metric in zip(models, metrics):
        path_data = f'{data_path}{model}/'
        path_model = model_path
        path_image = f'{image_path}{model}\\'

        # get stored predictions
        file_pred = f'{path_data}{model}_{metric}_y{predict_on}pred{suffix}.csv'
        file_probs = f'{path_data}{model}_{metric}_y{predict_on}probs{suffix}.csv'

        y_pred = genfromtxt(f'{file_pred}', delimiter=' ')
        y_probs = genfromtxt(f'{file_probs}', delimiter=' ')    

        if predict_on == 'cv':
            filename = f'{model}_{metric}_umap_{predict_on}{suffix}'  
        else:
            filename = f'{model}_{metric}_umap{suffix}' 

        th_auc = thresholds_auc[model]
        th_pr = thresholds_pr[model]
        title = f'{dataset}: {model.upper()} {metric.capitalize()}'
        
        params = {'x': x,
                'SMILES': x_smiles,
                'y': y, 
                'y_pred': y_pred, 
                'y_probs': y_probs,   
                'bestth_roc': th_auc, 
                'bestth_PrecRecall': th_pr,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,    
                'metrics': metrics_umap,    
                'removeticks': removeticks,
                'title': title,
                'subtitle': subtitle,
                'filename': filename,                
                'path_data' : path_data,
                'path_model' : path_model,
                'path_image' : path_image,
                'getSavedModel': getSavedModel}

        # plot predictions
        plot_predictions_umap(**params)





# SCALING DATA
#-----------------------------------------------------------------------------------------------------
def scale_dataset(file_2Descriptors, file_Molecules, file_TrainingData, file_scalar, file_ScaledData, FirstTimeIn=True, col_NonDescriptors=['SMILES', 'Inhibition']):   
    # Dataset was split in data analysis notebook into 2 separate files:  
    # Descriptors_{suffix}.csv:  processed unscaled 2D descriptors. used for scoring.
    # Molecules{suffix}.csv:  contains MoleculeId, SMILES, SaltStripping, Cluster 
    df2D = pd.read_csv(file_2Descriptors, index_col=0)
    dfMol = pd.read_csv(file_Molecules, index_col=0)

    #get data the model was trained on, and retrieve features used to train model 
    dfTrain = pd.read_csv(f'{file_TrainingData}', index_col=0)
    dfTrain.drop(col_NonDescriptors, axis=1, inplace=True)
    training_columns = dfTrain.columns

    if FirstTimeIn == True:
        # get the StandardScaler used when training the model
        scaler_std = joblib.load(file_scalar) 

        # use it to scale new data, for only the columns used during training
        # output of the scaler is a NumPy array
        arr2DScaled = scaler_std.transform(df2D[training_columns])

        #convert array back to DataFrame & add column names
        df2DScaled = pd.DataFrame(arr2DScaled, columns=training_columns)

        #save scaled data
        df2DScaled.to_csv(file_ScaledData)
    else:
        df2DScaled = pd.read_csv(file_ScaledData, index_col=0)

    return dfMol, df2DScaled, training_columns





# UMAP
#-----------------------------------------------------------------------------------------------------
def plot_umap(df, umap_data, classes, class_colours, title='', imgfilename='', subtitle='', figsize1=7, figsize2=7, n_components=2, replaceBackslash=True):
    # create scatter plot
    fig, ax = plt.subplots(1, figsize=(figsize1, figsize2))
    recs = []

    if n_components == 2:
        plt.gca().set_aspect('equal', 'datalim')
        plt.scatter(
                umap_data[:, 0],
                umap_data[:, 1],
                c=[x for x in df.Activity.map({1:class_colours[1], 0:class_colours[0]})],
                s=25,
                zorder=2
                )  
        plt.setp(ax, xticks=[], yticks=[])  
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            umap_data[:,0], 
            umap_data[:,1], 
            umap_data[:,2], 
            c=[x for x in df.Activity.map({1:class_colours[1], 0:class_colours[0]})], 
            s=25,
            zorder=2)


    #create legend
    for i in range(0, len(class_colours)):
        recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    
    plt.legend(recs, classes, loc='upper right', prop={'size': 10})
    

    if len(subtitle) == 0:
        plt.title(title, fontsize=13, y=1.0, pad=20, fontweight='bold', color='#363062')
    else:
        plt.title(title, y=1.05, pad=10, fontsize=13, fontweight='bold', color='#363062')
        plt.suptitle(subtitle, y=.92, fontsize=11, fontweight='bold', color='#363062')

    #save image, add border and display
    if len(imgfilename) == 0:
        imgfilename = re.sub('[\W_]+', '_', title)
    
    if replaceBackslash:
        imgfilename = re.sub('[\W_]+', '_', imgfilename)
    
    
    image = f'{imgfilename}_umap.jpg'     
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()

    #add_imageborder(image)
    display(Image(image))





# TSNE
#-----------------------------------------------------------------------------------------------------
def plot_tsne(dfFeatures, pca=None, dfTSNEResults=None, classes=['Inactive', 'Active'], class_colours={0:'#8DA0CB', 1:'#FFD92F'}
    , n_components=2, perplexity=30, learning_rate=200, n_iter=1000, verbose=0, random_state=None
    , removeticks=True, title='', subtitle='', save_model=True, filename='', path_image='', path_model='', path_data=''
    , getSavedModel=False, figsize1=7, figsize2=7, replaceBackslash=True):

    if replaceBackslash:
        filename = re.sub('[\W_]+', '_', filename)

    if getSavedModel:
        #retrieve saved model/results
        path = os.path.abspath(f'{path_model}{filename}_tSNE.mdl')
        tsne_results = joblib.load(path)                
    else:
        #instantiate tsne
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, verbose=verbose, random_state=random_state)
        
        #if pca output not passed, used training dataset
        if (pca is None):
            tsne_results = tsne.fit_transform(dfFeatures)
        else:
            #if df to store results is passed in, store the pca results
            if not (dfTSNEResults is None):
                columns = []
                for x in range(1, pca_result.shape[1] + 1):
                    columns.append(f'PC_{x}')                    
                
                pca_results = pd.DataFrame(pca,columns=columns)                  
                dfTSNEResults = pd.concat([dfTSNEResults, pca_results], axis = 1)

            #use to pca output to seed tsne
            tsne_results = tsne.fit_transform(pca)

        if save_model:
            #save model and results            
            joblib.dump(tsne_results, f'{path_model}{filename}_TSNE.mdl')


    #if df to store results is passed in, store the tsne results
    if not (dfTSNEResults is None):
        dfTSNEResults['tSNE 1'] = pd.Series(tsne_results[:,0])
        dfTSNEResults['tSNE 2'] = pd.Series(tsne_results[:,1])
        dfTSNEResults.to_csv(f'{path_data}{filename}_tSNE.csv')

    #add results to df
    dfFeatures['tSNE 1'] = tsne_results[:,0]
    dfFeatures['tSNE 2'] = tsne_results[:,1]

    fig, ax = plt.subplots(1, figsize=(figsize1, figsize2))

    plt.scatter(
            dfFeatures['tSNE 1'],
            dfFeatures['tSNE 2'],
            c=[x for x in dfFeatures.Activity.map({1:class_colours[1], 0:class_colours[0]})],
            s=25,
            zorder=2
            )  

    #create legend
    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[i]))

    #remove ticks from x and y axis
    if removeticks and (n_components == 2):
        plt.setp(ax, xticks=[], yticks=[]) 

    #create legend
    plt.legend(recs, classes, loc='upper right', prop={'size': 12})

    # add title    
    if len(subtitle) == 0:
        plt.title(title, fontsize=20, y=1.0, pad=13, fontweight='bold', color='#363062')
    else:
        plt.title(title, y=1.05, pad=10, fontsize=13, fontweight='bold', color='#363062')
        plt.suptitle(subtitle, y=.92, fontsize=11, fontweight='bold', color='#363062')

    #save image, add border and display
    image = f'{filename}_tSNE.jpg'      
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()

    #add_imageborder(image)
    display(Image(image))





# SCORE PRODUCTION DATA
#-----------------------------------------------------------------------------------------------------
def score_dataset(file_model, df2DScaled, file_pred, file_probs, FirstTimeIn=True):   
    if FirstTimeIn == True:
        #instantiate trained model
        model = os.path.abspath(file_model)
        rfClassifier = joblib.load(model)

        #make prediction/probabilities
        y_pred = rfClassifier.predict(df2DScaled)
        y_probs = rfClassifier.predict_proba(df2DScaled)[:, 1]

        #store predictions
        np.savetxt(file_pred, y_pred, delimiter=" ")
        np.savetxt(file_probs, y_probs, delimiter=" ")
    else:
        rfClassifier, y_pred, y_probs = get_savedmodel(file_model, file_pred, file_probs)

    return rfClassifier, y_pred, y_probs



def predict_dataset(dfMol, dfActive, y_probs, y_pred, bestth_roc, bestth_PrecRecal, filename, index_name):   
    dfMolScores = dfMol[['MoleculeId', 'SMILES', 'molStripped', 'Cluster']].copy()
    dfMolScores['Prob'] = y_probs.tolist()
    dfMolScores['Pred_NoTH'] = y_pred.astype(int).tolist()
    dfMolScores['Pred_ROCTH'] = (dfMolScores["Prob"] >= bestth_roc).astype(int)
    dfMolScores['Pred_PRTH'] = (dfMolScores["Prob"] >= bestth_PrecRecal).astype(int)
    dfMolScores.to_csv(filename)

    active_noth = len(dfMolScores.loc[(dfMolScores['Pred_NoTH'] == 1)])
    active_rocth = len(dfMolScores.loc[(dfMolScores['Pred_ROCTH'] == 1)])
    active_prth = len(dfMolScores.loc[(dfMolScores['Pred_PRTH'] == 1)])
    active_all = len(dfMolScores.loc[(dfMolScores['Pred_NoTH'] == 1) & (dfMolScores['Pred_ROCTH'] == 1) & (dfMolScores['Pred_PRTH'] == 1)])

    list = [active_noth, active_rocth, active_prth, active_all]
    dfActive.loc[index_name] = list

    return dfMolScores, dfActive



def predict_dataset_umap(df2DScaled, dfMolScores, thresholds, n_components, filename_model, filename_data, filename_image, title='', classes=['Inactive', 'Active'], class_colours={0:'#8DA0CB', 1:'#FFD92F'}, getSavedModel=True):   
    if getSavedModel:
        #retrieve saved model/results
        path = os.path.abspath(filename_model)
        reducer = joblib.load(path)
        umap_data = genfromtxt(filename_data)
    else:
        #first time in, save model and results
        reducer = umap.UMAP(n_components=n_components)
        umap_data = reducer.fit_transform(df2DScaled)    
        joblib.dump(reducer, filename_model)
        np.savetxt(filename_data, umap_data, delimiter=" ")


    dfPredictions = df2DScaled.copy()
    for type, value in thresholds.items():
        if type == 'default':
            #append label to training data for graphs
            dfPredictions['Activity'] = dfMolScores.Pred_NoTH.to_list()
            filename = f'{filename_image}_NoTH'
            subtitle = f'Default Threshold:  {value}'

        if type == 'roc':
            #append label to training data for graphs
            dfPredictions['Activity'] = dfMolScores.Pred_ROCTH.to_list()
            filename = f'{filename_image}_ROCTH'
            subtitle = f'ROC Threshold:  {value}'

        if type == 'pr':
            #append label to training data for graphs
            dfPredictions['Activity'] = dfMolScores.Pred_PRTH.to_list()
            filename = f'{filename_image}_PRTH'
            subtitle = f'Precision Recall Threshold:  {value}'

        plot_umap(df=dfPredictions, umap_data=umap_data, classes=classes, class_colours=class_colours, title=title, imgfilename=filename, subtitle=subtitle, n_components=n_components, replaceBackslash=False)



def predict_dataset_tsne(df2DScaled, dfMolScores, thresholds, pc_num, filename_image, RandomState, title='', classes=['Inactive', 'Active'], class_colours={0:'#8DA0CB', 1:'#FFD92F'}, getSavedModel=True):      
    dfPredictions = df2DScaled.copy()
    for type, value in thresholds.items():
        if type == 'default':
            #append label to training data for graphs
            dfPredictions['Activity'] = dfMolScores.Pred_NoTH.to_list()
            filename = f'{filename_image}_NoTH'
            subtitle = f'Default Threshold: {value}'

        if type == 'roc':
            #append label to training data for graphs
            dfPredictions['Activity'] = dfMolScores.Pred_ROCTH.to_list()
            filename = f'{filename_image}_ROCTH'
            subtitle = f'ROC Threshold:  {value}'

        if type == 'pr':
            #append label to training data for graphs
            dfPredictions['Activity'] = dfMolScores.Pred_PRTH.to_list()
            filename = f'{filename_image}_PRTH'
            subtitle = f'Precision Recall Threshold:  {value}'

        pca = PCA(n_components=pc_num)
        pca_result = pca.fit_transform(df2DScaled)    
    
        #set params valuable to send to function
        params = {'dfFeatures': dfPredictions,
            'pca': pca_result, 
            'classes': classes, 
            'class_colours': class_colours, 
            'getSavedModel': getSavedModel,
            'title': title,
            'subtitle': subtitle,
            'filename': filename,
            'random_state': RandomState,
            'replaceBackslash': False}

        # plot tsne
        plot_tsne(**params)