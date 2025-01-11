#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# ## Variance Threshold 

# In[7]:


def variance_threshold_selector(x, threshold = 0.2):
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold)
    selector.fit(x)
    return x[x.columns[selector.get_support(indices=True)]] 


# ## Z-Score Normalization

# In[3]:


def z_score (x):
    import pandas as pd
    import numpy as np 
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    scaler = preprocessing.StandardScaler().fit(x)
    X = scaler.transform(x) 
    x1 = pd.DataFrame(X) 
    x1.columns = x.columns
    return x1


# # Decision Trees 

# ## ROC Curve

# In[1]:


def roc_curve (model, xpredict, ytrue):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import numpy as np
    
    probs = model.predict_proba(xpredict)
    preds = probs[:,1]
    
    fpr, tpr, threshold = metrics.roc_curve(ytrue, preds)
    roc_auc = metrics.auc(fpr, tpr)
    J = tpr-fpr 
    ix =np.argmax(J) 
    best_thresh = threshold[ix]
    
    legend_best = f'TH: {round(best_thresh, 3)}'
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')

    plt.plot([0,1],[0,1], linestyle=':', color='#EB5353')
    plt.plot(fpr, tpr, label = 'AUC: %0.2f'% roc_auc, color = '#0F2C67')
    plt.legend(loc = 'lower right')

    plt.ylabel('True Positive Rate', fontsize=14, labelpad=20)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=20)
    plt.rc('legend',fontsize='13')

    th_suffix = 'TH'
    title = f'{"ROC Curve"}'
    plt.title(title, fontsize=16, y=1.05, pad=10, fontweight='bold', color='#363062') 


# ## Precision-Recall Curve

# In[2]:


def prec_recall_curve (model, xpredict, ytrue):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    
    probs = model.predict_proba(xpredict)
    predictions = probs[:,1]
    
    precision, recall, thresholds = precision_recall_curve(ytrue, predictions)
    
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')
    
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


# ## Confusion Matrix

# In[3]:


def confusion_matrix(model, xconf_mat, yconf_mat, threshold):
    
    import seaborn as sns
    from sklearn import metrics
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    y_pred = (model.predict_proba(xconf_mat)[:,1] >= threshold).astype(bool)
        
    matrix = confusion_matrix(yconf_mat, y_pred)
    tn, fp, fn, tp = matrix.ravel()

    fig = plt.figure()
    fig.patch.set_facecolor('#FFFFFF')
    cfm_plot = sns.heatmap(matrix, cmap='Blues', annot=True, fmt='.0f', annot_kws = {'size':16},
                           xticklabels = ["Inactive", "Active"] , yticklabels = ["Inactive", "Active"], 
                           facecolor='white')
    cfm_plot.set_xlabel('Predicted',fontsize=16, labelpad=28, color='#4D4C7D')
    cfm_plot.set_ylabel('True',fontsize=16, labelpad=28, color='#4D4C7D')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt.show()


# ## Metrics 

# In[4]:


def classification_metrics (model, cv, ycv, et, yet, best_threshold):
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve   
    
    y_predcv = (model.predict_proba(cv)[:,1] >= best_threshold)
    y_predcv = y_predcv.astype(int)    
    
    y_predet = (model.predict_proba(et)[:,1] >= best_threshold)
    y_predet = y_predet.astype(int)  
    
    roc_auc_et=roc_auc_score(yet, y_predet) 
    roc_auc_cv=roc_auc_score(ycv, y_predcv) 
    
    precision_cv, recall_cv, thresholds_cv = precision_recall_curve(ycv, y_predcv)
    auc_precision_recall_cv = auc(recall_cv, precision_cv)
    
    precision_et, recall_et, thresholds_et = precision_recall_curve(yet, y_predet)
    auc_precision_recall_et = auc(recall_et, precision_et)
    
    sklearn.metrics.confusion_matrix(ycv, y_predcv)
    tn, fp, fn, tp = confusion_matrix(ycv, y_predcv).ravel()
    specificitycv = tn / (tn+fp) 
    sensitivitycv = tp / (tp+fn) 
    balancedaccuracycv = (sensitivitycv + specificitycv) / 2 
    mcc_cv=matthews_corrcoef(ycv, y_predcv)  

    sklearn.metrics.confusion_matrix(yet, y_predet)
    Tn, Fp, Fn, Tp = confusion_matrix(yet, y_predet).ravel()
    specificityet = Tn / (Tn+Fp)
    sensitivityet = Tp / (Tp+Fn)
    balancedaccuracyet = (specificityet + sensitivityet) / 2
    mcc_et=matthews_corrcoef(yet, y_predet)
    
    p_cv = precision_score(ycv, y_predcv)
    p_et = precision_score(yet, y_predet)
    r_cv = recall_score(ycv, y_predcv)
    r_et = recall_score(yet, y_predet)
    
    f1_cv = f1_score(ycv, y_predcv)
    f1_et = f1_score(yet, y_predet)
      
    df = {'CV Set': [best_threshold, roc_auc_cv,auc_precision_recall_cv, balancedaccuracycv, sensitivitycv, specificitycv,mcc_cv, p_cv, r_cv, f1_cv], 
         'Test Set':[best_threshold, roc_auc_et, auc_precision_recall_et,balancedaccuracyet, sensitivityet, specificityet, mcc_et, p_et, r_et, f1_et]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df


# ## Metrics 80/20 

# In[5]:


def classification_metrics80_20 (model, train, ytrain, test, ytest, best_threshold):
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    
    y_predtrain = (model.predict_proba(train)[:,1] >= best_threshold)
    y_predtrain = y_predtrain.astype(int)    
    
    y_predtest = (model.predict_proba(test)[:,1] >= best_threshold)
    y_predtest = y_predtest.astype(int)  
    
    roc_auc_test=roc_auc_score(ytest, y_predtest) 
    roc_auc_train=roc_auc_score(ytrain, y_predtrain) 
    
    precision_train, recall_train, thresholds_train = precision_recall_curve(ytrain, y_predtrain)
    auc_precision_recall_train = auc(recall_train, precision_train)
    
    precision_test, recall_test, thresholds_test = precision_recall_curve(ytest, y_predtest)
    auc_precision_recall_test = auc(recall_test, precision_test)
    
    sklearn.metrics.confusion_matrix(ytrain, y_predtrain)
    tn, fp, fn, tp = confusion_matrix(ytrain, y_predtrain).ravel()
    specificitytrain = tn / (tn+fp) 
    sensitivitytrain = tp / (tp+fn) 
    balancedaccuracytrain = (sensitivitytrain + specificitytrain) / 2 
    mcc_train=matthews_corrcoef(ytrain, y_predtrain)  

    sklearn.metrics.confusion_matrix(ytest, y_predtest)
    Tn, Fp, Fn, Tp = confusion_matrix(ytest, y_predtest).ravel()
    specificitytest = Tn / (Tn+Fp)
    sensitivitytest = Tp / (Tp+Fn)
    balancedaccuracytest = (specificitytest + sensitivitytest) / 2
    mcc_test=matthews_corrcoef(ytest, y_predtest)
    
    p_train = precision_score(ytrain, y_predtrain)
    p_test = precision_score(ytest, y_predtest)
    r_train = recall_score(ytrain, y_predtrain)
    r_test = recall_score(ytest, y_predtest)
    
    f1_train = f1_score(ytrain, y_predtrain)
    f1_test = f1_score(ytest, y_predtest)
    
    df = {'Train Set': [best_threshold, roc_auc_train,auc_precision_recall_train, balancedaccuracytrain, sensitivitytrain, specificitytrain,mcc_train, p_train, r_train, f1_train], 
         'Test Set':[best_threshold, roc_auc_test, auc_precision_recall_test,balancedaccuracytest, sensitivitytest, specificitytest, mcc_test, p_test, r_test, f1_test]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df


# ## Regression Metrics

# In[6]:


def regression_metrics (model, xcv, ycv, xet, yet):
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    y_pred1cv= model.predict(xcv)
    y_pred1et= model.predict(xet)
    mse_et1 = mean_squared_error(yet, y_pred1et)
    mse_cv1 = mean_squared_error(ycv, y_pred1cv)

    from sklearn.metrics import r2_score
    rcv1 = r2_score(ycv, y_pred1cv)
    ret1 = r2_score(yet, y_pred1et)

    rmse_cv1 = mse_cv1**0.5
    rmse_et1 = mse_et1**0.5

    data = {r'$r^{2}$': [rcv1, ret1], 'RMSE':[rmse_cv1, rmse_et1]}
    df = pd.DataFrame(data, index=['Cross Validation', 'Test'])
    return df


# ## Multi-class metrics 

# In[6]:


def multiclass_metrics(model, x1_cv, y_cv, x1_et, y_et):
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.preprocessing import label_binarize
    
    y_predcv_proba = model.predict_proba(x1_cv)
    y_predet_proba = model.predict_proba(x1_et)
    y_predcv = model.predict(x1_cv)
    y_predet = model.predict(x1_et)

    y_cv = np.array(y_cv)
    y_et = np.array(y_et)

    balancedaccuracycv = balanced_accuracy_score(y_cv, y_predcv)
    balancedaccuracyet = balanced_accuracy_score(y_et, y_predet)
    acc_cv = accuracy_score(y_cv, y_predcv)
    acc_et = accuracy_score(y_et, y_predet)
    precision_cv, recall_cv, f1_score_cv, _= precision_recall_fscore_support(y_cv, y_predcv)
    precision_et, recall_et, f1_score_et, _= precision_recall_fscore_support(y_et, y_predet)

    p_et = sum(precision_et)/len(precision_et)
    p_cv = sum(precision_cv)/len(precision_cv)
    r_et = sum(recall_et)/len(recall_et)
    r_cv = sum(recall_cv)/len(recall_cv)
    f1_et = sum(f1_score_et)/len(f1_score_et)
    f1_cv = sum(f1_score_cv)/len(f1_score_cv)

    mcc_cv=matthews_corrcoef(y_cv, y_predcv)
    mcc_et=matthews_corrcoef(y_et, y_predet) 
    
    y_cv = label_binarize(y_cv, classes=np.unique(y_train))
    y_et = label_binarize(y_et, classes=np.unique(y_train))

    roc_auc_cv = roc_auc_score(y_cv, y_predcv_proba, average='macro')
    roc_auc_et = roc_auc_score(y_et, y_predet_proba, average='macro')

    df = {'CV Set': [roc_auc_cv, balancedaccuracycv, acc_cv, mcc_cv, p_cv, r_cv, f1_cv], 
         'Test Set':[roc_auc_et,balancedaccuracyet, acc_et, mcc_et, p_et, r_et, f1_et]}
    df = pd.DataFrame(df, index = ['ROC AUC','Balanced Accuracy','Accuracy', 'MCC','Precision','Recall', 'F1 Score'])
    return df


# ## Thresholds 

# In[7]:


def optimized_threshold(model, xpredict, ytrue, threshold_type):
    import numpy as np
    import ghostml
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics 
    from sklearn.metrics import f1_score 
    
    preds = model.predict_proba(xpredict)
    probs = preds[:, 1]
    
    if threshold_type == 'f_score':
        precision, recall, thresholds = precision_recall_curve(ytrue, probs)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
    if threshold_type == 'youden':
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, probs)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    if threshold_type == 'range':
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        thresholds = np.arange(0, 1, 0.001)
        scores = [f1_score(ytrue, to_labels(probs, t)) for t in thresholds]
        ix = np.argmax(scores)
        best_thresh = thresholds[ix]
    if threshold_type == 'ghost kappa':
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ytrue, probs, thresholds, ThOpt_metrics = 'Kappa', random_seed=1) 
    if threshold_type == 'ghost roc':
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ytrue, probs, thresholds, ThOpt_metrics = 'ROC', random_seed=1) 
        
    return best_thresh


# ## KFold Thresholds

# In[1]:


def kfold_optimized_threshold (model, xtrain, ytrain, threshold_type, skf):
    import numpy as np
    import ghostml
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics 
    from sklearn.metrics import f1_score
    
    best_thresholds = []
    
    for train_index, cv_index in skf.split(xtrain, ytrain):
        x_train, x_cv = xtrain.iloc[train_index], xtrain.iloc[cv_index]
        y_train, y_cv = ytrain.iloc[train_index], ytrain.iloc[cv_index]
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_cv)[:,1]
        
        if threshold_type == 'f_score':
            precision, recall, thresholds = precision_recall_curve(y_cv, probs)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscore)
            best_thresh = thresholds[ix]
        if threshold_type == 'youden':
            fpr, tpr, thresholds = metrics.roc_curve(y_cv, probs)
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]
        if threshold_type == 'range':
            def to_labels(pos_probs, threshold):
                return (pos_probs >= threshold).astype('int')
            thresholds = np.arange(0, 1, 0.001)
            scores = [f1_score(y_cv, to_labels(probs, t)) for t in thresholds]
            ix = np.argmax(scores)
            best_thresh = thresholds[ix]
        if threshold_type == 'ghost kappa':
            thresholds = np.arange(0, 1, 0.001)
            best_thresh = ghostml.optimize_threshold_from_predictions(y_cv, probs, thresholds, ThOpt_metrics = 'Kappa', random_seed=1) 
        if threshold_type == 'ghost roc':
            thresholds = np.arange(0, 1, 0.001)
            best_thresh = ghostml.optimize_threshold_from_predictions(y_cv, probs, thresholds, ThOpt_metrics = 'ROC', random_seed=1) 
        
        best_thresholds.append(best_thresh)
    
    return np.mean(best_thresholds)


# # KFold Metrics

# In[2]:


def kfold_metrics (model, xtrain, ytrain, xtest, ytest, best_threshold, skf):
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve 
    
    roc_auc_cv_scores = []
    pr_auc_cv_scores = []
    sensitivity_cv_scores = []
    specificity_cv_scores = []
    balanced_accuracy_cv_scores = []
    mcc_cv_scores = []
    precision_cv_scores = []
    recall_cv_scores = []
    f1_cv_scores = []
    
    for train_index, cv_index in skf.split(xtrain, ytrain):
        x_train, x_cv = xtrain.iloc[train_index], xtrain.iloc[cv_index]
        y_train, y_cv = ytrain.iloc[train_index], ytrain.iloc[cv_index]
        model.fit(x_train, y_train)
        y_predcv = model.predict_proba(x_cv)[:,1] >= best_threshold
        y_predcv = y_predcv.astype(int)
        
        roc_auc_cv = roc_auc_score(y_cv, y_predcv)
        roc_auc_cv_scores.append(roc_auc_cv)
        
        precision_cv, recall_cv, thresholds_cv = precision_recall_curve(y_cv, y_predcv)
        pr_auc_cv = auc(recall_cv, precision_cv)
        pr_auc_cv_scores.append(pr_auc_cv)
        
        sklearn.metrics.confusion_matrix(y_cv, y_predcv)
        tn, fp, fn, tp = confusion_matrix(y_cv, y_predcv).ravel()
        specificitycv = tn / (tn+fp) 
        specificity_cv_scores.append(specificitycv)
        
        sensitivitycv = tp / (tp+fn) 
        sensitivity_cv_scores.append(sensitivitycv)
        
        balancedaccuracycv = (sensitivitycv + specificitycv) / 2 
        balanced_accuracy_cv_scores.append(balancedaccuracycv)
        
        mcc_cv=matthews_corrcoef(y_cv, y_predcv)
        mcc_cv_scores.append(mcc_cv)
        
        p_cv = precision_score(y_cv, y_predcv)
        precision_cv_scores.append(p_cv)
        
        r_cv = recall_score(y_cv, y_predcv)
        recall_cv_scores.append(r_cv)
        
        f1_cv = f1_score(y_cv, y_predcv)
        f1_cv_scores.append(f1_cv)
        
    roc_auc_cv = np.mean(roc_auc_cv_scores)
    auc_precision_recall_cv = np.mean(pr_auc_cv_scores)
    balancedaccuracycv = np.mean(balanced_accuracy_cv_scores)
    sensitivitycv = np.mean(sensitivity_cv_scores)
    specificitycv = np.mean(specificity_cv_scores)
    mcc_cv = np.mean(mcc_cv_scores)
    p_cv = np.mean(precision_cv_scores)
    r_cv = np.mean(recall_cv_scores)
    f1_cv = np.mean(f1_cv_scores)
    
    y_predet = (model.predict_proba(xtest)[:,1] >= best_threshold)
    y_predet = y_predet.astype(int)  
    
    roc_auc_et=roc_auc_score(ytest, y_predet) 
    precision_et, recall_et, thresholds_et = precision_recall_curve(ytest, y_predet)
    auc_precision_recall_et = auc(recall_et, precision_et)
    
    sklearn.metrics.confusion_matrix(ytest, y_predet)
    Tn, Fp, Fn, Tp = confusion_matrix(ytest, y_predet).ravel()
    specificityet = Tn / (Tn+Fp)
    sensitivityet = Tp / (Tp+Fn)
    balancedaccuracyet = (specificityet + sensitivityet) / 2
    mcc_et=matthews_corrcoef(ytest, y_predet)
    
    p_et = precision_score(ytest, y_predet)
    r_et = recall_score(ytest, y_predet)
    f1_et = f1_score(ytest, y_predet)
    
    df = {'CV Set': [best_threshold, roc_auc_cv,auc_precision_recall_cv, balancedaccuracycv, sensitivitycv, specificitycv,mcc_cv, p_cv, r_cv, f1_cv], 
         'Test Set':[best_threshold, roc_auc_et, auc_precision_recall_et,balancedaccuracyet, sensitivityet, specificityet, mcc_et, p_et, r_et, f1_et]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df   


# In[ ]:





# In[ ]:





# # AutoML

# ## Threshold 

# In[5]:


def h2o_optimized_threshold(model, xcv, ycv, threshold_type):
    import numpy as np
    predictions = model.predict(xcv).as_data_frame().values
    probs = predictions[:,2]
    ycv = ycv.as_data_frame()
    if threshold_type == 'f_score':
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(ycv, probs)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
    if threshold_type == 'youden':
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(ycv, probs)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    if threshold_type == 'range':
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        from sklearn.metrics import f1_score
        thresholds = np.arange(0, 1, 0.001)
        scores = [f1_score(ycv, to_labels(probs, t)) for t in thresholds]
        ix = np.argmax(scores)
        best_thresh = thresholds[ix]
    return best_thresh


# ## Metrics

# In[6]:


def h2o_metrics_60_40_own (model, train, ytrain, et, yet, cv, ycv, best_threshold):
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    
    y_predcv = (model.predict(cv)[:,2] >= best_threshold).as_data_frame().values   
    y_predet = (model.predict(et)[:,2] >= best_threshold).as_data_frame().values
    
    ycv = ycv.as_data_frame()
    yet = yet.as_data_frame()

    roc_auc_et=roc_auc_score(yet, y_predet) 
    roc_auc_cv=roc_auc_score(ycv, y_predcv) 
    
    precision_cv, recall_cv, thresholds_cv = precision_recall_curve(ycv, y_predcv)
    auc_precision_recall_cv = auc(recall_cv, precision_cv)
    
    precision_et, recall_et, thresholds_et = precision_recall_curve(yet, y_predet)
    auc_precision_recall_et = auc(recall_et, precision_et)
    
    sklearn.metrics.confusion_matrix(ycv, y_predcv)
    tn, fp, fn, tp = confusion_matrix(ycv, y_predcv).ravel()
    specificitycv = tn / (tn+fp) 
    sensitivitycv = tp / (tp+fn) 
    balancedaccuracycv = (sensitivitycv + specificitycv) / 2 
    mcc_cv=matthews_corrcoef(ycv, y_predcv)  

    sklearn.metrics.confusion_matrix(yet, y_predet)
    Tn, Fp, Fn, Tp = confusion_matrix(yet, y_predet).ravel()
    specificityet = Tn / (Tn+Fp)
    sensitivityet = Tp / (Tp+Fn)
    balancedaccuracyet = (specificityet + sensitivityet) / 2
    mcc_et=matthews_corrcoef(yet, y_predet)
    
    p_cv = precision_score(ycv, y_predcv)
    p_et = precision_score(yet, y_predet)
    r_cv = recall_score(ycv, y_predcv)
    r_et = recall_score(yet, y_predet)
    
    p_r_m_cv = p_cv*r_cv
    p_r_a_cv = p_cv+r_cv
    
    p_r_m_et = p_et*r_et
    p_r_a_et = p_et+r_et
    
    f1_cv = 2*(p_r_m_cv/p_r_a_cv)
    f1_et = 2*(p_r_m_et/p_r_a_et)
    
    roc_av = (roc_auc_cv+roc_auc_et)/2
    ba_av = (balancedaccuracycv+balancedaccuracyet)/2
    sens_av = (sensitivitycv+sensitivityet)/2
    spec_av = (specificitycv+specificityet)/2
    mcc_av = (mcc_cv+mcc_et)/2
    pr_av = (p_cv+p_et)/2
    r_av = (r_cv+r_et)/2
    f_av = (f1_cv+f1_et)/2
    pr_auc_av = (auc_precision_recall_cv+auc_precision_recall_et)/2
    
    df = {'CV Set': [best_threshold, roc_auc_cv,auc_precision_recall_cv, balancedaccuracycv, sensitivitycv, specificitycv,mcc_cv, p_cv, r_cv, f1_cv], 
         'Test Set':[best_threshold, roc_auc_et, auc_precision_recall_et,balancedaccuracyet, sensitivityet, specificityet, mcc_et, p_et, r_et, f1_et],
         'Average': [best_threshold, roc_av, pr_auc_av, ba_av, sens_av, spec_av, mcc_av, pr_av, r_av, f_av]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df


# ## Regression Metrics

# In[21]:


def h2o_reg_metrics (model, test, ytest, train, ytrain):
    import pandas as pd
    import h2o
    from h2o.automl import H2OAutoML
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    
    ytrain=ytrain.as_data_frame() 
    ytrain=ytrain.astype(np.int64)

    prediction = model.predict(train)
    prediction = prediction.as_data_frame() 
    prediction=prediction.astype(np.int64)
    
    ytest=ytest.as_data_frame() 
    ytest=ytest.astype(np.int64)

    prediction1 = model.predict(test)
    prediction1 = prediction1.as_data_frame() 
    prediction1=prediction1.astype(np.int64)
   
    mse_train = mean_squared_error(ytrain, prediction)
    mse_test = mean_squared_error(ytest, prediction1)
    
    rmse_train = mean_squared_error(ytrain, prediction, squared=False)
    rmse_test = mean_squared_error(ytest, prediction1,squared=False)
    
    r2_train = r2_score(ytrain, prediction)
    r2_test = r2_score(ytest, prediction1)
    
    df = {'Training Set': [mse_train, rmse_train, r2_train], 
         'Test Set':[mse_test, rmse_test, r2_test]}
    df = pd.DataFrame(df, index = ['MSE', 'RMSE', r'$r^{2}$'])

    return df


# ## ROC Curve

# In[42]:


def h2o_roc (model, y, x):
    import h2o
    from h2o.automl import H2OAutoML
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import numpy as np
    
    y = y.as_data_frame()
    prediction = model.predict(x).as_data_frame().values
    preds = prediction[:,2]
    fpr, tpr, thresholds = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)
    J = tpr-fpr 
    ix =np.argmax(J) 
    best_thresh = thresholds[ix]
    legend_best = f'TH: {round(best_thresh, 3)}'
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')

    plt.plot([0,1],[0,1], linestyle=':', color='#EB5353')
    plt.legend(loc = 'lower right')
    plt.plot(fpr, tpr, label = 'AUC: %0.2f'% roc_auc, color = '#0F2C67')
    plt.legend(loc = 'lower right')

    plt.ylabel('True Positive Rate', fontsize=14, labelpad=20)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=20)
    plt.rc('legend',fontsize='13')

    th_suffix = 'TH'
    title = f'{"ROC Curve"}'
    plt.title(title, fontsize=16, y=1.05, pad=10, fontweight='bold', color='#363062') 


# ## Confusion Matrix

# In[10]:


def h2o_conf_mat_thresh(model,xconf_mat, yconf_mat, threshold):
    import h2o
    import numpy as np
    import seaborn as sns
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from h2o.automl import H2OAutoML
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    
    y_pred = (model.predict(xconf_mat)[:,2] >= threshold).as_data_frame().values
    yconf_mat = yconf_mat.as_data_frame()
    
    matrix = confusion_matrix(yconf_mat, y_pred)
    tn, fp, fn, tp = matrix.ravel()

    fig = plt.figure()
    fig.patch.set_facecolor('#FFFFFF')
    cfm_plot = sns.heatmap(matrix, cmap='Blues', annot=True, fmt='.0f', annot_kws = {'size':16},
                           xticklabels = ["Inactive", "Active"] , yticklabels = ["Inactive", "Active"], 
                           facecolor='white')
    cfm_plot.set_xlabel('Predicted',fontsize=16, labelpad=28, color='#4D4C7D')
    cfm_plot.set_ylabel('True',fontsize=16, labelpad=28, color='#4D4C7D')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt.show()


# ## Scoring Datasets

# In[2]:


def h2o_scoring (model, et_test_set, xtrain, ytrain):
    import pandas as pd
    import h2o
    from h2o.automl import H2OAutoML
    from sklearn import metrics
    import numpy as np
    
    ytrain = ytrain.as_data_frame()
    prediction = model.predict(xtrain).as_data_frame().values
    preds = prediction[:,2]
    fpr, tpr, thresholds = metrics.roc_curve(ytrain, preds)
    roc_auc = metrics.auc(fpr, tpr)
    J = tpr-fpr 
    ix =np.argmax(J) 
    best_thresh = thresholds[ix]
    
    et_test_set = h2o.H2OFrame(et_test_set)
    probabilities = model.predict(et_test_set)
    prob_1 = probabilities[:,2]    
    prob_1 = prob_1.as_data_frame()
    scores = (prob_1 > best_thresh).astype(bool)
    scores = pd.DataFrame(scores) #making it a dataframe
    scores.columns = ['score']
    prob_1 = pd.DataFrame (prob_1)
    prob_1.columns = ['probability']
    df = pd.concat([scores, prob_1], axis=1)
    df["score"]=df["score"].astype(int)
    return df


# # TensorFlow

# ## ROC Curve 

# In[33]:


def keras_roc (model, xtrain, ytrain, xcv, ycv, proba = True):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import numpy as np
    
    if proba == True:
        probs = model.predict_proba(xcv)
        preds = probs[:, 1]
    else:
        probs = model.predict(xcv)
        preds = probs[:, 0]
        
    fpr, tpr, threshold = metrics.roc_curve(ycv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    J = tpr-fpr 
    ix =np.argmax(J) 
    best_thresh = threshold[ix]
    
    legend_best = f'TH: {round(best_thresh, 3)}'
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')

    plt.plot([0,1],[0,1], linestyle=':', color='#EB5353')
    plt.plot(fpr, tpr, label = 'AUC: %0.2f'% roc_auc, color = '#0F2C67')
    plt.legend(loc = 'lower right')

    plt.ylabel('True Positive Rate', fontsize=14, labelpad=20)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=20)
    plt.rc('legend',fontsize='13')

    th_suffix = 'TH'
    title = f'{"ROC Curve"}'
    plt.title(title, fontsize=16, y=1.05, pad=10, fontweight='bold', color='#363062') 


# ## Precision-Recall Curve

# In[34]:


def keras_pr_curve (model, xtrain, ytrain, xcv, ycv, proba = True):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    
    if proba == True:
        probs = model.predict_proba(xcv)
        predictions = probs[:, 1]
    else:
        probs = model.predict(xcv)
        predictions = probs[:, 0]
    
    precision, recall, thresholds = precision_recall_curve(ycv, predictions)
    
    fig = plt.figure(figsize=(7, 4))
    fig.patch.set_facecolor('#FFFFFF')
    
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


# ## Metrics

# In[7]:


def keras_metrics (model, cv, ycv, et, yet, best_threshold, proba = True):
    
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve 
    import tensorflow as tf
    from numpy.random import seed
    
    np.random.seed(1)
    tf.random.set_seed(1)
    
    if proba == True:
        y_predcv = (model.predict_proba(cv)[:,1] >= best_threshold).astype(int)
        y_predet = (model.predict_proba(et)[:,1] >= best_threshold).astype(int)
    else:
        y_predcv = (model.predict(cv)[:,0] >= best_threshold).astype(int)
        y_predet = (model.predict(et)[:,0] >= best_threshold).astype(int)
    
    roc_auc_et=roc_auc_score(yet, y_predet) 
    roc_auc_cv=roc_auc_score(ycv, y_predcv) 
    
    precision_cv, recall_cv, thresholds_cv = precision_recall_curve(ycv, y_predcv)
    auc_precision_recall_cv = auc(recall_cv, precision_cv)
    
    precision_et, recall_et, thresholds_et = precision_recall_curve(yet, y_predet)
    auc_precision_recall_et = auc(recall_et, precision_et)
    
    sklearn.metrics.confusion_matrix(ycv, y_predcv)
    tn, fp, fn, tp = confusion_matrix(ycv, y_predcv).ravel()
    specificitycv = tn / (tn+fp) 
    sensitivitycv = tp / (tp+fn) 
    balancedaccuracycv = balanced_accuracy_score(ycv, y_predcv)
    mcc_cv=matthews_corrcoef(ycv, y_predcv)  

    sklearn.metrics.confusion_matrix(yet, y_predet)
    Tn, Fp, Fn, Tp = confusion_matrix(yet, y_predet).ravel()
    specificityet = Tn / (Tn+Fp)
    sensitivityet = Tp / (Tp+Fn)
    balancedaccuracyet = balanced_accuracy_score(yet, y_predet)
    mcc_et=matthews_corrcoef(yet, y_predet)
    
    p_cv = precision_score(ycv, y_predcv)
    p_et = precision_score(yet, y_predet)
    r_cv = recall_score(ycv, y_predcv)
    r_et = recall_score(yet, y_predet)
    
    p_r_m_cv = p_cv*r_cv
    p_r_a_cv = p_cv+r_cv
    
    p_r_m_et = p_et*r_et
    p_r_a_et = p_et+r_et
    
    f1_cv = 2*(p_r_m_cv/p_r_a_cv)
    f1_et = 2*(p_r_m_et/p_r_a_et)
    
    roc_av = (roc_auc_cv+roc_auc_et)/2
    ba_av = (balancedaccuracycv+balancedaccuracyet)/2
    sens_av = (sensitivitycv+sensitivityet)/2
    spec_av = (specificitycv+specificityet)/2
    mcc_av = (mcc_cv+mcc_et)/2
    pr_av = (p_cv+p_et)/2
    r_av = (r_cv+r_et)/2
    f_av = (f1_cv+f1_et)/2
    pr_auc_av = (auc_precision_recall_cv+auc_precision_recall_et)/2
    
    df = {'CV Set': [best_threshold, roc_auc_cv,auc_precision_recall_cv, balancedaccuracycv, sensitivitycv, specificitycv,mcc_cv, p_cv, r_cv, f1_cv], 
         'Test Set':[best_threshold, roc_auc_et, auc_precision_recall_et,balancedaccuracyet, sensitivityet, specificityet, mcc_et, p_et, r_et, f1_et],
         'Average': [best_threshold, roc_av, pr_auc_av, ba_av, sens_av, spec_av, mcc_av, pr_av, r_av, f_av]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df


# ## Threshold 

# In[5]:


def keras_threshold(model,xcv, ycv, threshold_type, proba = True):
    import numpy as np
    import ghostml
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics
    from sklearn.metrics import f1_score
    from numpy.random import seed
    import tensorflow as tf

    np.random.seed(1)
    tf.random.set_seed(1)
    
    if proba == True:
        preds = model.predict_proba(xcv)
        probs = preds[:, 1]
    else:
        preds = model.predict(xcv)
        probs = preds[:, 0]
        
    if threshold_type == 'f_score':
        precision, recall, thresholds = precision_recall_curve(ycv, probs)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
    if threshold_type == 'youden':
        fpr, tpr, thresholds = metrics.roc_curve(ycv, probs)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    if threshold_type == 'range':
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        thresholds = np.arange(0, 1, 0.001)
        scores = [f1_score(ycv, to_labels(probs, t)) for t in thresholds]
        ix = np.argmax(scores)
        best_thresh = thresholds[ix] 
    if threshold_type == 'ghost kappa':
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ycv, probs, thresholds, ThOpt_metrics = 'Kappa', random_seed=1) 
    if threshold_type == 'ghost roc':
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ycv, probs, thresholds, ThOpt_metrics = 'ROC', random_seed=1) 
        
    return best_thresh


# ## Confusion Matrix 

# In[6]:


def keras_conf_mat(model, xconf_mat, yconf_mat, threshold, proba = True):
    
    import seaborn as sns
    from sklearn import metrics
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from numpy.random import seed
    import tensorflow as tf

    np.random.seed(1)
    tf.random.set_seed(1)
    
    if proba == True:
        y_pred = (model.predict_proba(xconf_mat)[:,1] >= threshold).astype(int)
    else:
        y_pred = (model.predict(xconf_mat)[:,0] >= threshold).astype(int)
    
            
    matrix = confusion_matrix(yconf_mat, y_pred)
    tn, fp, fn, tp = matrix.ravel()

    fig = plt.figure()
    fig.patch.set_facecolor('#FFFFFF')
    cfm_plot = sns.heatmap(matrix, cmap='Blues', annot=True, fmt='.0f', annot_kws = {'size':16},
                           xticklabels = ["Inactive", "Active"] , yticklabels = ["Inactive", "Active"], 
                           facecolor='white')
    cfm_plot.set_xlabel('Predicted',fontsize=16, labelpad=28, color='#4D4C7D')
    cfm_plot.set_ylabel('True',fontsize=16, labelpad=28, color='#4D4C7D')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt.show()


# # Preprocess SMILES for CNN

# In[17]:


def preprocess_cnn (xsmiles, batch ):
    import tensorflow as tf
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    #Convert SMILES to Mol
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in xsmiles]
    
    #Convert Mol to Images
    image_list = [Draw.MolToImage(mol, size=(32, 32)) for mol in mol_list]
    
    #Convert Images to Array
    image_array = np.array([np.asarray(image) for image in image_list])
    
    #Scale pixels from [0,1]
    image_array_scaled = image_array/255 

    #Add batch dimension only for training set
    if batch == True:
        image_array_scaled = tf.expand_dims(image_array_scaled, 0)
    
    return image_array_scaled


# # Fingerprints

# In[29]:


def fp(smiles, smiles_column, x, y, bits):
    from rdkit import Chem
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    import pandas as pd 
    import numpy as np
    
    smiles = pd.DataFrame(smiles)
    fp = []
    for i in smiles[smiles_column]:
        w = Chem.MolFromSmiles(i)
        fp.append(w)
    
    fp = pd.DataFrame(fp)
    fp.columns = ['Fingerprint']

    x_wfp = pd.concat([x, fp], axis=1)
    x_wfp1 = x_wfp.dropna(axis=0, how='any') 
    x_wfp2 = x_wfp1.reset_index(drop=True)
    
    fp1 = []

    for i in x_wfp2['Fingerprint']:
        fingerprint = GetMorganFingerprintAsBitVect(i, radius=2, nBits=bits)
        fp1.append(fingerprint)
    
    fp_final = []
    for i in fp1:
        fp_array = np.zeros((1, ))
        ConvertToNumpyArray(i, fp_array)
        fp_final.append(fp_array)
    
    x_fp = pd.DataFrame(fp_final)
    x_fp.reset_index(drop=True, inplace=True)

    del x_wfp2['Fingerprint']

    x1 = pd.concat([x_wfp2, x_fp], axis=1)
    combined = pd.concat([y,smiles,x1], axis = 1)
    return combined


# # Model w/Best F1

# In[ ]:


def max_val_f1score(table):
    import numpy as np
    import pandas as pd
    
    max_f1 = None
    max_f1_header = None
    max_f1_with_test = None
    max_f1_indices = []

    for header in table.columns.levels[0]:
        f1_scores_cv = table[header, 'CV Set']['F1 Score']
        f1_scores_test = table[header, 'Test Set']['F1 Score']
        max_score_cv = np.nanmax(f1_scores_cv)
        max_score_test = np.nanmax(f1_scores_test)

        if np.isnan(max_score_cv):
            continue

        if max_f1 is None or max_score_cv > max_f1 or (max_score_cv == max_f1 and (not np.isnan(max_score_test) and max_score_test > np.nanmax(table[max_f1_header, 'Test Set']['F1 Score']))):
            max_f1 = max_score_cv
            max_f1_header = header
            max_f1_indices = [header]

        elif max_score_cv == max_f1 and not np.isnan(max_score_test) and max_score_test == np.nanmax(table[max_f1_header, 'Test Set']['F1 Score']):
            max_f1_indices.append(header)

    if len(max_f1_indices) > 1:
        max_test_f1 = None
        for index in max_f1_indices:
            f1_scores_test = table[index, 'Test Set']['F1 Score']
            max_score_test = np.nanmax(f1_scores_test)
            if max_test_f1 is None or max_score_test > max_test_f1:
                max_test_f1 = max_score_test
                max_f1_header = index

    df = table[max_f1_header]
    df.columns = pd.MultiIndex.from_tuples([(max_f1_header, 'CV Set'), (max_f1_header, 'Test Set')])
    return df


# # Model w/Best F1 (80/20)

# In[1]:


def max_val_f1score_80_20(table):
    import numpy as np
    import pandas as pd
    
    max_f1 = None
    max_f1_header = None
    max_f1_with_test = None
    max_f1_indices = []

    for header in table.columns.levels[0]:
        f1_scores_cv = table[header, 'Train Set']['F1 Score']
        f1_scores_test = table[header, 'Test Set']['F1 Score']
        max_score_cv = np.nanmax(f1_scores_cv)
        max_score_test = np.nanmax(f1_scores_test)

        if np.isnan(max_score_cv):
            continue

        if max_f1 is None or max_score_cv > max_f1 or (max_score_cv == max_f1 and (not np.isnan(max_score_test) and max_score_test > np.nanmax(table[max_f1_header, 'Test Set']['F1 Score']))):
            max_f1 = max_score_cv
            max_f1_header = header
            max_f1_indices = [header]

        elif max_score_cv == max_f1 and not np.isnan(max_score_test) and max_score_test == np.nanmax(table[max_f1_header, 'Test Set']['F1 Score']):
            max_f1_indices.append(header)

    if len(max_f1_indices) > 1:
        max_test_f1 = None
        for index in max_f1_indices:
            f1_scores_test = table[index, 'Test Set']['F1 Score']
            max_score_test = np.nanmax(f1_scores_test)
            if max_test_f1 is None or max_score_test > max_test_f1:
                max_test_f1 = max_score_test
                max_f1_header = index

    df = table[max_f1_header]
    df.columns = pd.MultiIndex.from_tuples([(max_f1_header, 'Train Set'), (max_f1_header, 'Test Set')])
    return df


# # Chemprop 

# ## Thresholds 

# In[2]:


def optimized_threshold_chemprop(ytrue, y_probs, threshold_type):
    import numpy as np
    import ghostml
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics 
    from sklearn.metrics import f1_score 
    
    probs = y_probs
    
    if threshold_type == 'f_score':
        precision, recall, thresholds = precision_recall_curve(ytrue, probs)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
    if threshold_type == 'youden':
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, probs)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    if threshold_type == 'range':
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        thresholds = np.arange(0, 1, 0.001)
        scores = [f1_score(ytrue, to_labels(probs, t)) for t in thresholds]
        ix = np.argmax(scores)
        best_thresh = thresholds[ix]
    if threshold_type == 'ghost kappa':
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ytrue, probs, thresholds, ThOpt_metrics = 'Kappa', random_seed=1) 
    if threshold_type == 'ghost roc':
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ytrue, probs, thresholds, ThOpt_metrics = 'ROC', random_seed=1) 
        
    return best_thresh


# ## Metrics 

# In[1]:


def classification_metrics80_20_chemprop (ytrain, ytrain_probs, ytest, ytest_probs, best_threshold):
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    
    y_predtrain = (ytrain_probs >= best_threshold)
    y_predtrain = y_predtrain.astype(int)
    
    y_predtest = (ytest_probs >= best_threshold)
    y_predtest = y_predtest.astype(int)  
    
    roc_auc_test=roc_auc_score(ytest, y_predtest) 
    roc_auc_train=roc_auc_score(ytrain, y_predtrain) 
    
    precision_train, recall_train, thresholds_train = precision_recall_curve(ytrain, y_predtrain)
    auc_precision_recall_train = auc(recall_train, precision_train)
    
    precision_test, recall_test, thresholds_test = precision_recall_curve(ytest, y_predtest)
    auc_precision_recall_test = auc(recall_test, precision_test)
    
    sklearn.metrics.confusion_matrix(ytrain, y_predtrain)
    tn, fp, fn, tp = confusion_matrix(ytrain, y_predtrain).ravel()
    specificitytrain = tn / (tn+fp) 
    sensitivitytrain = tp / (tp+fn) 
    balancedaccuracytrain = (sensitivitytrain + specificitytrain) / 2 
    mcc_train=matthews_corrcoef(ytrain, y_predtrain)  

    sklearn.metrics.confusion_matrix(ytest, y_predtest)
    Tn, Fp, Fn, Tp = confusion_matrix(ytest, y_predtest).ravel()
    specificitytest = Tn / (Tn+Fp)
    sensitivitytest = Tp / (Tp+Fn)
    balancedaccuracytest = (specificitytest + sensitivitytest) / 2
    mcc_test=matthews_corrcoef(ytest, y_predtest)
    
    p_train = precision_score(ytrain, y_predtrain)
    p_test = precision_score(ytest, y_predtest)
    r_train = recall_score(ytrain, y_predtrain)
    r_test = recall_score(ytest, y_predtest)
    
    f1_train = f1_score(ytrain, y_predtrain)
    f1_test = f1_score(ytest, y_predtest)
    
    df = {'Train Set': [best_threshold, roc_auc_train,auc_precision_recall_train, balancedaccuracytrain, sensitivitytrain, specificitytrain,mcc_train, p_train, r_train, f1_train], 
         'Test Set':[best_threshold, roc_auc_test, auc_precision_recall_test,balancedaccuracytest, sensitivitytest, specificitytest, mcc_test, p_test, r_test, f1_test]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df


# # PyTorch

# ## Thresholds 

# In[8]:


def torch_threshold(model,xcv, ycv, threshold_type):
    import numpy as np
    import ghostml
    from sklearn.metrics import precision_recall_curve
    from sklearn import metrics
    from sklearn.metrics import f1_score
    from numpy.random import seed
    import tensorflow as tf

    probs = model(xcv)
    probs=probs.detach().numpy()
    ycv = ycv.detach().numpy()

    if threshold_type == 'f_score':
        precision, recall, thresholds = precision_recall_curve(ycv, probs)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
    if threshold_type == 'youden':
        fpr, tpr, thresholds = metrics.roc_curve(ycv, probs)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    if threshold_type == 'range':
        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')
        thresholds = np.arange(0, 1, 0.001)
        scores = [f1_score(ycv, to_labels(probs, t)) for t in thresholds]
        ix = np.argmax(scores)
        best_thresh = thresholds[ix] 
    if threshold_type == 'ghost kappa':
        probs = probs.squeeze()
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ycv, probs, thresholds, ThOpt_metrics = 'Kappa', random_seed=1) 
    if threshold_type == 'ghost roc':
        probs = probs.squeeze()
        thresholds = np.arange(0, 1, 0.001)
        best_thresh = ghostml.optimize_threshold_from_predictions(ycv, probs, thresholds, ThOpt_metrics = 'ROC', random_seed=1) 
        
    return best_thresh


# ## Metrics 

# In[10]:


def torch_metrics (model, cv, ycv, et, yet, best_threshold):
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve   
    
    probs_cv = model(cv)
    probs_cv=probs_cv.detach().numpy()
    probs_et = model(et)
    probs_et = probs_et.detach().numpy()
    ycv = ycv.detach().numpy()
    yet = yet.detach().numpy()
    
    
    y_predcv = (probs_cv >= best_threshold)
    y_predcv = y_predcv.astype(int)    
    
    y_predet = (probs_et >= best_threshold)
    y_predet = y_predet.astype(int)  
    
    roc_auc_et=roc_auc_score(yet, y_predet) 
    roc_auc_cv=roc_auc_score(ycv, y_predcv) 
    
    precision_cv, recall_cv, thresholds_cv = precision_recall_curve(ycv, y_predcv)
    auc_precision_recall_cv = auc(recall_cv, precision_cv)
    
    precision_et, recall_et, thresholds_et = precision_recall_curve(yet, y_predet)
    auc_precision_recall_et = auc(recall_et, precision_et)
    
    sklearn.metrics.confusion_matrix(ycv, y_predcv)
    tn, fp, fn, tp = confusion_matrix(ycv, y_predcv).ravel()
    specificitycv = tn / (tn+fp) 
    sensitivitycv = tp / (tp+fn) 
    balancedaccuracycv = (sensitivitycv + specificitycv) / 2 
    mcc_cv=matthews_corrcoef(ycv, y_predcv)  

    sklearn.metrics.confusion_matrix(yet, y_predet)
    Tn, Fp, Fn, Tp = confusion_matrix(yet, y_predet).ravel()
    specificityet = Tn / (Tn+Fp)
    sensitivityet = Tp / (Tp+Fn)
    balancedaccuracyet = (specificityet + sensitivityet) / 2
    mcc_et=matthews_corrcoef(yet, y_predet)
    
    p_cv = precision_score(ycv, y_predcv)
    p_et = precision_score(yet, y_predet)
    r_cv = recall_score(ycv, y_predcv)
    r_et = recall_score(yet, y_predet)
    
    f1_cv = f1_score(ycv, y_predcv)
    f1_et = f1_score(yet, y_predet)
      
    df = {'CV Set': [best_threshold, roc_auc_cv,auc_precision_recall_cv, balancedaccuracycv, sensitivitycv, specificitycv,mcc_cv, p_cv, r_cv, f1_cv], 
         'Test Set':[best_threshold, roc_auc_et, auc_precision_recall_et,balancedaccuracyet, sensitivityet, specificityet, mcc_et, p_et, r_et, f1_et]}
    df = pd.DataFrame(df, index = ['Threshold', 'ROC AUC','PR AUC','Balanced Accuracy', 'Sensitivity', 'Specificity','MCC',
                                   'Precision','Recall', 'F1 Score'])
    return df


# In[ ]:




