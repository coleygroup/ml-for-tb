import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from scipy import stats

import h2o
from h2o.automl import H2OAutoML

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold

from pk.metrics import binary_classification_metrics, regression_metrics
from pk.search_spaces import random_search_space
from pk.confidence import delong_confidence_intervals


def train_model(X_train, y_train, X_test, y_test, model, mode):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    if mode == 'bin_class':
        scores = model.predict_proba(X_test)[:,1]
        metrics = binary_classification_metrics(y_test, preds, scores)
    elif mode == 'reg':
        metrics = regression_metrics(y_test, preds)
    elif mode == 'cat_class':
        raise NotImplementedError("Metrics for 'cat_class' not implemented yet.")
    else:
        raise ValueError(f"mode: '{mode}' not recognized.")
    
    return metrics


class CrossValidator:

    def __init__(self, mode='bin_class', estimator=None, model_type='xgboost', params=None, n_splits=5, random_state=1, verbose=False):
        """
        Cross-validation class for binary classification, regression and categorical classification. 
        
        Parameters
        ----------
        mode : str, optional
            Mode of dataset, by default 'bin_class'
        estimator : sklearn estimator, optional
            Estimator to use for cross-validation, by default None
        model_type : str, optional
            Type of model to use for cross-validation, by default 'xgboost'
        params : dict, optional
            Parameters to use for model, by default None
        n_splits : int, optional
            Number of folds for cross-validation, by default 5
        """
        self.mode = mode
        self.model_type = model_type
        self.params = params if params else {}
        self.n_splits = n_splits
        self.estimator = estimator
        self.verbose = verbose
        self.random_state = random_state

        if self.mode == 'bin_class':
            self.cv = StratifiedKFold(n_splits=n_splits)
        elif self.mode == 'reg':
            self.cv = KFold(n_splits=n_splits)
        
        self.feat_imps = []
        self.metrics = []
        self.models = []
        
        self.mean_feat_imps = None
        self.mean_metrics = None
        self.std_metrics = None

        if self.model_type is None and self.estimator is None:
            raise ValueError("Must specify either model_type or estimator.")
        
    def _init_model(self):
        if self.estimator is not None:
            if self.params is not None:
                self.params['random_state'] = self.random_state
                model = self.estimator(**self.params)
            else:
                model = self.estimator(random_state=self.random_state)
        elif self.model_type == 'xgboost':
            if self.mode == 'bin_class':
                model = XGBClassifier(**self.params)
            elif self.mode == 'reg':
                model = XGBRegressor(**self.params)
        elif self.model_type == 'random_forest':
            if self.mode == 'bin_class':
                model = RandomForestClassifier(**self.params)
            elif self.mode == 'reg':
                model = RandomForestRegressor(**self.params)
        elif self.model_type == 'lightgbm':
            if self.mode == 'bin_class':
                model = LGBMClassifier(**self.params)
            else:
                model = LGBMRegressor(**self.params)
        elif self.model_type == 'catboost':
            if self.mode == 'bin_class':
                model = CatBoostClassifier(**self.params)
            elif self.mode == 'reg':
                model = CatBoostRegressor(**self.params)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented!")
            
        return model
    
    def _calc_metrics(self, y_val, val_preds, val_scores=None):
        if self.mode == 'bin_class':
            if val_scores is None:
                raise ValueError("val_scores cannot be None for mode = 'bin_class'.")
            fold_metrics = binary_classification_metrics(y_val, val_preds, val_scores)
        elif self.mode == 'reg':
            fold_metrics = regression_metrics(y_val, val_preds)
        elif self.mode == 'cat_class':
            raise NotImplementedError("Metrics for 'cat_class' not implemented yet.")
        else:
            raise ValueError(f"mode: '{self.mode}' not recognized.")
        
        return fold_metrics
    
    def zero_imp_feats(self):
        if self.mean_feat_imps is None:
            raise ValueError("Model not fitted yet!")
            
        return list(self.mean_feat_imps[self.mean_feat_imps == 0.].keys())
    
    def plot_feat_imps(self):
        if self.mean_feat_imps is None:
            raise ValueError("Model not fitted yet!")
        
        fig = plt.figure(figsize=(10, 15))
        cmap = plt.get_cmap("plasma")
        plt.barh(self.mean_feat_imps.keys(), self.mean_feat_imps.values, color=cmap.colors)
        plt.xlabel("Model Feature Importance")
        plt.show()

    def calculate_thresholds(self):
        if self.mode != 'bin_class':
            raise ValueError("calculate_thresholds() only works for binary classification!")
        
        if self.mean_metrics is None:
            raise ValueError("Model not fitted yet!")

        # Get precision and recall for all thresholds
        prec, rec, thresh = precision_recall_curve(self.y, self.val_scores)

        # Get F1 score for all thresholds
        f1 = (2*prec*rec)/(prec + rec)
        thresh = np.append(thresh, 1.)

        # Indices for best F1 and precision
        best_f1_idx = np.argmax(f1)
        best_prec_idx = np.argmax([x for x in prec if x != 1.])

        # Thresholds for best F1 and precision
        best_f1_thresh = thresh[best_f1_idx]
        best_prec_thresh = thresh[best_prec_idx]

        # Best F1 and precision values
        best_f1 = f1[best_f1_idx]
        best_prec = prec[best_prec_idx]

        # F1 and recall at best precision threshold
        best_prec_f1 = f1[best_prec_idx]
        best_prec_rec = rec[best_prec_idx]
        
        # Precision and recall at best F1 threshold
        best_f1_prec = prec[best_f1_idx]
        best_f1_rec = rec[best_f1_idx]
        
        return {
            'best_f1_thresh': best_f1_thresh,
            'best_prec_thresh': best_prec_thresh,
            'best_f1': best_f1,
            'best_f1_prec': best_f1_prec,
            'best_f1_rec': best_f1_rec,
            'best_prec': best_prec,
            'best_prec_f1': best_prec_f1,
            'best_prec_rec': best_prec_rec
        }

    def fit(self, X, y, feat_names=None):
        if feat_names is not None:
            self.feat_names = feat_names
        elif isinstance(X, pd.DataFrame):
            self.feat_names = X.columns
        else:
            self.feat_names = None

        metrics, models, feat_imps = [], [], []
            
        self.val_preds = np.zeros(len(X))
        self.val_scores = np.zeros(len(X))

        self.X = X
        self.y = y

        fold_count = 1
        for trn_idx, val_idx in self.cv.split(X, y):
            if self.verbose:
                print(f"Fold {fold_count} of {self.n_splits}...")
            
            # Divide data into train and test
            X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Create model
            model = self._init_model()

            # Fit model
            model.fit(X_trn, y_trn)

            # Get validation predictions and scores
            val_preds = model.predict(X_val)
            if self.mode == 'bin_class':
                val_scores = model.predict_proba(X_val)[:,1]
            elif self.mode == 'reg':
                val_scores = model.predict(X_val)

            self.val_preds[val_idx] = val_preds
            self.val_scores[val_idx] = val_scores

            # Metrics for fold
            fold_metrics = self._calc_metrics(y_val, val_preds, val_scores)

            metrics.append(fold_metrics['Value'].values)
            metric_names = fold_metrics['Metric'].values

            models.append(model)
            feat_imps.append(model.feature_importances_)
            
            fold_count += 1

        metrics = np.array(metrics)
        self.mean_metrics = pd.Series(np.mean(metrics, axis=0), index=metric_names)
        self.std_metrics = pd.Series(np.std(metrics, axis=0), index=metric_names)
        metrics = pd.DataFrame(metrics, columns=metric_names)

        feat_imps = np.array(feat_imps)
        
        if self.feat_names is None:
            try:
                self.feat_names = model.get_booster().feature_names
            except:
                raise ValueError("Must specify feat_names if X is not a DataFrame.")
        
        self.mean_feat_imps = pd.Series(np.mean(feat_imps, axis=0), index=self.feat_names)\
                                .sort_values(ascending=True)
        feat_imps = pd.DataFrame(feat_imps, columns=self.feat_names)
        
        self.metrics = metrics
        self.feat_imps = feat_imps
        self.models = models
        
        if self.verbose:
            print("Done.")


def randomized_search(X_train, y_train, X_test, y_test, estimator=None,
                           model_type='xgboost', n_candidates=5, metric='auroc',
                           cross_val_k=5, params=None, random_state=0, verbose=1):
    warnings.filterwarnings("ignore")

    t0 = time()
    
    if estimator is not None:
        clf_template = estimator
    elif model_type == 'xgboost':
        clf_template = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    elif model_type == 'lightgbm':
        clf_template = LGBMClassifier(objective='binary', metric='binary_logloss')
    elif model_type == 'catboost':
        clf_template = CatBoostClassifier(loss_function='Logloss', verbose=False)
    elif model_type == 'random_forest':
        clf_template = RandomForestClassifier()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented!")

    params = params if params else random_search_space(clf_template)

    stratified_kfold = StratifiedKFold(n_splits=cross_val_k, shuffle=True)

    metric_to_scoring = {
        'auroc': 'roc_auc',
        'accuracy': 'accuracy',
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
        'f1': 'f1',
        'precision': 'precision'
    }

    clf = RandomizedSearchCV(clf_template,
                             param_distributions=params, 
                             cv=stratified_kfold, 
                             n_iter=n_candidates, 
                             scoring=metric_to_scoring[metric], 
                             error_score='raise', 
                             verbose=verbose, 
                             n_jobs=-1,
                             random_state=random_state)

    clf.fit(X_train, y_train)
    
    best_estimator = clf.best_estimator_
    
    preds = best_estimator.predict(X_test)
    scores = best_estimator.predict_proba(X_test)[:,1]

    ci = delong_confidence_intervals(y_test, scores)
    
    if verbose > 0:
        print(f"  Time taken = {round(time() - t0)}s")
    
    return {
        "best_estimator": best_estimator,
        "test_preds": preds,
        "test_scores": scores,
        "test_metrics": binary_classification_metrics(y_test, preds, scores),
        "confidence_intervals": ci
    }



def xgboost_with_randomized_grid_search(X_train, y_train, X_test, y_test, 
                                        n_candidates=5, cross_val_k=5, params=None, random_state=0):
    warnings.filterwarnings("ignore")

    t0 = time()
    
    clf_xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

    if params is None:
        params = {
            'n_estimators': stats.randint(150, 1000),
            'learning_rate': stats.uniform(0.01, 0.59),
            'subsample': stats.uniform(0.3, 0.6),
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'colsample_bytree': stats.uniform(0.5, 0.4),
            'min_child_weight': [1, 2, 3, 4]
        }

    stratified_kfold = StratifiedKFold(n_splits=cross_val_k, shuffle=True)

    clf = RandomizedSearchCV(clf_xgb,
                             param_distributions=params, 
                             cv=stratified_kfold, 
                             n_iter=n_candidates, 
                             scoring='roc_auc', 
                             error_score='raise', 
                             verbose=1, 
                             n_jobs=-1,
                             random_state=random_state)

    clf.fit(X_train, y_train)
    
    best_estimator = clf.best_estimator_
    
    preds = best_estimator.predict(X_test)
    scores = best_estimator.predict_proba(X_test)[:,1]
    
    print(f"Time taken = {round(time() - t0)}s")
    
    return {
        "best_estimator": best_estimator,
        "test_preds": preds,
        "test_scores": scores,
        "test_metrics": binary_classification_metrics(y_test, preds, scores)
    }

def automl_fit(df, y_col, **kwargs):
    """ Use H2O AutoML for finding the best possible models. 
    See https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html for
    which parameters can be used.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the label column
    y_col: str
        Name of label column
    
    Returns
    -------
    aml: h2o.automl.H2OAutoML
        The AutoML object containing names and params of best models.
    """
    
    # Convert dataframe to H2OFrame
    hf = h2o.H2OFrame(df)
    
    # Select feature cols
    X_cols = hf.columns
    X_cols.remove(y_col)
    
    # For classification, factor = categorical
    hf[y_col] = hf[y_col].asfactor()  # TODO: Change for regression
    
    # Get some important params for AutoML
    aml = H2OAutoML(
        max_models = kwargs.get('max_models', None),
        max_runtime_secs = kwargs.get('max_runtime_secs', 3600),
        stopping_metric = kwargs.get('stopping_metric', 'AUTO'),
        nfolds = kwargs.get('nfolds', -1),
        exclude_algos=kwargs.get('exclude_algos', None)
    )
    
    aml.train(x=X_cols, y=y_col, training_frame=hf)
    
    return aml