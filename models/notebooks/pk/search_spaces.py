import numpy as np
from scipy import stats
from hyperopt import hp


def random_search_space(estimator):
    estimator_type = estimator().__class__.__name__

    if estimator_type in ('XGBClassifier', 'XGBRegressor'):
        params = {
            'n_estimators': stats.randint(100, 500),
            'max_depth': stats.randint(5, 20),
            # 'min_child_weight': stats.uniform(0.1, 1),
            'learning_rate': stats.uniform(0.001, 0.1),
            'subsample': stats.uniform(0.5, 0.5),
            'colsample_bytree': stats.uniform(0.5, 0.5),
            'gamma': stats.uniform(0, 0.5),
            'reg_alpha': stats.uniform(0, 1),
            'reg_lambda': stats.uniform(0, 1),
        }
    elif estimator_type in ('LGBMClassifier', 'LGBMRegressor'):
        params = {
            'n_estimators': stats.randint(100, 500),
            'max_depth': stats.randint(5, 20),
            'num_leaves': stats.randint(31, 200),
            'min_child_samples': stats.randint(20, 500),
            'subsample': stats.uniform(0.5, 0.5),
            'colsample_bytree': stats.uniform(0.5, 0.5),
            'reg_alpha': stats.uniform(0, 1),
            'reg_lambda': stats.uniform(0, 1),
        }
    elif estimator_type in ('CatBoostClassifier', 'CatBoostRegressor'):
        params = {
            'iterations': stats.randint(100, 500),
            'depth': stats.randint(5, 20),
            'learning_rate': stats.uniform(0.001, 0.1),
            'l2_leaf_reg': stats.uniform(0.001, 0.1),
            'border_count': stats.randint(32, 255),
            'bagging_temperature': stats.uniform(0, 1),
            'random_strength': stats.uniform(0, 1),
        }
    elif estimator_type in ('RandomForestClassifier', 'RandomForestRegressor'):
        params = {
            'n_estimators': stats.randint(100, 500),
            'max_depth': stats.randint(5, 20),
            'min_samples_split': stats.randint(2, 11),
            'min_samples_leaf': stats.randint(1, 11),
            'max_features': ['auto', 'sqrt', 'log2'],
            'criterion': ['gini', 'entropy'],
        }
    else:
        raise ValueError(f"Estimator type '{estimator_type}' is invalid!")

    if 'Regressor' in estimator_type:
        if 'min_samples_split' in params:
            del params['min_samples_split']
        if 'min_child_weight' in params:
            del params['min_child_weight']

    return params


def optuna_space(trial, estimator):
    estimator_type = estimator().__class__.__name__

    if estimator_type in ('XGBClassifier', 'XGBRegressor'):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 1),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
        }
    elif estimator_type in ('LGBMClassifier', 'LGBMRegressor'): 
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1e-1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
    elif estimator_type in ('RandomForestClassifier', 'RandomForestRegressor'):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 11),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 11),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        }
    elif estimator_type in ('CatBoostClassifier', 'CatBoostRegressor'):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 1, 10),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),
            'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 0.001, 0.1),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_uniform('random_strength', 0, 1),
        }
    else:
        raise ValueError(f"Estimator type '{estimator_type}' is invalid!")
    
    if 'Regressor' in estimator_type:
        if 'min_samples_split' in params:
            del params['min_samples_split']
        if 'min_child_weight' in params:
            del params['min_child_weight']

    return params


def hyperopt_space(estimator):
    estimator_type = estimator().__class__.__name__

    if estimator_type in ('RandomForestClassifier', 'RandomForestRegressor'):
        params = {
            'n_estimators': hp.choice('n_estimators', np.arange(50, 1000+1, 1, dtype=int)),
            'max_depth': hp.choice('max_depth', np.arange(1, 100+1, 1, dtype=int)),
            'min_samples_split': hp.choice('min_samples_split', np.arange(2, 100+1, 1)),
            'n_jobs': -1
        }
    elif estimator_type in ('XGBClassifier', 'XGBRegressor'):
        params = {
            'max_depth': hp.choice('max_depth', np.arange(2, 10+1, 1, dtype=int)),
            'reg_alpha' : hp.uniform('reg_alpha', 0, 1),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
            'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
            'gamma' : hp.uniform('gamma', 0, 5),
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
            'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),
            'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
            'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
            'nthread' : -1
        }
    elif estimator_type in ('LGBMClassifier', 'LGBMRegressor'):
        params = {
            'n_estimators': hp.choice('n_estimators', np.arange(20, 100+1, 1, dtype=int)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'max_depth': hp.choice('max_depth', np.arange(3, 12+1, 1, dtype=int)), 
            'num_leaves': hp.choice('num_leaves', np.arange(20, 100+1, 1, dtype=int)),
            'verbose': -1, 
            'n_jobs': -1
        }
    elif estimator_type in ('CatBoostClassifier', 'CatBoostRegressor'):
        params = {
            'iterations': hp.choice('iterations', np.arange(100, 500+1, 1, dtype=int)),
            'depth': hp.choice('depth', np.arange(1, 9+1, 1, dtype=int)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.001, 0.1),
            'border_count': hp.choice('border_count', np.arange(32, 255+1, 1, dtype=int)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
            'random_strength': hp.uniform('random_strength', 0, 1),
            'verbose': False
        }
    else:
        raise ValueError(f"Estimator type '{estimator_type}' is invalid!")

    if 'Regressor' in estimator_type:
        if 'min_samples_split' in params:
            del params['min_samples_split']

    return params
