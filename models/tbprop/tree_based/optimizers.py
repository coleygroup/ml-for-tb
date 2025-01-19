import warnings
import numpy as np
import matplotlib.pyplot as plt

import optuna
import pandas as pd
from hyperopt import tpe, fmin, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold

from tbprop.tree_based.search_spaces import optuna_space, hyperopt_space, random_search_space
from tbprop.tree_based.model_selection import CrossValidator, randomized_search
from tbprop.tree_based.utils import adjust_params_by_model
from tbprop.metrics import binary_classification_metrics, regression_metrics, max_f1_score

from sklearn.metrics import roc_auc_score, average_precision_score

SUPPORTED_CLF_METRICS = [
    "n_pos",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "f1_score",
]

SUPPORTED_REG_METRICS = ["mse", "rmse", "r2", "mae"]


class Optimizer:

    def __init__(
        self,
        X_train,
        y_train,
        estimator,
        metric="auroc",
        X_val=None,
        y_val=None,
        val_mode="k_fold_fixed_split",
        random_state=1,
    ):
        """
        Initialize an Optimizer object.

        Parameters
        ----------
        X_train: pd.DataFrame
            Training data.
        y_train: pd.Series
            Training labels.
        estimator: sklearn estimator
            Estimator to optimize.
        metric: str
            Metric to optimize.
        X_val: pd.DataFrame
            Validation data.
        y_val: pd.Series
            Validation labels.
        enable_cv: bool
            Whether to use cross validation or not.
        random_state: int
            Random state.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.estimator = estimator

        self.metric = metric
        self.random_state = random_state
        self.val_mode = val_mode

        self.supported_clf_metrics = SUPPORTED_CLF_METRICS
        self.supported_reg_metrics = SUPPORTED_REG_METRICS
        self.supported_metrics = self.supported_clf_metrics + self.supported_reg_metrics

        # Logging metrics
        self.metric_by_iter = []

        # Concatenate train and validation sets if cross validation is enabled
        if self.val_mode.startswith("k_fold") and self.X_val is not None and self.y_val is not None:
            self.X_train = pd.concat([self.X_train, self.X_val])

            if isinstance(self.y_train, pd.Series):
                self.y_train = pd.concat([self.y_train, self.y_val])
            elif isinstance(self.y_train, np.ndarray):
                self.y_train = np.concatenate([self.y_train, self.y_val])
            else:
                raise ValueError("y_train must be either a pd.Series or a np.ndarray.")

        if self.val_mode == "k_fold_fixed_split":
            self.cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ).split(self.X_train, self.y_train)
        elif self.val_mode not in ("k_fold_random_split", "fixed_split"):
            raise ValueError(
                "val_mode must be one of ('k_fold_fixed_split', 'k_fold_random_split', 'fixed_split')."
            )

        # Check if metric is supported and set mode
        if self.metric in self.supported_clf_metrics:
            self.mode = "classification"
            self.metric_names = np.array(self.supported_clf_metrics)
        elif self.metric in self.supported_reg_metrics:
            self.mode = "regression"
            self.metric_names = np.array(self.supported_reg_metrics)
        else:
            raise ValueError(f"metric must be one of {self.supported_metrics}.")

        # Check if validation set is provided or cross validation is enabled
        if not self.val_mode.startswith("k_fold") and (X_val is None or y_val is None):
            raise ValueError(
                "X_val and y_val must be provided if k-fold cross validation is not used."
            )

    def _objective_with_params(self, params):
        """
        Objective function to optimize, given iteration parameters for model.

        Parameters
        ----------
        params: dict
            Parameters of model for iteration.
        """

        _, _, params = adjust_params_by_model(params, self.estimator)

        if self.val_mode == "k_fold_random_split":
            # Use cross validation
            cv = CrossValidator(
                mode="reg" if self.mode == "regression" else "bin_class",
                estimator=self.estimator,
                params=params,
                random_state=self.random_state,
            )
            try:
                cv.fit(self.X_train, self.y_train)
            except TypeError:
                print("params:", params)
                print("estimator:", self.estimator)
                raise TypeError
            score = -cv.mean_metrics[self.metric] / 100

        elif self.val_mode == "k_fold_fixed_split":
            cv_metrics = []

            for _, (trn_idx, val_idx) in enumerate(self.cv):
                # Divide data into train and test
                X_trn, y_trn = self.X_train.iloc[trn_idx], self.y_train.iloc[trn_idx]
                X_val, y_val = self.X_train.iloc[val_idx], self.y_train.iloc[val_idx]

                if self.estimator().__class__.__name__ == "LGBMClassifier":
                    params["verbose"] = params.get("verbose", -1)

                clf = self.estimator(**params, random_state=self.random_state)
                clf.fit(X_trn, y_trn)
                y_pred = clf.predict(X_val)

                if self.mode == "classification":
                    y_score = clf.predict_proba(X_val)[:, 1]
                    metrics = binary_classification_metrics(y_val, y_pred, y_score)
                else:
                    metrics = regression_metrics(y_val, y_pred)

                cv_metrics.append(metrics["Value"].values)

            cv_metrics = np.array(cv_metrics)
            mean_metrics = np.mean(cv_metrics, axis=0)
            score = -np.where(self.metric_names == self.metric, mean_metrics, 0.0).sum() / 100

        else:
            if self.estimator().__class__.__name__ == "LGBMClassifier":
                params["verbose"] = params.get("verbose", -1)

            # Use a specific validation set
            clf = self.estimator(**params, random_state=self.random_state)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_val)

            if self.mode == "classification":
                y_score = clf.predict_proba(self.X_val)[:, 1]
                metrics = binary_classification_metrics(
                    self.y_val, y_pred, y_score, return_type="dict"
                )
            else:
                metrics = regression_metrics(self.y_val, y_pred, return_type="dict")

            score = -metrics[self.metric] / 100

        self.metric_by_iter.append(-score)

        return score

    def plot_metrics(self):
        """
        Plot metric by iteration.
        """
        title = f"{self.metric} vs. Iterations"
        plt.title(title)
        plt.plot(range(len(self.metric_by_iter)), self.metric_by_iter, label=f"{self.metric}")
        plt.xlabel("Iteration")
        plt.ylabel("-Score")
        plt.legend()
        plt.show()


class HyperoptOptimizer(Optimizer):
    """
    Hyperopt optimizer class.
    Models supported: XGBClassifier, LGBMClassifier, RandomForestClassifier
    Predefined search spaces are in pk.search_spaces.
    """

    def __init__(
        self,
        X_train,
        y_train,
        estimator,
        metric="auroc",
        X_val=None,
        y_val=None,
        val_mode="k_fold_random_split",
        random_state=1,
    ):
        super().__init__(X_train, y_train, estimator, metric, X_val, y_val, val_mode, random_state)

    def objective(self, params):
        score = self._objective_with_params(params)
        return {"loss": score, "status": STATUS_OK}


class OptunaOptimizer(Optimizer):
    """
    Optuna optimizer class.
    Models supported: XGBClassifier, LGBMClassifier.
    Predefined search spaces are in pk.search_spaces.
    """

    def __init__(
        self,
        X_train,
        y_train,
        estimator,
        metric="auroc",
        X_val=None,
        y_val=None,
        val_mode="k_fold_random_split",
        random_state=1,
    ):
        super().__init__(X_train, y_train, estimator, metric, X_val, y_val, val_mode, random_state)

    def _suggest_params(self, trial):
        return optuna_space(trial, self.estimator)

    def objective(self, trial):
        params = self._suggest_params(trial)
        score = self._objective_with_params(params)

        return score


def optimize_models(
    models,
    optimizer,
    X_trn,
    y_trn,
    X_tst,
    y_tst,
    X_val=None,
    y_val=None,
    random_state=1,
    random_seed=1,
    metric="auroc",
    max_evals=20,
    val_mode="k_fold_random_split",
    verbose=True,
    pipeline_suffix="",
):

    # Settings
    warnings.filterwarnings("ignore")
    np.random.seed(random_seed)
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    ## Find best parameters for each model

    # Best params for each model
    best_params_collect = []

    for model in models:
        model_name = model().__class__.__name__
        if verbose:
            print(f"Optimizing: {model_name}.")

        if optimizer == "hyperopt":
            opt = HyperoptOptimizer(
                X_trn,
                y_trn,
                model,
                X_val=X_val,
                y_val=y_val,
                val_mode=val_mode,
                metric=metric,
                random_state=random_state,
            )

            # if model_name == 'CatBoostClassifier':
            #     print(f"Hyperopt space = {hyperopt_space(model)}")

            best_params = fmin(
                opt.objective,
                hyperopt_space(model),
                algo=tpe.suggest,
                trials=Trials(),
                max_evals=max_evals,
                rstate=np.random.default_rng(1),
            )

            best_params_collect.append(best_params)

        elif optimizer == "optuna":
            opt = OptunaOptimizer(
                X_trn,
                y_trn,
                model,
                X_val=X_val,
                y_val=y_val,
                val_mode=val_mode,
                metric=metric,
                random_state=random_state,
            )

            study = optuna.create_study()
            study.optimize(opt.objective, n_trials=max_evals, show_progress_bar=True)
            best_params_collect.append(study.best_params)

        elif optimizer == "random_search":
            _, est, _ = adjust_params_by_model({}, model, init=True)

            if val_mode.startswith("k_fold"):
                results = randomized_search(
                    X_trn,
                    y_trn,
                    X_tst,
                    y_tst,
                    est,
                    params=random_search_space(model),
                    verbose=verbose,
                )
            else:
                results = randomized_search(
                    pd.concat([X_trn, X_val]),
                    np.concatenate([y_trn, y_val]),
                    X_tst,
                    y_tst,
                    est,
                    params=random_search_space(model),
                    verbose=verbose,
                )

            best_params_collect.append(results["best_estimator"].get_params())
        else:
            raise ValueError(f"Optimizer: {optimizer} invalid.")

    if verbose:
        print()

    ## Test models with best parameters

    # Trained models stored here
    test_metrics = [["model", "auroc", "ap", "max_f1_score"]]

    for i, model in enumerate(models):
        model_name, best_model, _ = adjust_params_by_model(
            best_params_collect[i], model, optimizer=optimizer, init=True
        )
        if verbose:
            print(f"Model: {model_name} | ", end="")

        best_model.fit(pd.concat([X_trn, X_val]), np.concatenate([y_trn, y_val]))

        preds = best_model.predict(X_tst)

        if metric in SUPPORTED_REG_METRICS:
            metric_val = regression_metrics(y_tst, preds, return_type="dict")[metric]
        if metric in SUPPORTED_CLF_METRICS:
            probs = best_model.predict_proba(X_tst)[:, 1]
            auroc = roc_auc_score(y_tst, probs)
            ap = average_precision_score(y_tst, probs)
            max_f1 = max_f1_score(y_tst, probs)

            # metric_val = binary_classification_metrics(y_tst, preds, probs, return_type='dict')[metric]

        # auc, (lb, ub) = delong_confidence_intervals(y_tst, best_model.predict_proba(X_tst)[:, 1])
        # auc, lb, ub = np.round(auc*100, 3), np.round(lb*100, 3), np.round(ub*100, 3)

        test_metrics.append(
            [model_name + "/" + optimizer + "/" + pipeline_suffix, auroc, ap, max_f1]
        )

        if verbose:
            # print(f"Test AUC: {auc} +/- ({lb}, {ub})")
            print(f"Test AUROC: {auroc}, AP: {ap}, Max F1: {max_f1}")

    test_metrics = pd.DataFrame(test_metrics[1:], columns=test_metrics[0])

    ## Train models on the entire dataset

    trained_models = {}
    thresholds = []

    for i, model in enumerate(models):
        model_name, best_model, _ = adjust_params_by_model(
            best_params_collect[i], model, optimizer=optimizer, init=True
        )

        best_model.fit(pd.concat([X_trn, X_val, X_tst]), np.concatenate([y_trn, y_val, y_tst]))

        trained_models[model_name + "/" + optimizer + "/" + pipeline_suffix] = best_model

        cv = CrossValidator(
            mode="reg" if metric in SUPPORTED_REG_METRICS else "bin_class",
            estimator=model,
            params=best_params_collect[i],
        )
        cv.fit(pd.concat([X_trn, X_val, X_tst]), pd.concat([y_trn, y_val, y_tst]))

        if metric in SUPPORTED_CLF_METRICS:
            thresh_values = cv.calculate_thresholds()
            thresholds.append(thresh_values)

    if metric in SUPPORTED_CLF_METRICS:
        test_metrics["thresh_values"] = thresholds
    else:
        test_metrics["thresh_values"] = 0.0

    return trained_models, test_metrics
