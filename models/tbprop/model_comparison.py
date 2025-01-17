import time
import itertools
import numpy as np
import pandas as pd
from tbprop.tree_based.optimizers import optimize_models
from mlxtend.evaluate import cochrans_q, mcnemar_table, mcnemar, combined_ftest_5x2cv


def compare_two_models(X, y, est1, est2, preds1=None, preds2=None, test_type='mcnemar'):
    """ 
    Compares two models with each other using either McNemar's test
    or 5x2 cross validation.
        
    Parameters
    ----------
    X: pd.DataFrame
        Dataset
    y: pd.Series
        Ground truth
    est1, est2: estimators
    preds1, preds2: np.arrays
    test_type: str
        One of 'mcnemar', '5x2cv'
    """
    
    if preds1 is None and preds2 is None and test_type == 'mcnemar':
        print("preds1 and preds2 are None, changing test to 5x2cv.")
        test_type = '5x2cv'
    
    if test_type == 'mcnemar':
        # metric = chi2
        metric, p = mcnemar(
            mcnemar_table(
                y, 
                preds1, 
                preds2
            ),
            corrected=False
        )
    elif test_type == '5x2cv':
        # metric = f
        metric, p = combined_ftest_5x2cv(
            estimator1=est1,
            estimator2=est2,
            X=X,
            y=y
        )
    else:
        raise ValueError(f"Unknown test type, '{test_type}'!")
    
    return metric, p


def hypothesis_test_multiple_comparison(estimators, predictions, X_test, y_test, multi_alpha=.05, post_hoc_alpha=.05, 
                                        n_splits=5, post_hoc_test_type='mcnemar', verbose=True,
                                        models_trained=True):
    """
    Compares multiple classifiers with each other to detect if the
    difference in performance is statistically significant.
    
    The procedure is to first perform a Cochran's Q test on all classifiers
    and post-hoc tests on each pair of classifiers to detect differences.
    
    Parameters
    ----------
    X_train: pd.DataFrame
        Training set features
    y_train: pd.Series
        Labels for training set
    X_test: pd.DataFrame
        Test set features
    y_test: pd.Series
        Labels for test set
    estimators: dict
        Dict with names (str) as keys, and their corresponding classifier
        objects as values.
    multi_alpha: float
        Critical value for the multiple classifier test.
    post_hoc_alpha: float
        Critical value for the pairwise test.
    n_splits: int
        No. of cv splits during prediction phase.
    post_hoc_test_type: str
        One of 'mcnemar' or '5x2cv'
    models_trained: bool
        True if models are already trained on their respective datasets
        
    Returns
    -------
    report: Optional[pd.DataFrame]
        Results of pairwise post-hoc tests if Cochran's Q rejects H0 else None
    """

    # Cochran's Q test
    q, p_value = cochrans_q(y_test.values, *predictions)

    if verbose:
        print("\nCochran's Q-Test.")
        print("Q: %.3f" % q)
        print("p-value: %.3f" % p_value)
        if p_value < multi_alpha:
            print(f"p-value < alpha ({multi_alpha}) => Reject H0.")
        else:
            print(f"p-value >= alpha ({multi_alpha}) => Accept H0.")

    clf_names = list(estimators.keys())
    clf_objs = list(estimators.values())
    clf_combinations = \
        list(itertools.combinations(range(len(estimators)), 2))
    
    # Bonferroni Correction
    post_hoc_alpha /= len(clf_combinations)

    metrics, p_values, comb1, comb2, reject_h0 = [], [], [], [], []

    # Post-hoc McNemar's test
    print("\nConducting post-hoc tests.\n")
    for i, j in clf_combinations:
        metric, p_value = compare_two_models(X_test,
                                             y_test,
                                             est1=clf_objs[i],
                                             est2=clf_objs[j],
                                             preds1=predictions[i],
                                             preds2=predictions[j],
                                             test_type=post_hoc_test_type)
        
        if not post_hoc_alpha/2 <= p_value <= 1-(post_hoc_alpha/2):
            print(f"Combination: {(clf_names[i], clf_names[j])},\n\t p-value = {round(p_value, 3)}\n")

        metrics.append(metric)
        p_values.append(p_value)
        comb1.append(clf_names[i])
        comb2.append(clf_names[j])
        reject_h0.append(not (post_hoc_alpha/2 <= p_value <= 1-(post_hoc_alpha/2)))
        
    metric = 'chi2' if post_hoc_test_type == 'mcnemar' else 'f'
    
    report = pd.DataFrame({'Model 1': comb1, 
                           'Model 2': comb2, 
                            metric: metrics, 
                           'p': p_values, 
                           'alpha': post_hoc_alpha, 
                           'Reject H0': reject_h0})
    
    return report


def compare_models_optimizers_on_split(models, 
                                       X_trn, y_trn, 
                                       X_tst, y_tst, 
                                       X_val=None, y_val=None, 
                                       random_state=1, 
                                       random_seed=1, 
                                       max_evals=20, 
                                       metric='auroc', 
                                       val_mode='k_fold_random_split', 
                                       pipeline_suffix='', 
                                       optimizer_types=['hyperopt', 'optuna', 'random_search']):
    """
    Compares the performance of models trained using Hyperopt, Optuna, and RandomizedSearchCV.
    """

    times, trained_models, test_metrics = [time.time()], {}, []
    
    for optimizer in optimizer_types:
        print(f"Training models using '{optimizer}'.\n")
        opt_trained_models, opt_test_metrics = optimize_models(models, 
                                                               optimizer, 
                                                               X_trn, y_trn, 
                                                               X_tst, y_tst, 
                                                               X_val=X_val, y_val=y_val, 
                                                               random_state=random_state, 
                                                               random_seed=random_seed, 
                                                               max_evals=max_evals,
                                                               metric=metric, 
                                                               val_mode=val_mode,
                                                               pipeline_suffix=pipeline_suffix)

        trained_models.update(opt_trained_models)
        test_metrics.append(opt_test_metrics.values)

        times.append(time.time())
        print(f"\nOptimizer time = {round(times[-1] - times[-2], 3)} s.\n")

    test_metrics = pd.DataFrame(
        np.concatenate(test_metrics, axis=0), 
        columns=['model', metric, 'lb', 'ub', 'thresh_values'] 
    ).sort_values(metric, ascending=False, axis=0)

    print(f"Total time = {round(times[-1] - times[0], 3)} s.")
    
    return trained_models, test_metrics