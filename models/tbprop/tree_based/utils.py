import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, accuracy_score


def plot_losses(losses, title="Loss vs. Epochs"):
    """
    Plots training loss by epoch.

    Parameters
    ----------
    losses: List[float]
        List of training losses by epoch.
    title: str
        Title of plot.
    """
    plt.title(title)
    plt.plot(range(len(losses)), losses, label="train loss")
    plt.ylim([min(losses) - 0.05, max(losses) + 0.05])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def adjust_params_by_model(params, model, optimizer="optuna", init=False):
    """
    Utility function to adjust parameters for a given model.

    Parameters
    ----------
    params: dict
        Parameters to adjust.
    model: sklearn estimator
        Model to adjust parameters for.
    init: bool
        Whether to initialize model with parameters or not.
    """
    model_name = model().__class__.__name__

    if model_name == "LGBMClassifier" and optimizer != "random_search":
        legal_lgbm_params = list(LGBMClassifier().get_params().keys())
        params = {k: v for k, v in params.items() if k in legal_lgbm_params}
    elif model_name == "CatBoostClassifier":
        params["verbose"] = False
    elif model_name == "CatBoostRegressor":
        if "n_jobs" in params:
            del params["n_jobs"]
        params["verbose"] = False
    elif model_name == "RandomForestRegressor":
        if "learning_rate" in params:
            del params["learning_rate"]
    # if 'min_samples_split' in params:
    #     del params['min_samples_split']

    initialized_model = model(**params) if init else None

    return model_name, initialized_model, params


def threshold_analysis(X_test, y_test, clf, X_et=None, clf_name=None, plot_values=True):
    """
    Utility function to analyze various thresholds for a given classifier.

    Parameters
    ----------
    X_test: np.ndarray
        Test features.
    y_test: np.ndarray
        Test labels.
    clf: sklearn estimator
        Classifier to analyze.
    clf_name: str
        Name of classifier.
    plot_values: bool
        Whether to plot thresholds and precision-recall curves.

    Returns
    -------
    thresh_values: dict
        Dictionary of metrics at best F1 threshold and best precision threshold.
    """

    # Get precision and recall for all thresholds
    prec, rec, thresh = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])

    # Get F1 score for all thresholds
    f1 = (2 * prec * rec) / (prec + rec)
    thresh = np.append(thresh, 1.0)

    # Indices for best F1 and precision
    best_f1_idx = np.argmax(f1)
    best_prec_idx = np.argmax([x for x in prec if x != 1.0])

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

    if plot_values:
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axes[0].plot(thresh, prec, label="Precision", color="tab:blue")
        axes[0].plot(thresh, rec, label="Recall", color="tab:orange")
        axes[0].plot(thresh, f1, label="F1 score", color="tab:green")
        axes[0].axvline(best_f1_thresh, color="tab:cyan", linestyle="--", label="Best F1 Threshold")
        axes[0].text(best_f1_thresh + 0.02, 0, str(round(best_f1_thresh, 2)), rotation=90)
        axes[0].axvline(
            best_prec_thresh, color="tab:pink", linestyle="--", label="Best Precision Threshold"
        )
        axes[0].text(best_prec_thresh + 0.02, 0, str(round(best_prec_thresh, 2)), rotation=90)
        axes[0].legend(loc="lower right")
        axes[0].set_title("Precision, Recall, F1 vs Threshold")
        axes[0].set_xlabel("Threshold")

        if clf_name is None:
            clf_name = clf.__class__.__name__

        display = PrecisionRecallDisplay.from_estimator(
            clf, X_test, y_test, name=clf_name, ax=axes[1]
        )

        _ = display.ax_.set_title("Precision vs Recall")
        display.ax_.axvline(
            best_prec_rec, color="tab:pink", linestyle="--", label="Best Precision Threshold"
        )
        display.ax_.axhline(best_prec, color="tab:pink", linestyle="--")

        display.ax_.axvline(
            best_f1_rec, color="tab:cyan", linestyle="--", label="Best F1 Threshold"
        )
        display.ax_.axhline(best_f1_prec, color="tab:cyan", linestyle="--")

        plt.legend()
        plt.show()

    thresh_values = {
        "best_f1_thresh": best_f1_thresh,
        "best_prec_thresh": best_prec_thresh,
        "best_f1": best_f1,
        "best_f1_prec": best_f1_prec,
        "best_f1_rec": best_f1_rec,
        "best_prec": best_prec,
        "best_prec_f1": best_prec_f1,
        "best_prec_rec": best_prec_rec,
    }

    if X_et is not None:
        best_prec_thresh_support_tst = (
            np.sum(clf.predict_proba(X_test)[:, 1] >= best_prec_thresh) * 100 / X_test.shape[0]
        )
        best_f1_thresh_support_tst = (
            np.sum(clf.predict_proba(X_test)[:, 1] >= best_f1_thresh) * 100 / X_test.shape[0]
        )

        best_prec_thresh_support_et = (
            np.sum(clf.predict_proba(X_et)[:, 1] >= best_prec_thresh) * 100 / X_et.shape[0]
        )
        best_f1_thresh_support_et = (
            np.sum(clf.predict_proba(X_et)[:, 1] >= best_f1_thresh) * 100 / X_et.shape[0]
        )

        best_prec_test_acc = accuracy_score(
            y_test, clf.predict_proba(X_test)[:, 1] >= best_prec_thresh
        )
        best_f1_test_acc = accuracy_score(y_test, clf.predict_proba(X_test)[:, 1] >= best_f1_thresh)

        df_analysis = pd.DataFrame(
            [
                [
                    "Best Prec. Thresh.",
                    best_prec_thresh,
                    best_prec,
                    best_prec_rec,
                    best_prec_f1,
                    best_prec_test_acc,
                    best_prec_thresh_support_tst,
                    best_prec_thresh_support_et,
                ],
                [
                    "Best F1 Thresh.",
                    best_f1_thresh,
                    best_f1_prec,
                    best_f1_rec,
                    best_f1,
                    best_f1_test_acc,
                    best_f1_thresh_support_tst,
                    best_f1_thresh_support_et,
                ],
            ],
            columns=[
                "Scheme",
                "Threshold",
                "Precision",
                "Recall",
                "F1 Score",
                "Test Acc.",
                "% Active in Test Set",
                "% Active in E.T. (956) Set",
            ],
        )

        for col in df_analysis.columns:
            if col != "Scheme":
                df_analysis[col] = df_analysis[col].apply(lambda x: np.round(x, 3))
    else:
        df_analysis = None

    return thresh_values, df_analysis


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def overlap_heatmap(
    trained_models, test_metrics, df_et, pipeline_features, n_models=5, top_k=100, plot_heatmap=True
):

    top_models = test_metrics.head(n_models)["model"].values
    overlaps = np.zeros((n_models, n_models))

    et_results = df_et[["mol"]].copy()
    et_sorted_preds = pd.DataFrame()

    models_to_idx = {}
    for i, m in enumerate(top_models):
        models_to_idx[m] = i
        pipeline = "P1" if "P1" in m else "P2"

        # TODO: remove this
        if pipeline == "P1":
            continue

        et_results[m] = trained_models[m].predict(pipeline_features[pipeline])
        m_scores = trained_models[m].predict_proba(pipeline_features[pipeline])[:, 1]
        et_sorted_preds[m] = df_et["mol"].values[np.argsort(m_scores)]

    top_n_sorted_preds = et_sorted_preds.head(top_k)

    for m1, m2 in itertools.combinations(et_sorted_preds.columns, 2):
        overlap = len(set(top_n_sorted_preds[m1]).intersection(set(top_n_sorted_preds[m2])))
        overlaps[models_to_idx[m1], models_to_idx[m2]] = overlap
        overlaps[models_to_idx[m2], models_to_idx[m1]] = overlap

    # overlaps /= top_k

    if plot_heatmap:
        fig, ax = plt.subplots(figsize=(7, 5))

        _, _ = heatmap(
            overlaps,
            top_models,
            top_models,
            ax=ax,
            cmap="Oranges",
            cbarlabel=f"top-{top_k} overlap",
        )

        ax.set_title(f"Top {top_k} overlap between {n_models} best models")
        fig.tight_layout()
        plt.show()

    return et_results, et_sorted_preds
