import pandas as pd
import numpy as np
from numpy import genfromtxt

import joblib
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.image as mpimg
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
import cv2
from PIL import Image
from IPython.display import display 
import dataframe_image as dfi

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

import scipy.stats as stats
from statsmodels.stats.weightstats import ztest as ztest
import itertools
from math import sqrt

DATA_PATH = f'data/'
IMAGE_PATH = f'ignore\\images\\'
RANDOMSEED = 42

dataset = ''
suffix = ''
CI = 0.95
z_value = 0


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
    'props': 'font-size: 16px; padding: 20px; font-weight: bold; color: #363062;'
}
index_names2 = {
    'selector': '.row_heading',
    'props': 'font-weight:bold; text-align:left;'
}
index_names3 = {
    'selector': '.row_heading',
    'props': 'text-align:left;'
}

def hola(x=1):
    return x

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
        style = f'background-color: {colors[i]}'
        x.loc[x[groupby] == factor, :] = style
        
        i = not i
    return x


def format_column_leftalign(df, column):
    if df.name == column:
        return ['text-align: left'] * len(df)
    return [''] * len(df)



# MODEL
#---------------------------------------------------------------------------------------------------
def get_savedmodel(file_model):
    # get stored model
    model = os.path.abspath(file_model)
    rfClassifier = joblib.load(model)

    return rfClassifier


def get_savedmodelNpredictions(file_model, file_pred, file_probs):
    # get stored model
    rfClassifier = get_savedmodel(file_model)

    # get stored predictions/probabilities
    y_pred = genfromtxt(f'{file_pred}', delimiter=' ')
    y_probs = genfromtxt(f'{file_probs}', delimiter=' ')

    return rfClassifier, y_pred, y_probs


def get_RFModel_withparams(params):
    #instantiate random forest    
    rfClassifier = RandomForestClassifier(**params)

    return rfClassifier

    
def train_RF(rfClassifier, X_train, y_train):
    #train model
    rfClassifier.fit(X_train, np.ravel(y_train))

    return rfClassifier


def predict_RF(rfClassifier, x_test):
    # Make predictions 
    y_pred = rfClassifier.predict(x_test)

    # get probabilities 
    # keep probabilities for the positive outcome only
    y_probs = rfClassifier.predict_proba(x_test)[:,1]

    return y_pred, y_probs



# TRAINSPLITS
#---------------------------------------------------------------------------------------------------
def get_trainssplit_ratios(split_size_start, split_size_increment):
    test_set_sizes = np.arange(split_size_start, 1 - split_size_increment / 2, split_size_increment)
    return test_set_sizes


def get_trainsplits(df, stratifyColumn, trainsize, randomstate): 
    # split data into train and validation/test combo
    X_train, x_test = train_test_split(df, stratify=df[stratifyColumn], train_size=trainsize, random_state=randomstate)
    
    #get labels & delete them from train
    # np.ravel:  return a contiguous flattened array
    y_train = np.ravel(pd.DataFrame(X_train['Activity']))
    del X_train['Activity']

    #get labels & delete them from cv dataset
    y_test = np.ravel(pd.DataFrame(x_test['Activity']))
    del x_test ['Activity']            

    return X_train, y_train, x_test, y_test


def score_trainsplits(df, classifier, stratifyColumn, split_size_start, split_size_increment, metric, cv, random_state, filename):
    trainsplit_ratios = get_trainssplit_ratios(split_size_start, split_size_increment)

    # df to save scores
    dfResults = pd.DataFrame(columns=['trainsize', 'metric', 'mean_score', 'std'])

    # iterate through trainsplit_ratios
    for trainsize in trainsplit_ratios:

        # Split data into test and train sets and score
        X_train, y_train, x_test, y_test = get_trainsplits(df, stratifyColumn, trainsize, random_state)
        scores = cross_val_score(classifier, X_train, y_train, scoring=metric, cv=cv)
        dfResults.loc[len(dfResults.index)] = [trainsize, metric, np.mean(scores), np.std(scores)]    
        dfResults.to_csv(f'{filename}.csv')

    return dfResults


def save_trainsplits_results(dfTrainSplitScores, dfResults, metric, order, filename, dataset='Train', caption=''):
    #set identification variables
    index = 'training size'
    method = 'Train/test splits'

    #set ci values
    f1_mean = dfTrainSplitScores['mean_score'].mean()
    std_mean = dfTrainSplitScores['std'].mean()
    ci_lower = f1_mean - std_mean
    ci_upper = f1_mean + std_mean
  
    dfResults.loc[index] = [order, method, dataset, metric, CI, f1_mean, ci_lower, ci_upper]    
    dfResults.to_csv(f'{filename}.csv')

    return dfResults


def format_df(df, caption=''):
    df_styled = df.copy()
    df_styled = df_styled.style.set_caption(caption)\
        .set_table_styles([index_names, headers, caption_css])\
        .hide(axis='columns', subset='order')\
        .hide(axis='index')
    
    return df_styled


def save_model_ci(df, caption='', filename='', show=True):
    df_styled = format_df(df, caption)
        
    if len(filename) > 0:    
        dfi.export(df_styled, filename)
        add_imageborder(filename)

        if show:
            plt.figure()
            plt.axis('off')
            img = plt.imread(filename)
            plt.imshow(img)            
            plt.show()
    
    return df_styled


def save_trainsplits_minstd(dfTrainSplitScores, model, F1, dfSplitsMinSTD, filename):
    #set identification variables
    metric, trainingsize, metric_score, std_min = get_trainsplits_minstd(dfTrainSplitScores)

    dfSplitsMinSTD.loc[model] = [metric, trainingsize, std_min, metric_score, F1]    
    dfSplitsMinSTD.to_csv(f'{filename}.csv')

    return dfSplitsMinSTD


def get_trainsplits_minstd(df):
    # get scores from df
    trainsize = df['trainsize'].to_numpy()
    metric = df['metric'].iloc[0].capitalize()
    mean_score = df['mean_score'].to_numpy()
    standard_deviations = df['std'].to_numpy()

    # get min scores and standard deviations
    standard_deviation_min_index = standard_deviations.argmin()
    trainingsize = f'{trainsize[standard_deviation_min_index]:.2f}'
    metric_score = f'{mean_score[standard_deviation_min_index]:.3f}'
    std_min = f'{standard_deviations[standard_deviation_min_index]:.4f}'

    return metric, trainingsize, metric_score, std_min



def plot_traintest_splits_scores(df, filename, y_range=0, resize='min', title='', show=True):
    # minimize y axis range if value passed in
    if y_range != 0:
        if resize == 'min': 
            y_min = df['mean_score'].min() - (df['mean_score'].min() / y_range)
            y_max = df['mean_score'].max() + (df['mean_score'].max() / y_range)
        else:
            y_min = df['mean_score'].min() + (df['mean_score'].min() / y_range)
            y_max = df['mean_score'].max() - (df['mean_score'].max() / y_range)
    

    # plot f1 average by train splits
    #----------------------------------
    fig = go.Figure([
        go.Scatter(
            name='<font font_color="#65647C"><b>Avg F1</b></font>',
            x=df['trainsize'],
            y=round(df['mean_score'], 2),
            mode='lines',
            line=dict(color='#a5c0da', width=3),
            hoverlabel=dict(
                font_color="#65647C",
                bgcolor="#f7f8fb"
            ),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Upper',
            x=df['trainsize'],
            y=round(df['mean_score'] + df['std'], 2),
            mode='lines',
            marker=dict(color='#65647C'),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='95% CI Lower',
            x=df['trainsize'],
            y=round(df['mean_score'] - df['std'], 2),
            marker=dict(color='#65647C'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(240, 219, 219, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.update_layout(
        xaxis_title='<b>Training Size</b>',
        yaxis_title='<b>Avg F1</b>',
        title=f'<b>{title}<br>Avg F1 by Train/Test Split</b>',
        title_x=0.5,
        hovermode='x',
        width=600,
        height=400,
        margin=dict(t=60, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
    )

    if y_range != 0:
        fig.update_layout(
            yaxis_range=[y_min, y_max]
        )

    #fig.update_yaxes(rangemode='tozero')
    config = {'displayModeBar': False}
    if show:
        fig.show(config=config)
        print('')
        print('')

    #save image 
    image = f'{filename}_TrainTestSplits_Metric.jpg'  
    fig.write_image(image)



    # plot f1 sd by train splits
    #----------------------------------
    fig_std = go.Figure([
            go.Scatter(
                name='Std Deviation',
                x=df['trainsize'],
                y=round(df['std'], 4),
                mode='lines',
                line=dict(color='#a5c0da'),
                showlegend=False
            )
        ])

    fig_std.update_layout(
            xaxis_title='<b>Training Size</b>',
            yaxis_title='<b>F1 Standard Deviation</b>',
            title=f'<b>{title}<br>F1 Standard Deviation by Train/Test Split</b>',
            title_x=0.5,
            hovermode='x',
            hoverlabel=dict(
                font_color="#65647C",
                bgcolor="rgba(240, 219, 219, 0.3)"
            ),
            width=600,
            height=400,
            margin=dict(t=60, b=70),
            plot_bgcolor='rgba(0,0,0,0)',
        )

    config = {'displayModeBar': False}

    metric, trainingsize, metric_score, std_min = get_trainsplits_minstd(df)
    footnote = f'Training size  ({trainingsize})  with the smallest deviation  ({std_min}), has an average {metric} score of: {metric_score}'
    fig_std.add_annotation(
        showarrow=False,
        text=footnote,
        font=dict(size=11), 
        xref='x domain',
        x=0.5,
        yref='y domain',
        y=-0.25
        )

    if show:
        fig_std.show(config=config)

    #save image 
    image = f'{filename}_TrainTestSplits_SD.jpg'  
    fig_std.write_image(image)

    return trainingsize, std_min, metric_score



# NORMAL APPROXIMATION
#---------------------------------------------------------------------------------------------------
def save_normalapprox_results(f1_score, y_test, dfResults, metric, order, filename):
    # get mean F1 score on test data
    ci_length = z_value * np.sqrt((f1_score  * (1 - f1_score )) / y_test.shape[0])
    ci_lower = f1_score - ci_length
    ci_upper = f1_score + ci_length

    index = 'normal approx'
    method = 'Normal Approximation'
    dataset = 'Test'
    dfResults.loc[index] = [order, method, dataset, metric, CI, f1_score, ci_lower, ci_upper]
    dfResults.to_csv(f'{filename}.csv')

    return dfResults, ci_length


def plot_normalapprox(f1, ci_length, filename, range, title=''):
    fig, ax = plt.subplots(figsize=(7, 3))
    if len(title) > 0:
        fig.suptitle(title, fontsize=14, x=0.54)

    ax.errorbar(f1, 0, xerr=ci_length, fmt="o")
    ax.set_xlim(range)
    ax.set_yticks(np.arange(1))

    plt.ylabel('Normal\nApproximation', fontsize=14, labelpad=20)
    plt.xlabel('F1 Score', fontsize=14, labelpad=20)
    plt.tight_layout()
    plt.grid(axis="x")

    # save plot to image and display    
    image = f'{filename}_NormalApprox.jpg'    
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.show()



# BOOTSTRAP
#---------------------------------------------------------------------------------------------------
def save_bootstrap_results(dfBoot, dfResults, metric, orderBoot, orderBoot632, filename):
    dataset = 'Train/CV'
    
    # get scores and their mean
    f1_scores = dfBoot['f1_bootTest'].values.tolist()
    f1_mean = np.mean(f1_scores)
    f1_632_scores = dfBoot['f1_632'].values.tolist()
    f1_632_mean = np.mean(f1_632_scores)
   

    # calculate ci
    bootstrap_rounds = dfBoot['round'].max()
    t_value = stats.t.ppf((1 + CI) / 2.0, df=bootstrap_rounds - 1)


    # bootstrap
    f1_se = 0.0
    for f1 in f1_scores:
        f1_se += (f1 - f1_mean) ** 2
    f1_se = np.sqrt((1.0 / (bootstrap_rounds - 1)) * f1_se)

    ci_length = t_value * f1_se
    ci_lower = f1_mean - ci_length
    ci_upper = f1_mean + ci_length

    index = 'bootstrap'
    method = 'Bootstrap'
    order = orderBoot
    dfResults.loc[index] = [order, method, dataset, metric, CI, f1_mean, ci_lower, ci_upper]
    dfResults.to_csv(f'{filename}.csv')


    # bootstrap percentile:  no longer implemented.  gave same results as bootstrap.  keeping for reference.
    '''
    ci_lower = np.percentile(f1_scores, 2.5)
    ci_upper = np.percentile(f1_scores, 97.5)

    index = 'bootstrap%'
    method = 'Bootstrap Percentile'
    order = orderBootPercent
    dfResults.loc[index] = [order, method, dataset, metric, CI, f1_mean, ci_lower, ci_upper]
    dfResults.to_csv(f'{filename}.csv')
    '''

    # bootstrap 632
    f1_632_se = 0.0
    for f1 in f1_632_scores:
        f1_632_se += (f1 - f1_632_mean) ** 2
    f1_632_se = np.sqrt((1.0 / (bootstrap_rounds - 1)) * f1_632_se)
    ci_length = t_value * f1_632_se
    ci_lower = f1_632_mean - ci_length
    ci_upper = f1_632_mean + ci_length
    
  
    index = 'bootstrap632'
    method = 'Bootstrap .632'
    order = orderBoot632
    dfResults.loc[index] = [order, method, dataset, metric, CI, f1_632_mean, ci_lower, ci_upper]
    dfResults.to_csv(f'{filename}.csv')

    return dfResults, f1_scores, f1_mean, f1_632_scores, f1_632_mean



def plot_bootstrap(f1_test, f1, f1_scores, ci, ci_upper, ci_lower, range, filename, bins=7, title='', type='', ymax_mean=50, ymax_ci=25):
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(title) > 0:
        fig.suptitle(title, fontsize=14, x=0.54)

    ax.vlines(f1_test, [0], ymax_mean, lw=2.5, linestyle="-", label=f"F1 Test:  {round(f1_test,3)}", color='#BD574E')
    ax.vlines(f1, [0], ymax_mean, lw=2.5, linestyle="-", label=f"Mean:  {round(f1,3)}", color='#484C7F')
    ax.vlines(ci_lower, [0], ymax_ci, lw=3, linestyle="dotted", label=f"{ci}% CI", color="#EB8242")
    ax.vlines(ci_upper, [0], ymax_ci, lw=3, linestyle="dotted", color="#EB8242")
    ax.hist(
        f1_scores, bins=bins, color="#f0dbdb66", edgecolor="none"
    )
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.ylabel('Count', fontsize=14, labelpad=20)
    plt.xlabel('F1', fontsize=14, labelpad=20)
    plt.tight_layout()
    plt.xlim(range)

    plt.legend(loc="upper right")

    # save plot to image and display    
    image = f'{filename}_Bootstrap{type}.jpg'    
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.show()


def save_bootstrap_testset_results(F1, y_test, y_pred, y_probs, dfResults, metric, order, filename_results, filename, rounds=1100):
    rng = np.random.RandomState(seed=RANDOMSEED)
    idx = np.arange(y_test.shape[0])

    f1_scores = []

    for i in range(rounds):
        predictions_idx = rng.choice(idx, size=idx.shape[0], replace=True)

        truth = y_test[predictions_idx]
        predictions = y_pred[predictions_idx]
        probabilities = y_probs[predictions_idx]

        precision, recall, thresholds = precision_recall_curve(truth, probabilities)  
        np.seterr(invalid='ignore')
        fscore = (2 * (precision * recall)) / (precision + recall)
        ix = np.nanargmax(fscore)
        best_fscore = round(fscore[ix], ndigits = 4)

        f1_scores.append(best_fscore)

    
    #calculate CI
    f1_mean = np.mean(f1_scores)
    t_value = stats.t.ppf((1 + CI) / 2.0, df=rounds - 1)

    # bootstrap
    f1_se = 0.0
    for f1 in f1_scores:
        f1_se += (f1 - f1_mean) ** 2
    f1_se = np.sqrt((1.0 / (rounds - 1)) * f1_se)

    ci_length = t_value * f1_se
    ci_lower = f1_mean - ci_length
    ci_upper = f1_mean + ci_length


    index = 'bootstrapTest'
    method = 'Bootstrap TestSet'
    dataset = 'Test'
    dfResults.loc[index] = [order, method, dataset, metric, CI, f1_mean, ci_lower, ci_upper]
    dfResults.to_csv(f'{filename_results}.csv')

    dff1_scores = pd.DataFrame(f1_scores, columns=['f1test']) 
    dff1_scores.to_csv(f'{filename}.csv')

    return dfResults, dff1_scores



# CI COMPARISON
#---------------------------------------------------------------------------------------------------
def plot_ci_comparisons(df, f1, filename, x_min=0.3, x_max=0.4, title='', redline_height=4):
    y_labels = list(df['method'])
    means = np.array(df['score'])
    lower_error = np.array(df['ci_lower'])
    upper_error = np.array(df['ci_upper'])

    asymmetric_error = [means - lower_error, upper_error - means]


    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.suptitle(title, fontsize=14, fontweight='bold', x=0.57)

    ax.errorbar(means, np.arange(len(means)), xerr=asymmetric_error, fmt="o")
    ax.set_xlim([x_min, x_max])
    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel("F1 Score", fontsize=12)    

    ax.vlines(f1, [0], redline_height, lw=2, color="red", linestyle="--", label=f'F1  -  {round(f1, 4)}')

    #plt.grid()
    plt.tight_layout()
    plt.legend()
    
    # save plot to image and display    
    image = f'{filename}_CIComparisons.jpg'    
    plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.show()


def plot_ci_distribution(dataset, model, metric, dfTrainSplits, dfBootstrap, filename_ts, filename_bs, title=''):    
    dfTrainSplits.mean_score.hist(bins=10, color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False).figure.savefig(filename_ts)
    dfBootstrap.f1_bootTest.hist(bins=10, color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False).figure.savefig(filename_bs)
    plt.clf()
    
    image_ts = mpimg.imread(filename_ts)
    image_bs = mpimg.imread(filename_bs)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(13, 13))
    ax1.axis('off')
    ax1.set_title('Train Splits Distribution', y=.9)
    ax1.imshow(image_ts)
    ax1.margins(x=0)
    
    ax2.axis('off')
    ax2.set_title('Bootstrap Distribution', y=.9)
    ax2.imshow(image_bs)
    ax2.margins(x=0)
    
    #plt.axis('off')
    title = f'{dataset.upper()}: {model.upper()} {metric.upper()} CI Distribution'
    fig.suptitle(title, fontsize=12, fontweight='bold', color='#363062', y=.65)
    plt.show()



def calc_ci_diff(df, dfCIUncertainty, filename, method='method'):    
    df = df.sort_values('order')
    order = 1
    for CI1, CI2 in itertools.combinations(df[method].values.tolist(), 2):
        test = 'ttest'

        # get sample sizes
        N1 = len(df.loc[df[method] == CI1, 'scores'].iloc[0])
        N2 = len(df.loc[df[method] == CI2, 'scores'].iloc[0])

        # get if both samples have normalized distribution
        normalized1 = df.loc[df[method] == CI1, 'normalized'].iloc[0]
        normalized2 = df.loc[df[method] == CI2, 'normalized'].iloc[0]

        if normalized1 == 0 or normalized2 == 0:
            normalized = False
            test = 'mannwhitneyu'
        else:
            normalized = True

        # get sample mean and difference
        mean1 = df.loc[df[method] == CI1, 'scores'].iloc[0].mean()   
        mean2 = df.loc[df[method] == CI2, 'scores'].iloc[0].mean()   
        mean_diff = mean1 - mean2  
        
        # get sample std
        if CI1 == 'normal_approximation':
            std1 = 0
        else:
            std1 = df.loc[df[method] == CI1, 'scores'].iloc[0].std()
        
        if CI2 == 'normal_approximation':
            std2 = 0
        else:
            std2 = df.loc[df[method] == CI2, 'scores'].iloc[0].std()

        # get scores
        scores1 = df.loc[df[method] == CI1, 'scores'].iloc[0]
        scores2 = df.loc[df[method] == CI2, 'scores'].iloc[0]

        # degrees of freedom
        dof = (N1 + N2) - 2

        # get statistical signficance between the CIs and set t/z-value
        if normalized:
            if N1 > 30 and N2> 30:
                statistic, pvalue = ztest(scores1, scores2)
                test = 'ztest'
                tz_val = z_value
            else:
                statistic, pvalue = stats.ttest_ind(scores1, scores2)
                tz_val = stats.t.ppf(q=CI,df=dof)
        else:
            statistic, pvalue = stats.mannwhitneyu(scores1, scores2)
            tz_val = z_value        

        # get average standard deviations between groups
        # Cohen (1988)'s formula
        std_N1N2 = sqrt(
            (
                ((N1 - 1)*(std1)**2) + ((N2 - 1)*(std2)**2)
            ) 
            / dof
        ) 

        # calculate margin of error
        # moe = z × σ / √(n)
        # z = critical value (z or t score)
        # n = sample size
        # σ = population std
        # moe = tz_val * std_N1N2 * sqrt(1/N1 + 1/N2)
        moe = tz_val * (std_N1N2 / sqrt(N1 + N2))

        # calculate ci bounds
        ci_lower = mean_diff - moe
        ci_upper = mean_diff + moe

        # add results to df
        index = f'{CI1} — {CI2}'
        index = f'{index.replace("normal_approximation", "norm approx")}'
        
        dfCIUncertainty.loc[index] = [order, CI1, CI2, test, pvalue, mean_diff, moe, ci_lower, ci_upper]    
        dfCIUncertainty.to_csv(f'{filename}.csv')
        order += 1

    return dfCIUncertainty


def plot_ci_diff(df, title, filename, plot_summary=True, plot_individual=False, width=12, height=6, xmin=-0.6, xmax=0.05, legendxy=(1, 1.05), c='center', legendha='left', legendva='top', legendcolor='black', legendfontsize='10'):
    # summary graph    
    if plot_summary:
        y_labels = df.index.values.tolist()
        means = np.array(df['mean_diff'])
        moe = np.array(df['moe'])
        lower_error = np.array(df['ci_lower'])
        upper_error = np.array(df['ci_upper'])
        asymmetric_error = [means - moe, moe - means]

        fig, ax = plt.subplots(figsize=(width, height))
        fig.suptitle(title)

        ax.errorbar(means, np.arange(len(means)), xerr=asymmetric_error, fmt="o")
        x_min = xmin
        x_max = xmax
        ax.set_xlim([x_min, x_max])
        ax.set_yticks(np.arange(len(means)))
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.set_xlabel("F1 Score", fontsize=12)  

        plt.tight_layout()
        plt.grid(axis="x")
        image = f'{filename}.jpg'    
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.show()

    df = df.sort_values(by='order', ascending=False)
    # individual graphs
    if plot_individual:
        for index, row in df.iterrows():
            y_labels = index.replace('-', '\n')
            means = round(row['mean_diff'], 3)
            pvalue = round(row['pvalue'], 4)
            moe = row['moe']
            lower_error = row['ci_lower']
            upper_error = row['ci_upper']
            CI = str(round(lower_error, 4)) + ' to ' + str(round(upper_error, 4))
            test = row['test'].replace('mannwhitneyu', 'Mann-Whitney U').replace('ztest', 'Z-test')
            x_min = xmin
            x_max = xmax

            fig, ax = plt.subplots(figsize=(6, 3))
            fig.suptitle(title)
            ax.errorbar(means, '', xerr=moe, fmt="o")
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            #ax.set_xlim([x_min, x_max])
            #ax.set_yticks(np.arange(1))

            plt.ylabel(y_labels, fontsize=12, labelpad=10)
            plt.xlabel('CI Difference of Means', fontsize=14, labelpad=20)
            plt.tight_layout()
            plt.grid(axis="x")

            footnote = f'{test}:  p-value={pvalue}  CI= {CI}'
            plt.annotate(footnote,
                xy = legendxy,
                xycoords='axes fraction',
                ha=legendha,
                va=legendva,
                fontsize=legendfontsize,
                color=legendcolor)

            # save plot to image and display    
            image = f'{filename}_{index}.jpg'    
            plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
            plt.show()


def calc_model_diff(df, dfModelComparison, filename):    
    df = df.sort_values('order')

    order = 1
    for index, row in df.iterrows():
        test = 'ttest'
        method = row['method']
        normalizedModels = row['normalized']
        scores1 =  np.array(row['model1_scores'])
        scores2 =  np.array(row['model2_scores'])

        # get sample sizes
        N1 = scores1.size
        N2 = scores2.size

        if normalizedModels == 0:
            normalized = False
            test = 'mannwhitneyu'
        else:
            normalized = True

        # get sample mean and difference
        mean1 = np.mean(scores1) 
        mean2 = np.mean(scores2)   
        mean_diff = mean1 - mean2  
        
        # get sample std
        std1 = np.std(scores1)
        std2 = np.std(scores2)

        # degrees of freedom
        dof = (N1 + N2) - 2

        # get statistical signficance between the CIs and set t/z-value
        if normalized:
            if N1 > 30 and N2> 30:
                statistic, pvalue = ztest(scores1, scores2)
                test = 'ztest'
                tz_val = z_value
            else:
                statistic, pvalue = stats.ttest_ind(scores1, scores2)
                tz_val = stats.t.ppf(q=CI,df=dof)
        else:
            statistic, pvalue = stats.mannwhitneyu(scores1, scores2)
            tz_val = z_value        

        # get average standard deviations between groups
        # Cohen (1988)'s formula
        std_N1N2 = sqrt(
            (
                ((N1 - 1)*(std1)**2) + ((N2 - 1)*(std2)**2)
            ) 
            / dof
        ) 

        # calculate margin of error
        # moe = z × σ / √(n)
        # z = critical value (z or t score)
        # n = sample size
        # σ = population std
        # moe = tz_val * std_N1N2 * sqrt(1/N1 + 1/N2)
        moe = tz_val * (std_N1N2 / sqrt(N1 + N2))

        # calculate ci bounds
        ci_lower = mean_diff - moe
        ci_upper = mean_diff + moe

        # add results to df
        dfModelComparison.loc[method] = [order, test, pvalue, mean_diff, moe, ci_lower, ci_upper]    
        dfModelComparison.to_csv(f'{filename}.csv')
        order += 1

    return dfModelComparison



def format_modelcomparison(df, caption=''):
    df_styled = df.copy()
    df_styled = df_styled.style.set_caption(caption)\
        .set_table_styles([index_names, headers, caption_css])\
        .hide(axis='columns', subset='order')
    return df_styled





def get_model_uncertainty(dataset, suffix, models, sequences, metrics
    , data_path, model_path, image_path, x, y_truth, ci
    , graph_trainsplits_yrange, graph_bootstrap_range
    , graph_bootstrap_bins, graph_bootstrap_ymax_mean, graph_bootstrap_ymax_ci
    , graph_bootstrap632_range, graph_bootstrap632_bins, graph_bootstrap632_ymax_mean
    , graph_bootstrap632_ymax_ci, graph_normapprox_range, graph_ci_range, graph_ci_redline_height
    , filename_compare = 'ci_compare', filename_std = 'ci_minstd', filename_uncertainty='ci_uncertainty'):
    

    # variables for final model comparison
    dfModelComparison_trainsplits = pd.DataFrame(columns=['order', 'model1', 'model2', 'test', 'pvalue', 'mean_diff', 'moe', 'ci_lower', 'ci_upper'])
    dfModelComparison_bootstrap = pd.DataFrame(columns=['order', 'model1', 'model2', 'test', 'pvalue', 'mean_diff', 'moe', 'ci_lower', 'ci_upper'])
    dfResults_trainsplits = pd.DataFrame(columns=['order', 'model', 'normalized', 'scores'])
    dfResults_bootstrap = pd.DataFrame(columns=['order', 'model', 'normalized', 'scores'])
    file_models_compare_trainsplits = f'ModelsComparison_TrainsplitsCI{suffix}'
    file_models_compare_bootstrap = f'ModelsCompare_BootstrapCI{suffix}'


    for model, sequence, metric, ts_yrange, bs_range, bs_bins, bs_ymax_mean, bs_ymax_ci \
      , bs632_range, bs632_bins, bs632_ymax_mean, bs632_ymax_ci, normapprox_range \
      , ci_range, ci_redline_height \
      in zip(models, sequences, metrics, graph_trainsplits_yrange, graph_bootstrap_range \
        , graph_bootstrap_bins, graph_bootstrap_ymax_mean, graph_bootstrap_ymax_ci \
        , graph_bootstrap632_range, graph_bootstrap632_bins, graph_bootstrap632_ymax_mean \
        , graph_bootstrap632_ymax_ci, graph_normapprox_range, graph_ci_range \
        , graph_ci_redline_height):
        
        print (f'\n\nModel:  {model.upper()}')
        print (f'-----------------------------------------------------------------------------------------------')
        # set path and filenames
        path_data = f'{data_path}{model}/'
        path_model = model_path
        path_image = f'{image_path}{model}\\'

        file_base = f'{model}_{metric}'
        file_base_img = f'{path_image}{file_base}{suffix}'
        file_model = f'{path_model}{file_base}{suffix}.mdl'
        file_pred = f'{path_data}{file_base}_ytestpred{suffix}.csv'
        file_probs = f'{path_data}{file_base}_ytestprobs{suffix}.csv'
        file_splits = f'{file_base}_trainsplits{suffix}'
        file_bootstrap = f'{file_base}_bootstrap{suffix}'
        file_model_cicompare = f'{path_data}{file_base}_{filename_compare}{suffix}'
        file_models_cistd = f'{data_path}model_{filename_std}{suffix}'
        file_model_uncertainty = f'{data_path}model_{filename_uncertainty}{suffix}'        
        file_img_cisummary = f'{file_base_img}_CIs.png'
        
        
        # get model performance
        classifier, y_pred, y_probs = get_savedmodelNpredictions(file_model, file_pred, file_probs)
        classifier_f1 = f1_score(y_truth, y_pred)
        classifier_score = classifier.score(x, y_truth)    

        # create df for results
        dfModelResults = pd.DataFrame(columns=['order', 'method', 'dataset', 'metric', 'ci', 'score', 'ci_lower', 'ci_upper'])  
        dfModels_CISTD = pd.DataFrame(columns=['metric', 'training size', 'min deviation', 'mean score', 'F1'])
        dfCIUncertainty = pd.DataFrame(columns=['order', 'method1', 'method2', 'test', 'pvalue', 'mean_diff', 'moe', 'ci_lower', 'ci_upper'])        
        normal_approx = [] 

        

        # Train/test splits
        #----------------------------------------------------------------
        #  get the train test splits results for the model
        dfModelTrainSplits = pd.read_csv(f'{path_data}{file_splits}.csv', index_col=0)
        params = {'dfTrainSplitScores': dfModelTrainSplits, 
            'dfResults': dfModelResults, 
            'metric': metric,
            'order': 3, 
            'filename': file_model_cicompare}

        dfModelResults = save_trainsplits_results(**params)

        # compute the min std for the ci
        dfModels_CISTD = save_trainsplits_minstd(dfModelTrainSplits, model, classifier_f1, dfModels_CISTD, file_models_cistd)

        # graph results
        caption = f'{dataset}: {model.upper()} Model Uncertainty'
        params = {'df': dfModelTrainSplits, 
                'filename': file_base_img, 
                'y_range': ts_yrange,
                'resize': 'min', 
                'title': caption}

        results = plot_traintest_splits_scores(**params)
        trainingsize_metric_stdmin = results[0]
        standard_deviation_min = results[1]
        avgmetric_trainingsize_stdmin = results[2]



        # Bootstrap
        #----------------------------------------------------------------
        #  get the bootstrap results for the model
        dfModelBootstrap = pd.read_csv(f'{path_data}{file_bootstrap}.csv', index_col=0)

        params = {'dfBoot': dfModelBootstrap, 
          'dfResults': dfModelResults, 
          'metric': metric,
          'orderBoot': 1, 
          'orderBoot632': 2,
          'filename': file_model_cicompare}

        results = save_bootstrap_results(**params)
        dfModelResults = results[0]
        f1_scores = results[1]
        f1_mean = results[2]
        f1_632_scores = results[3]
        f1_632_mean = results[4]

        # graph Bootstrap
        ci_lower = dfModelResults.loc['bootstrap']['ci_lower']
        ci_upper = dfModelResults.loc['bootstrap']['ci_upper']
        caption = f'{dataset}: {model.upper()} \nStratified Bootstrap on Training Set'

        params = {'f1_test': classifier_f1, 
                  'f1': f1_mean, 
                  'f1_scores': f1_scores,
                  'ci': CI, 
                  'ci_upper': ci_upper,
                  'ci_lower': ci_lower,
                  'range': bs_range,
                  'filename': file_base_img,
                  'bins': bs_bins,
                  'title': caption,
                  'ymax_mean': bs_ymax_mean,
                  'ymax_ci': bs_ymax_ci
                  }
        plot_bootstrap(**params)

        # graph Bootstrap .632 
        ci_lower = dfModelResults.loc['bootstrap632']['ci_lower']
        ci_upper = dfModelResults.loc['bootstrap632']['ci_upper']
        caption = f'{dataset}: {model.upper()} \nStratified Bootstrap .632 on Training Set'
        type = '632'

        params = {'f1_test': classifier_f1, 
                  'f1': f1_632_mean, 
                  'f1_scores': f1_632_scores,
                  'ci': CI, 
                  'ci_upper': ci_upper,
                  'ci_lower': ci_lower,
                  'range': bs632_range,
                  'filename': file_base_img,
                  'bins': bs632_bins,
                  'title': caption,
                  'type': type,
                  'ymax_mean': bs632_ymax_mean,
                  'ymax_ci': bs632_ymax_ci
                  }
        plot_bootstrap(**params)
        


        # Normal Approximation
        #----------------------------------------------------------------
        normal_approx.append(classifier_f1)
        dfModelNormalApproximation = pd.DataFrame(normal_approx, columns=['score'])
        params = {'f1_score': classifier_f1, 
                  'y_test': y_truth, 
                  'dfResults': dfModelResults,
                  'metric': metric, 
                  'order': 4,
                  'filename': file_model_cicompare
                  }
        dfModelResults, ci_length = save_normalapprox_results(**params)

        caption = f'{dataset}: {model.upper()} \nNormal Approximation on Test Set'
        plot_normalapprox(classifier_f1, ci_length, file_base_img, normapprox_range, caption)



        # Model CI Summary
        #----------------------------------------------------------------
        caption = f'{dataset}: {model.upper()} CI Summary'
        save_model_ci(dfModelResults, caption=caption, filename=file_img_cisummary)
        


        # Model CI Distribution
        #----------------------------------------------------------------
        caption = f'{dataset}: {model.upper()} {ci}% CI'
        dfModelResults_CI = dfModelResults.copy().drop('bootstrap632').sort_values('order')

        params = {'df': dfModelResults_CI, 
                  'f1': classifier_f1, 
                  'filename': file_base_img,
                  'x_min': ci_range[0], 
                  'x_max': ci_range[1],
                  'title': caption,
                  'redline_height': ci_redline_height
                  }
        plot_ci_comparisons(**params)

        #Check distribution of data to help select parametric/nonparametric test        
        caption = f'{dataset}: {model.upper()} F1 Score Distributions'
        filename_ts = f'{file_base_img}_dist_trainsplits.png'
        filename_bs = f'{file_base_img}_dist_bootstrap.png'
        plot_ci_distribution(dataset, model, metric, dfModelTrainSplits, dfModelBootstrap, filename_ts, filename_bs, title=caption)
        


        # Model CI Comparison
        #----------------------------------------------------------------
        # build table for CI's to compare
        dfCICompare = pd.DataFrame(columns=['order', 'method', 'normalized', 'scores'])
        dfCICompare.loc[len(dfCICompare.index)] = [1, 'normal_approximation', 0, dfModelNormalApproximation.score.values]
        dfCICompare.loc[len(dfCICompare.index)] = [2, 'trainsplits', 0, dfModelTrainSplits.mean_score.values]  
        dfCICompare.loc[len(dfCICompare.index)] = [3, 'bootstrap', 1, dfModelBootstrap.f1_bootTest.values]  
        
        # get statistical significance between CI's
        dfCIUncertainty = calc_ci_diff(dfCICompare, dfCIUncertainty, file_model_uncertainty)
        dfCIUncertainty = dfCIUncertainty.round({'pvalue': 5})
        dfCIUncertainty.sort_values(by=['order'], inplace=True)

        title = f'{dataset}:  {model.upper()} CI Comparison'
        params = {'df': dfCIUncertainty, 
                'title': title, 
                'filename': file_model_uncertainty, 
                'plot_summary': False,
                'plot_individual': True, 
                'width': 8,
                'height': 3,
                'legendxy': (.14, 1.10),
                'legendcolor': '#27374D',
                'legendfontsize': 9}
        plot_ci_diff(**params)
        


        # Comparison of all Models
        #----------------------------------------------------------------
        # build table for CI's to compare
        dfResults_trainsplits.loc[len(dfResults_trainsplits.index)] = [sequence, model.upper(), 0, dfModelTrainSplits.mean_score.values]
        dfResults_bootstrap.loc[len(dfResults_bootstrap.index)] = [sequence, model.upper(), 0, dfModelBootstrap.f1_bootTest.values]



    print (f'\n\nModel Comparison:')
    print (f'-----------------------------------------------------------------------------------------------')    
    # get statistical significance between models
    dfModelComparison_trainsplits = calc_ci_diff(dfResults_trainsplits, dfModelComparison_trainsplits, f'{data_path}{file_models_compare_trainsplits}', method='model')
    dfModelComparison_trainsplits = dfModelComparison_trainsplits.round({'pvalue': 5})
    dfModelComparison_trainsplits.sort_values(by=['order'], ascending=False, inplace=True)

    title = f'{dataset} Model Comparison:  Train/Test Splits'
    params = {'df': dfModelComparison_trainsplits, 
            'title': title, 
            'filename': f'{image_path}{file_models_compare_trainsplits}', 
            'plot_summary': False,
            'plot_individual': True, 
            'width': 8,
            'height': 3,
            'legendxy': (.22, 1.10),
            'legendcolor': '#27374D',
            'legendfontsize': 9}
    plot_ci_diff(**params)

    # get statistical significance between models
    dfModelComparison_bootstrap = calc_ci_diff(dfResults_bootstrap, dfModelComparison_bootstrap, f'{data_path}{file_models_compare_bootstrap}', method='model')
    dfModelComparison_bootstrap = dfModelComparison_bootstrap.round({'pvalue': 5})
    dfModelComparison_bootstrap.sort_values(by=['order'], ascending=False, inplace=True)

    title = f'{dataset} Model Comparison:  Bootstrap'
    params = {'df': dfModelComparison_bootstrap, 
            'title': title, 
            'filename': f'{image_path}{file_models_compare_bootstrap}', 
            'plot_summary': False,
            'plot_individual': True, 
            'width': 8,
            'height': 3,
            'legendxy': (.22, 1.10),
            'legendcolor': '#27374D',
            'legendfontsize': 9}
    plot_ci_diff(**params)
