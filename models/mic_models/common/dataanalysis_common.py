import pandas as pd
import numpy as np
import itertools
import sys
from scipy import stats
import matplotlib.pyplot as plt


from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Variance Threshold
#-----------------------------------------------------------------------------------------------------
def get_Descriptors(df, col_NonDescriptors):
    # extract just 2D descriptors
    dfDescriptors = df.loc[:, ~df.columns.isin(col_NonDescriptors)].copy()

    # remove descriptors with zeros in the entire column
    dfDescriptors = dfDescriptors.loc[:, (dfDescriptors != 0).any(axis=0)]

    return dfDescriptors



# Variance Threshold
#-----------------------------------------------------------------------------------------------------
def apply_VarianceThreshold(df, threshold = 0.2, printStatus=True, title = ''):
    """
    Description
    ---------------------------------------------   
    applies sklearn variance threshold by dropping features with variances less than the threshold 
    
    Parameters
    ---------------------------------------------
    df:  panda dataframe to apply variance threshold
    threshold:  threshold to apply, defaults to 0.2
    
    Returns
    ---------------------------------------------
    dfSelected = df without features whose variance is less than threshold 
    droppedFeatures =  lest of features dropped

    Examples
    --------
    df, droppedFeatures = apply_VarianceThreshold(dfDescriptors, 0.2)
    """
    
    valDf = True

    #check valid params were sent
    if not isinstance(df, pd.DataFrame):
        valDf = False
        valMsg += "\nInvalid dataframe"
    
    #if invalid parameters sent, inform caller
    if (not valDf):
        print(valMsg)
    else:
        featuresCount = df.shape[1]

        vt = VarianceThreshold(threshold)
        vt.fit(df)

        # fit the estimator to df, which returns a numpy array w/out column names
        # call get_support() method returns True for columns which are not dropped
        # call those columns from the passed df
        dfVTFeatures = df[df.columns[vt.get_support(indices=True)]]
        droppedFeatures = get_ColumnsDroppped(dfVTFeatures, df)
        varianceCount = dfVTFeatures.shape[1]

        if printStatus:
            print(f'\n{title} Feature count')
            print('------------------------------------------------')
            print('Original count: ' + str(featuresCount))
            print('Count after VT applied: ' + str(varianceCount))
            print(f'Droped features:  {featuresCount - varianceCount}')
            print(f'{droppedFeatures}')


        return dfVTFeatures, droppedFeatures 



def get_ColumnsDroppped(dfNew, dfOld):
    """
    Description
    ---------------------------------------------   
    compares new df to old df, to see which columns do not exist in the new df that exist in the old df 
    
    Parameters
    ---------------------------------------------
    dfNew:  new df to compare to dfOld
    dfOld:  original df will all columns
    
    Returns
    ---------------------------------------------
    dropped_columns = columns dropped from dfOld in dfNew

    Examples
    --------
    droppedFeatures = get_ColumnsDroppped(dfSelected, df)
    """

    valDf = True

    #check valid params were sent
    if not isinstance(dfNew, pd.DataFrame) or not isinstance(dfOld, pd.DataFrame):
        valDf = False
        valMsg += "\nInvalid dataframe"

     #if invalid parameters sent, inform caller
    if (not valDf):
        print(valMsg)
    else:
        dropped_columns = [column for column in dfOld.columns
                        if column not in dfNew.columns]
        
        return dropped_columns
   


# Correlation
#-----------------------------------------------------------------------------------------------------
def apply_Correlation(df, threshold = 0.9, printStatus=True, title = ''):
    featuresCount = df.shape[1]

    #create correlation matrix and select upper triangle
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

    #get features with high correlation
    droppedFeatures = [column for column in upper_tri.columns if any(upper_tri[column] >= threshold)]
    droppedFeaturesCount = len(droppedFeatures)

    #drop features w/high correlation
    dfCorrFeatures = df.drop(droppedFeatures, axis=1)
    correlationCount = dfCorrFeatures.shape[1]

    if printStatus:
        print(f'\n{title} Feature count')
        print('------------------------------------------------')
        print('Original count: ' + str(featuresCount))
        print('Count after Correlated features removed: ' + str(correlationCount))
        print(f'Dropped features: {droppedFeaturesCount}')
        print(f'{droppedFeatures}')

    return dfCorrFeatures, droppedFeatures 



# Outliers
#-----------------------------------------------------------------------------------------------------
def get_Outliers(df, threshold = 3, graphResults=True, title = ''):
    #use Z-score to detect outliers
    z = np.abs(stats.zscore(df))

    # data points too far from zero will be treated as the outliers. In most cases a threshold of 3 or -3 is used
    threshold = 3

    #gets position of outliers (rows and columns)
    rows, columns = np.where(z > 3)

    #get the columns with outliers and their count
    uniqueColumn, columnCount = np.unique(columns, return_counts=True)

    #get the column names that have outliers
    colname = df.columns[uniqueColumn]

    print('Features with outliers: ' + str(len(colname)))
    print('Molecules affected: ' + str(np.count_nonzero(z > 3)))

    if graphResults:
        # Figure Size
        fig, ax = plt.subplots(figsize=(16, 55))

        # Horizontal Bar Plot
        ax.barh(colname, columnCount)

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)

        # Add x, y gridlines
        ax.grid(visible=True, color='grey',
                linestyle='-.', linewidth=0.5,
                alpha=0.2)
        
        # Show top values
        ax.invert_yaxis()

        # Add annotation to bars
        for i in ax.patches:
            #if i == 0:
            #    plt.margins(0)

            plt.text(i.get_width()+0.2, i.get_y()+0.5,
                    str(round((i.get_width()), 2)),
                    fontsize=10, fontweight='bold',
                    color='grey')

        # Add Plot Title
        ax.set_title(f'{title}:  Outliers using Z-score')
        ttl = ax.title
        ttl.set_position([.45, 1.05])
        ttl.set_fontsize(16)

        # Add Text watermark
        fig.text(0.9, 0.15, f'{title} datasets', fontsize=12,
                color='grey', ha='right', va='bottom',
                alpha=0.7)

        ax.margins(0.01, 0.01)

        # Show Plot
        plt.show()


# Feature Distribution
#-----------------------------------------------------------------------------------------------------
def plot_FeatureDistribution(df, title = ''):
    #there are too many features to plot using df.hist.  Had to break the features in subsets of 12
    dfPlot = pd.DataFrame()
    for index in range(df.shape[1]):
        if ((index/12).is_integer()) and (index/12 > 0):
            dfPlot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
            plt.tight_layout(rect=(0, 0, 2, 2))  
            dfPlot = pd.DataFrame()
    
        #get column   
        dfFeature = df.iloc[:, index].to_frame()
        columnName = df.columns[index]
        dfFeature.rename(columns={dfFeature.columns[0]: columnName})

        #add it to the subset of 12 features to plot
        dfPlot = pd.concat([dfPlot,dfFeature])

    dfPlot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
    plt.tight_layout(rect=(0, 0, 2, 2))  
    dfPlot = pd.DataFrame()
