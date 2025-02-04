import pandas as pd
import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

import joblib
import os
import pathlib
import math
import itertools
import re


import dataframe_image as df_image
from IPython.display import display, Image
import cv2
import PIL
from PIL import ImageFont, ImageDraw 

import matplotlib.lines as mplines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from mpl_toolkits.mplot3d import Axes3D

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE



# Common Functions
#-----------------------------------------------------------------------------------------------------
#adapted from https://www.geeksforgeeks.org/python-difference-two-lists/
def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def check_SelectKBest(X, y, score_func, k, features_2check):
    """
    Description
    ---------------------------------------------   
    Implements SelectKBest to get the top k features 
    Checks if the passed list of features is one of them

    Parameters
    ---------------------------------------------
    X:  panda dataframe to use for fit/transform data
    y:  panda dataframe containing label
    score_func:  scoring function
    k:  number of top features to return 
    features_2check:  list of features to check
    
    Returns
    ---------------------------------------------
    returns nothing

    Examples
    --------
    check_SelectKBest(X_train_active_TS, y_train_active_TS, f_classif, k, ts_feat_drop)
    """

    valX = valy = valScore_func = valK = valFeatures_2check = True
    valMsg = "Invalid parameters: "

    #check valid params were sent
    if not isinstance(X, pd.DataFrame):
        valX = False
        valMsg += "\nInvalid dataframe: X"
    
    if not isinstance(y, np.ndarray):
        valy = False
        valMsg += "\nInvalid dataframe:  y"
        print(type(y))
        
    if not callable(score_func):
        valScore_func = False
        valMsg += "\nInvalid score function"

    if not type(k) is int:
        valK = False
        valMsg += "\K must be an integer" 

    if not type(features_2check) is list:
        valFeatures_2check = False
        valMsg += "\features_2check must be a list"  


    #if invalid parameters sent, inform caller
    if (not valX) or (not valy) or (not valScore_func) or (not valK) or (not valFeatures_2check):
        print(valMsg)
    else: 
        #suppress warning 
        with np.errstate(divide='ignore',invalid='ignore'):    
            #get top k features
            selector = SelectKBest(score_func, k=k)    
            selector.fit_transform(X, y)
            
            #get the indices of the features selected
            selector_cols = selector.get_support(indices=True)

            #get the columns from the original df
            X_Important = X.iloc[:,selector_cols]

            message = ''
            columns = X_Important.columns
            for feat in features_2check:
                if feat in columns:
                    column_index = columns.get_loc(feat) + 1
                    if len(message) > 0:
                         message += '\n'
                    message += f'{feat} was ranked {column_index} in the top {k} most important features'

            if len(message) == 0:
                print(f'Features did not rank in the top {k}')
            else:
                print(message)


def add_imageborder(image_filename, color = [255,255,255], width=5):
    """
    Description
    ---------------------------------------------   
    adds a border around the image passed in 

    Parameters
    ---------------------------------------------
    image_filename:  image path & filename
    color:  color of the border to add, defaults to white
    width:  width of the border to add, defaults to 5px
    
    Returns
    ---------------------------------------------
    returns nothing

    Examples
    --------
    add_imageborder(df_filename)
    """

    valImage = True
    valMsg = "Invalid parameters: "
    

    #check valid params were sent
    if image_filename == "":
        valImage = False
        valMsg += "\nimage_filename not provided"

    #if invalid parameters sent, inform caller
    if (not valMsg):
        print(valMsg)
    else:
        img = cv2.imread(image_filename)
        top, bottom, left, right = [width]*4
        img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        cv2.imwrite(image_filename, img_with_border)





# Styles
#-----------------------------------------------------------------------------------------------------
# Set CSS properties for th elements in dataframe
cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'tr:hover',
    'props': 'background-color: #FFE162; color: black; font-weight: bold;'
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
    'props': 'font-size: 16px; padding: 20px; font-weight: bold; color: darkgrey;'
}
index_names2 = {
    'selector': '.row_heading',
    'props': 'font-weight:bold; text-align:left;'
}





# PCA
#-----------------------------------------------------------------------------------------------------
def plot_PCAVariance(var_ratio, pca, title='', subtitle='', filename=''):
    """
    Description
    ---------------------------------------------   
    plots the explained variance ratio and saves it 
    to the image_path declared under global variables

    Parameters
    ---------------------------------------------
    var_ratio:  explained variance ration returned by PCA
    title:  title of the plot, default is empty string
    
    Returns
    ---------------------------------------------
    returns image of plot

    Examples
    --------
    plot_PCAVariance(var_ratio, title)
    """
    
    valVarRatio = True
    valMsg = "Invalid parameters: "

    if not type(var_ratio) is np.ndarray:
        valVarRatio = False
        valMsg += "\nvar_ratio:  should be of type numpy.ndarray"
    
    #if invalid parameters sent, inform caller
    if (not valVarRatio):
        print(valMsg)
    else:
        plt.figure(figsize=(7,5))
        plt.plot(pca.explained_variance_ratio_)    
        plt.title(title, y=1.05, pad=10, fontsize=16, fontweight='bold', color='#363062')
        plt.suptitle(subtitle, y=.92, fontsize=14, fontweight='bold', color='#363062')
        plt.ylabel('explained variance', fontsize=16, labelpad=20);
        plt.xlabel('# of components', fontsize=16, labelpad=20);
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.grid()

        image = f'{filename}.jpg'   
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.close()   
        
        #add border to image
        add_imageborder(image)

        display(Image(image))



def plot_PCAScree(pca, title='', filename=''):
    scree = pca.explained_variance_ratio_*100
    screeLength = np.arange(len(scree))+1
    
    plt.figure(figsize=(36,18))
    plt.bar(screeLength, scree)
    plt.plot(screeLength, scree.cumsum(),c="red",marker='o')
    plt.title(title, y=1.01, pad=10, fontsize=40, fontweight='bold', color='#363062')
    
    ''''''
    ctr = 0
    for x,y in zip(screeLength,scree.cumsum()):
        ctr += 1
        label = "{:.0f}".format(y)

        if (ctr % 2) == 0:            
            if ctr<10:
                xtext = -15
            else:
                xtext = -25
            ytext = 10
        else:
            xtext = 25
            if ctr<10:
                ytext = -10
            else:
                ytext = -30

        plt.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(xtext,ytext), # distance from text to points (x,y)
                     ha='center',  # horizontal alignment can be left, right or center
                     fontsize=22)
    
    
    plt.ylabel('% explained variance', fontsize=34, labelpad=20);
    plt.xlabel('# of components', fontsize=34, labelpad=20);
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    filename = f'{filename}_Scree.jpg'   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()   
    
    #add border to image
    add_imageborder(filename)
    display(Image(filename))



def save_PCAMetrics(df, column, column_values):
    """
    Description
    ---------------------------------------------   
    saves the pca metrics to a df

    Parameters
    ---------------------------------------------
    df:  panda dataframe to add the pca results to
    column:  name of column in df 
    column_values:  ndarray of column values for passed column
    
    Returns
    ---------------------------------------------
    returns df

    Examples
    --------
    save_PCAMetrics(df, column, pc_variances)
    """

    valDf = valColumn = valColumn_values = True
    valMsg = "Invalid parameters: "

    #check valid params were sent
    if not isinstance(df, pd.DataFrame):
        valDf = False
        valMsg += "\nInvalid dataframe"
        
    if column == "":
        valColumn = False
        valMsg += "\nColumn not provided"

    if not type(column_values) is np.ndarray:
        valColumn_values = False
        valMsg += "\nndarray of column values not passed"        

    #if invalid parameters sent, inform caller
    if (not valDf) or (not valColumn) or (not valColumn_values):
        print(valMsg)
    else:        
        #df to keep results
        emptyDF = df.empty
        if emptyDF:
            df = pd.DataFrame()

        # format value
        for ix, value in enumerate(column_values):
            column_values[ix] = '' if math.isnan(value) else round(value, 3)


        if not emptyDF:
            #drop totals columns
            df = df.drop('Ttl')
            df = df.drop('Info Loss')

            len_df = len(df)
            len_new = len(column_values)

            #if new column has less #of rows
            if len_df > len_new:
                #add empty rows to the end, so both columns are of equal len
                start = len_new + 1
                end = len(df) + 1

                empty = []
                for x in range(start, end):
                    empty.append(math.nan)

                column_values = np.append(column_values, empty)
                
        #add column, reset the index, calculate totals     
        df[column] = column_values
        df.index = np.arange(1, len(df)+1)
        df.loc['Ttl'] = df.select_dtypes(np.number).sum()
        df.loc['Info Loss'] = 1 - df.loc['Ttl']

        return df
    


def run_PCAs(pc_nums, x_data, image_path, data_path, font_path, title='', subtitle='', filename=''):
    """
    Description
    ---------------------------------------------   
    Implements SelectKBest to get the top k features 
    Checks if the passed list of features is one of them

    Parameters
    ---------------------------------------------
    pc_nums:  list of components
    x_data:  panda dataframe to use for fit/transform data
    title:  title of polot
    filename_suffix:  suffix for filename if needed
    
    Returns
    ---------------------------------------------
    returns image of results

    Examples
    --------
    run_PCAs(pc_nums, x_data, title, suffixTS)
    """

    valPCNums = valX = True
    valMsg = "Invalid parameters: "


    #check valid params were sent
    if not type(pc_nums) is list:
        valPCNums = False
        valMsg += "\pc_nums must be a list"

    if not isinstance(x_data, pd.DataFrame):
        valX = False
        valMsg += "\nInvalid dataframe: x_data"
    
    #if invalid parameters sent, inform caller
    if (not valX) or (not valPCNums):
        print(valMsg)
    else: 
        # create df to store results
        df = pd.DataFrame()

        #sort list in descending order
        pc_nums = sorted(pc_nums, reverse=True)

        for pc_num in pc_nums:
            column = f'{pc_num} PCs'

            #run pca
            pca = PCA(n_components=pc_num)
            pca.fit_transform(x_data)
            pc_variances = pca.explained_variance_ratio_

            #store results
            df = save_PCAMetrics(df, column, pc_variances)

        df.fillna('', inplace=True)

        #format filename
        imgfilename = f'{image_path}{filename}.jpg' 

        #apply styles to df and save as image
        df.style.set_caption(title)\
                .set_table_styles([index_names, headers, caption_css])
        df_image.export(df, imgfilename)

        #add border to image
        border_width = 50
        add_imageborder(imgfilename, width=border_width)

        #retrieve image because caption/title is not saved
        img = cv2.imread(imgfilename)

        # Convert the image to RGB (OpenCV uses BGR instead of RGB)  
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

        # setup text
        NAVY = (54, 48, 98) ##363062
        font = cv2.FONT_HERSHEY_DUPLEX         
        font_size = [.5, 20]
        font_color = NAVY
        font_thickness = 2

        # get boundary of text
        textsize = cv2.getTextSize(title, font, font_size[0], font_thickness)[0]

        # get coords based on boundary
        img_width = img.shape[1]
        text_width = textsize[0]
        text_height = textsize[1]
        textX = int((img_width - text_width) / 2) 
        textY = int((border_width - text_height) / 2) - 5

        # Pass the image to PIL to use true type fonts, cv2 is limited in font types 
        pil_img = PIL.Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # use a truetype font  
        font = PIL.ImageFont.truetype(font_path, font_size[1])
        
        # Draw the text  
        draw.text((textX, textY), title, font=font, fill=font_color) 

        # convert image back to OpenCV and save  
        cv2_img_processed = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 
        cv2.imwrite(imgfilename, cv2_img_processed)

        #print df to csv and display image
        df.to_csv(f'{data_path}{filename}.csv')
        display(Image(imgfilename))



def format_pca_data(pc, pc_num, x, y, features):
    """
    Description
    ---------------------------------------------   
    creates 2D scatter plot for PCA

    Parameters
    ---------------------------------------------
    pc:  numpy array w/pca fit & transform data
    pc_num:  # of components 
    X:  df w/training data
    y:  label
    features:  list of features
    
    Returns
    ---------------------------------------------
    returns image of plot

    Examples
    --------
    format_pca_data(pc_Xtrain, pc_num, X_train, y_train, feature)
    """

    valPC = valPCNum = valX = valY = valfeatures = True
    valMsg = "Invalid parameters: "

    #check valid params were sent
    if not type(pc_num) is int:
        valPCNum = False
        valMsg += "\npc_num should be of type int"    
    if not isinstance(x, pd.DataFrame):
        valX = False
        valMsg += "\nx is not a dataframe"
    if not type(y) is np.ndarray:
        valY = False
        valMsg += "\ny should be of type numpy.ndarray"
    if not type(features) is pd.core.indexes.base.Index:
        valfeatures = False
        valMsg += "\nfeatures should be of type numpy.ndarray"


    #if invalid parameters sent, inform caller
    if (not valPC) or (not valPCNum) or (not valX) or (not valY) or (not valfeatures):
        print(valMsg)
    else:
        # df to store results
        columns = []
        for ix in range(pc_num):
            columns.append(f'PC{ix + 1}')

        # store pc data in df
        dfPC = pd.DataFrame(data = pc
                , columns = columns)
    
        dfLabel = pd.DataFrame()
        dfLabel['Activity'] = y

        dfPC = pd.concat([dfPC, dfLabel['Activity']], axis = 1)
        return dfPC
    


def plot_pca_2Dscatter(dfPC, XPC, YPC, pca, title='', targets=[1,0], legend = ['Active', 'Inactive'], colors = ['#242F9B', '#F7D716'], filename=''):
    """
    Description
    ---------------------------------------------   
    creates 2D scatter plot for PCA

    Parameters
    ---------------------------------------------
    dfPC:  panda dataframe to add the pca results to
    XPC:  x axis label 
    YPC:  y axis lable
    title:  title for plot
    subtitle:  subtitle for plot
    targets:  label values to plot
    lengend:  legend text for the targets to display on plot
    filename:  filename to save the plot under
    
    Returns
    ---------------------------------------------
    returns image of plot

    Examples
    --------
    plot_pca_2Dscatter(dfPC, XPC, YPC, title, subtitle)
    """

    valDf = valXPC = valYPC = True
    valMsg = "Invalid parameters: "

    #check valid params were sent
    if not isinstance(dfPC, pd.DataFrame):
        valDf = False
        valMsg += "\nInvalid dataframe"
        
    if XPC == "":
        valXPC = False
        valMsg += "\nXPC not provided"

    if YPC == "":
        valYPC = False
        valMsg += "\nYPC not provided"


    #if invalid parameters sent, inform caller
    if (not valDf) or (not valXPC) or (not valYPC):
        print(valMsg)
    else:        
        plt.figure()
        plt.figure(figsize=(10,10))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(XPC, fontsize=16, labelpad=20)
        plt.ylabel(YPC, fontsize=16, labelpad=20)
        plt.xlabel('{}  ({}%)'.format(XPC, round(100*pca.explained_variance_ratio_[0],1)), fontsize=16, labelpad=20)
        plt.ylabel('{}  ({}%)'.format(YPC, round(100*pca.explained_variance_ratio_[1],1)), fontsize=16, labelpad=20)

        if len(title) > 0:
            plt.title(title, y=.99, fontsize=20, pad=20, fontweight='bold', color='#363062')
        #if len(subtitle) > 0:
            #plt.suptitle(subtitle, y=.91, fontsize=14, fontweight='bold', color='#363062')

        for target, color in zip(targets,colors):
            if 'Rank' in dfPC.columns:
                color2use = colors[color]
                indicesToKeep = dfPC['Rank'] == target
                plt.scatter(dfPC.loc[indicesToKeep, XPC]
                    , dfPC.loc[indicesToKeep, YPC], c=color2use, cmap='YlGn', s=25, zorder=2)

                keys_list = list(colors)
                legend1 = mplines.Line2D([0],[0], linestyle="none", c=colors[keys_list[0]], marker = 'o')
                legend2 = mplines.Line2D([0],[0], linestyle="none", c=colors[keys_list[1]], marker = 'o')
                legend3 = mplines.Line2D([0],[0], linestyle="none", c=colors[keys_list[2]], marker = 'o')
                legend = plt.legend([legend1, legend2, legend3], targets, numpoints = 1, loc='upper right', prop={'size': 14}, title='Inhibition')
                legend.get_title().set_fontsize('14')
            else:
                indicesToKeep = dfPC['Activity'] == target
                plt.scatter(dfPC.loc[indicesToKeep, XPC]
                    , dfPC.loc[indicesToKeep, YPC], c=color, cmap='YlGn', s=25, zorder=2)
                plt.legend(legend, loc='upper right', prop={'size': 14})

        #save image, add border and display
        image = f'{filename}.jpg'
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.close()

        add_imageborder(image)
        display(Image(image))



def plot_pca_3Dscatter(dfPC, XPC, YPC, ZPC, pca, title='', targets=[1,0], colors = {1:'#242F9B', 0:'#F7D716'}, filename=''):
    """
    Description
    ---------------------------------------------   
    creates 3D scatter plot for PCA

    Parameters
    ---------------------------------------------
    dfPC:  panda dataframe to add the pca results to
    XPC:  x axis label 
    YPC:  y axis label
    ZPC:  z axis label
    title:  title for plot
    subtitle:  subtitle for plot
    targets:  label values to plot
    lengend:  legend text for the targets to display on plot
    filename:  filename to save the plot under
    
    Returns
    ---------------------------------------------
    returns image of plot

    Examples
    --------
    plot_pca_3Dscatter(dfPC, XPC, YPC, ZPC, title)
    """

    valDf = valXPC = valYPC = valZPC = True
    valMsg = "Invalid parameters: "

    #check valid params were sent
    if not isinstance(dfPC, pd.DataFrame):
        valDf = False
        valMsg += "\nInvalid dataframe"
        
    if XPC == "":
        valXPC = False
        valMsg += "\nXPC not provided"

    if YPC == "":
        valYPC = False
        valMsg += "\nYPC not provided"
    
    if ZPC == "":
        valZPC = False
        valMsg += "\nYPC not provided"


    #if invalid parameters sent, inform caller
    if (not valDf) or (not valXPC) or (not valYPC) or (not valZPC):
        print(valMsg)
    else:  
        fig = plt.figure(figsize=(10, 10))

        #get rid of warning:  https://copyfuture.com/blogs-details/202203030520270316
        axes = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(axes)

        if len(title) > 0:
            axes.set_title(title, fontsize=24, y=1.01, pad=20, fontweight='bold', color='#363062')
        
        axes.set_xlabel('{}  ({}%)'.format(XPC, round(100*pca.explained_variance_ratio_[0],1)), fontsize=16, labelpad=20)
        axes.set_ylabel('{}  ({}%)'.format(YPC, round(100*pca.explained_variance_ratio_[1],1)), fontsize=16, labelpad=20)
        axes.set_zlabel('{}  ({}%)'.format(ZPC, round(100*pca.explained_variance_ratio_[1],1)), fontsize=16, labelpad=20)

        
        if 'Rank' in dfPC.columns:
            axes.scatter(dfPC['PC1'], dfPC['PC2'], dfPC['PC3'], c=dfPC['Rank'].map(colors), s=10)

            keys_list = list(colors)
            legend1 = mplines.Line2D([0],[0], linestyle="none", c=colors[keys_list[0]], marker = 'o')
            legend2 = mplines.Line2D([0],[0], linestyle="none", c=colors[keys_list[1]], marker = 'o')
            legend3 = mplines.Line2D([0],[0], linestyle="none", c=colors[keys_list[2]], marker = 'o')
            legend = axes.legend([legend1, legend2, legend3], targets, numpoints = 1, loc='upper right', prop={'size': 14}, title='Inhibition')
            legend.get_title().set_fontsize('14')
        else:
            axes.scatter(dfPC['PC1'], dfPC['PC2'], dfPC['PC3'], c=dfPC['Activity'].map(colors), s=10)

            scatter1_proxy = mplines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 'o')
            if 0 in colors:
                scatter2_proxy = mplines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 'o')
            if 2 in colors:
                scatter2_proxy = mplines.Line2D([0],[0], linestyle="none", c=colors[2], marker = 'o')
            axes.legend([scatter1_proxy, scatter2_proxy], targets, numpoints = 1, loc='upper right', prop={'size': 14})


        #save image, add border and display
        image = f'{filename}.jpg'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.close()

        #add_imageborder(image)
        display(Image(image))



def get_pca(titles, subtitles, filenames, x_data):
    for title, subtitle, filename, x in zip(titles, subtitles, filenames, x_data):
        title += ':  ' + subtitle

        pca = PCA().fit(x)
        var_ratio = pca.explained_variance_ratio_
        plot_PCAVariance(var_ratio, pca, title=title, filename=filename)

        plot_PCAScree(pca, title, filename) 



def run_pca(pc_nums, titles, filenames, x_data, image_path, data_path, font_path, subtitles=''):
    for title, filename, x_data in zip(titles, filenames, x_data):
        filename_image = f'{image_path}filename'
        run_PCAs(pc_nums, x_data, image_path, data_path, font_path, title=title, filename=filename)



def plot_pca(pc_num, XPC, YPC, ZPC, titles, filenames, x_data, x_data_inhibits, y_data
    , features, legends, targets, image_path):

    for title, filename, x, x_inhibit, y, feature, legend, target \
        in zip(titles, filenames, x_data, x_data_inhibits, y_data, features, legends, targets):
        
        pca = PCA(n_components=pc_num)
        pca_Xtrain = pca.fit_transform(x)
        dfPC = format_pca_data(pca_Xtrain, pc_num, x, y, feature)

        #print(type(X_train_inhibit))
        if not x_inhibit is None:
            #add inhibition column to df
            dfPC['Inhibition'] = x_inhibit.tolist()

            #rank inhibition into bins
            ranking = pd.cut(dfPC['Inhibition'], bins = [89,93,96,120], labels=['90-93','94-96','96-100+'])
            dfPC['Rank'] = ranking

            #set color for ranking
            colors = {'90-93':'#92B4EC', '94-96':'#F7D716', '96-100+': 'indianred'}
            target = ['90-93', '94-96', '96-100+']
        else:
            if pc_num == 2:
                colors = ['#242F9B', '#F7D716']
            else:
                if 0 in np.unique(y).tolist():
                    colors = {1:'#242F9B', 0:'#F7D716'}
                else:
                    colors = {1:'#242F9B', 2:'#F7D716'}


        if pc_num == 3:            
            plot_pca_3Dscatter(dfPC, XPC, YPC, ZPC, pca, title, targets=target, colors=colors, filename=filename)
        else:
            plot_pca_2Dscatter(dfPC, XPC, YPC, pca, title, target, legend, colors, filename)





# UMAP
#-----------------------------------------------------------------------------------------------------
def plot_umap(dfTrain, x, y, filename, path_model, path_data, path_image \
    , classes=['Inactive', 'Active'], class_colours={0:'#8DA0CB', 1:'#FFD92F'} \
    , supervised=True, n_neighbors=15, min_dist=0.1, metrics='euclidean', n_components=2, removeticks=True, title='', subtitle='', save_model=False):
    """
    Description
    ---------------------------------------------   
    plots umap projection

    Parameters
    ---------------------------------------------
    dfTrain:  panda dataframe w/training data
    X_train:  panda dataframe w/training data
    y_train:  labels
    classes:  list of classes
    class_colours:  dictionary of colors for the classes
    supervised:  perform supervised umap
    n_neighbors: nearest neighbors
    min_dist:  min distance
    n_components: # of components
    metric:  metric to apply
    removeticks:  remove ticks on the plot
    title:  title for the plot
    save_model:  save the model
    filename:  filename for the plot
    
    Returns
    ---------------------------------------------
    returns image of plot

    Examples
    --------
    plot_umap_dataprojection(dfTrain, umap_data, title, filename)
    """
    
    valDf = valclasses = valclassColors = True
    valMsg = "Invalid parameters: "
    

    #check valid params were sent
    #check valid params were sent
    if not isinstance(dfTrain, pd.DataFrame):
        valDf = False
        valMsg += "\dfTrain:  not a valid dataframe"

    if not isinstance(x, pd.DataFrame):
        valDf = False
        valMsg += "\X_train:  not a valid dataframe"
   
    if not isinstance(classes, list):
        valclasses = False
        valMsg += "\nclasses:  Invalid data type.  Expected a list."
    
    if not isinstance(class_colours, dict):
        valclassColors = False
        valMsg += "\nclass_colours:  Invalid data type.  Expected a list."
    
    
    #if invalid parameters sent, inform caller
    if (not valDf) or (not valclasses) or (not valclassColors):
        print(valMsg)
    else:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metrics,
            random_state=42
        )

        if supervised:
            umap_data = reducer.fit_transform(x, y)
        else:
            umap_data = reducer.fit_transform(x)            

        #retrieve saved model/results
        #path = os.path.abspath(f'{model_path}{filename}_n{n_neighbors}_d{min_dist}_m{metric}.mdl')
        #reducer = joblib.load(path)
        #umap_data = genfromtxt(f'{model_path}{filename}_n{n_neighbors}_d{min_dist}_m{metric}.csv', delimiter=' ')

        filename = f'{filename}_n{n_neighbors}_d{min_dist}_m{metrics}'
        if save_model:
            #save model and results            
            joblib.dump(reducer, f'{path_model}{filename}.mdl')
            np.savetxt(f'{path_data}{filename}.csv', umap_data, delimiter=" ")

        fig, ax = plt.subplots(1, figsize=(10, 10))

        #create legend
        recs = []
        if n_components == 2:
            plt.gca().set_aspect('equal', 'datalim')
            #if dataset w/only actives

            if len(classes) == 1:
                plt.scatter(
                        umap_data[:, 0],
                        umap_data[:, 1],
                        c=[class_colours[1]],
                        s=25,
                        zorder=2
                        )   
                #create legend
                recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[1]))
            #if combined file
            elif not ('Active' in classes):
                plt.scatter(
                        umap_data[:, 0],
                        umap_data[:, 1],
                        c=[x for x in dfTrain.Activity.map({1:class_colours[1], 2:class_colours[2]})],
                        s=25,
                        zorder=2
                        )  
                #create legend
                for i in range(1, len(class_colours) + 1):
                    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
            #if dataset w/actives & inactives
            else:
                plt.scatter(
                        umap_data[:, 0],
                        umap_data[:, 1],
                        c=[x for x in dfTrain.Activity.map({1:class_colours[1], 0:class_colours[0]})],
                        s=25,
                        zorder=2
                        )  
                #create legend
                for i in range(0, len(class_colours)):
                    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[i]))


        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                umap_data[:,0], 
                umap_data[:,1], 
                umap_data[:,2], 
                c=[x for x in dfTrain.Activity.map({1:class_colours[1], 0:class_colours[0]})], 
                s=25,
                zorder=2)
        
        #remove ticks from x and y axis
        if removeticks and (n_components == 2):
            plt.setp(ax, xticks=[], yticks=[]) 

        #create legend
        plt.legend(recs, classes, loc='upper right', prop={'size': 12})

        # add title    
        if len(subtitle) == 0:
            plt.title(title, fontsize=20, y=1.0, pad=20, fontweight='bold', color='#363062')
        else:
            plt.title(title, y=1.05, pad=10, fontsize=16, fontweight='bold', color='#363062')
            plt.suptitle(subtitle, y=.92, fontsize=14, fontweight='bold', color='#363062')

        #save image, add border and display            
        image = f'{path_image}{filename}.jpg'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.close()

        #add_imageborder(image)
        display(Image(image))



def get_umap(n_neighbors, min_dist, metrics, titles, filenames, x_data, y_data, class_list, class_colours
    , path_image, path_model, path_data):
    
    for title, filename, x, y, classes, class_colour \
        in zip(titles, filenames, x_data, y_data, class_list, class_colours):
        
        # append label to training data for graphs
        dfTrain = x
        dfTrain['Activity'] = y.tolist()
        
        subtitle = f"n_neighbors: {n_neighbors} -  min_dist: {min_dist}  -  metrics: {metrics}"
        plot_umap(dfTrain, x, y, filename, path_model, path_data, path_image
                  , classes=classes, class_colours=class_colour
                  , supervised=False, n_neighbors=n_neighbors, min_dist=min_dist, metrics=metrics
                  , title=title, subtitle=subtitle, save_model=True)
 




# tSNE
#-----------------------------------------------------------------------------------------------------       
def plot_tsne(x_data, filename, path_model, path_data, path_image, pca=None, dfTSNEResults=None
    , classes=['Inactive', 'Active'], class_colours={0:'#8DA0CB', 1:'#FFD92F'}
    , n_components=2, perplexity=30, learning_rate=200, n_iter=1000, verbose=0, random_state=None
    , removeticks=True, title='', subtitle=''
    , save_model=True, getSavedModel=False):
    """
    Description
    ---------------------------------------------   
    plots umap projection

    Parameters
    ---------------------------------------------
    X_train:  panda dataframe w/training data
    classes:  list of classes
    class_colours:  dictionary of colors for the classes
    n_components: Dimension of the embedded space
    perplexity:  number of nearest neighbors
    learning_rate: 
    niter:  Maximum number of iterations for the optimization
    verbose: Verbosity level
    random_state:  random number generator
    removeticks:  remove ticks on the plot
    title:  title for the plot
    save_model:  save the model
    filename:  filename for the plot
    
    Returns
    ---------------------------------------------
    returns image of plot

    Examples
    --------
    plot_umap_dataprojection(dfTrain, umap_data, title, filename)
    """
    
    valDf = valclasses = valclassColors = True
    valMsg = "Invalid parameters: "
    

    #check valid params were sent

    if not isinstance(x_data, pd.DataFrame):
        valDf = False
        valMsg += "\X_train:  not a valid dataframe"
    
    
    #if invalid parameters sent, inform caller
    if (not valDf):
        print(valMsg)
    else:        
        if getSavedModel:
            #retrieve saved model/results
            path = os.path.abspath(f'{path_model}{filename}.mdl')
            tsne_results = joblib.load(path)                
        else:
            #instantiate tsne
            tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, verbose=verbose, random_state=random_state)
            
            #if pca output not passed, used training dataset
            if (pca is None):
                tsne_results = tsne.fit_transform(x_data)
            else:
                #if df to store results is passed in, store the pca results
                if not (dfTSNEResults is None):
                    columns = []
                    for x in range(1, pca.shape[1] + 1):
                        columns.append(f'PC_{x}')                    
                    
                    pca_results = pd.DataFrame(pca,columns=columns)                  
                    dfTSNEResults = pd.concat([dfTSNEResults, pca_results], axis = 1)

                #use to pca output to seed tsne
                tsne_results = tsne.fit_transform(pca)

            if save_model:
                #save model and results            
                joblib.dump(tsne_results, f'{path_model}{filename}.mdl')


        #if df to store results is passed in, store the tsne results
        if not (dfTSNEResults is None):
            dfTSNEResults['tSNE 1'] = pd.Series(tsne_results[:,0])
            dfTSNEResults['tSNE 2'] = pd.Series(tsne_results[:,1])
            dfTSNEResults.to_csv(f'{path_data}{filename}.csv')

        #add results to df
        x_data['tSNE 1'] = tsne_results[:,0]
        x_data['tSNE 2'] = tsne_results[:,1]

        fig, ax = plt.subplots(1, figsize=(10, 10))

        #create legend
        recs = []
        if n_components == 2:
            plt.gca().set_aspect('equal', 'datalim')
            #if dataset w/only actives

            if len(classes) == 1:
                plt.scatter(
                        x_data['tSNE 1'],
                        x_data['tSNE 2'],
                        c=[class_colours[1]],
                        s=25,
                        zorder=2
                        )   
                #create legend
                recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[1]))
            #if combined file
            elif not ('Active' in classes):
                plt.scatter(
                        x_data['tSNE 1'],
                        x_data['tSNE 2'],
                        c=[x for x in x_data.Activity.map({1:class_colours[1], 2:class_colours[2]})],
                        s=25,
                        zorder=2
                        )  
                #create legend
                for i in range(1, len(class_colours) + 1):
                    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
            #if dataset w/actives & inactives
            else:
                plt.scatter(
                        x_data['tSNE 1'],
                        x_data['tSNE 2'],
                        c=[x for x in x_data.Activity.map({1:class_colours[1], 0:class_colours[0]})],
                        s=25,
                        zorder=2
                        )  
                #create legend
                for i in range(0, len(class_colours)):
                    recs.append(mpatch.Rectangle((0, 0), 1, 1, fc=class_colours[i]))


        #remove ticks from x and y axis
        if removeticks and (n_components == 2):
            plt.setp(ax, xticks=[], yticks=[]) 

        #create legend
        plt.legend(recs, classes, loc='upper right', prop={'size': 12})

        # add title    
        if len(subtitle) == 0:
            plt.title(title, fontsize=20, y=1.0, pad=20, fontweight='bold', color='#363062')
        else:
            plt.title(title, y=1.05, pad=10, fontsize=16, fontweight='bold', color='#363062')
            plt.suptitle(subtitle, y=.92, fontsize=14, fontweight='bold', color='#363062')

        #save image, add border and display
        image = f'{path_image}{filename}.jpg'      
        plt.savefig(image, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        plt.close()

        #add_imageborder(image)
        display(Image(image))



def get_tsne(pc_num, perplexity, iterations, learning_rate
    , titles, subtitles, filenames, x_data, y_data, data_smiles, data_inhibit
    , class_list, class_colours, getSavedModels
    , path_image, path_model, path_data):

    for title, subtitle, filename, x, y, SMILES, dfInhibition, classes, class_colour, getSavedModel \
        in zip(titles, subtitles, filenames, x_data, y_data, data_smiles, data_inhibit
               , class_list, class_colours, getSavedModels):
        
        #run pca
        pca = PCA(n_components=pc_num)
        pca_result = pca.fit_transform(x)      

        #add non-feature columns 
        dfTSNEResults = pd.concat([SMILES, dfInhibition], axis = 1)
        dfTSNEResults['Activity'] = y.tolist()
        
        #set params valuable to send to function
        params = {'x_data': x,
            'pca': pca_result, 
            'dfTSNEResults': dfTSNEResults,
            'classes': classes, 
            'class_colours': class_colour, 
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': iterations,
            'getSavedModel': getSavedModel,
            'title': title,
            'subtitle': subtitle,
            'filename': filename,
            'path_model': path_model,
            'path_data': path_data,
            'path_image': path_image}

        # plot tsne
        plot_tsne(**params)