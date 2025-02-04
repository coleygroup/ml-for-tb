import pandas as pd
import numpy as np
import itertools
import sys



def deleteMolWithMissingData(df, deleteMissingData, printStatus, dataset=''):   
    if printStatus:
        print(f'{dataset} Data Info before deletion')
        print('---------------------------------------------')
        print(df.info())

    #remove rows w/NaN
    df = df.dropna()
    rowCount = df.shape[0]

    if printStatus:
        print(f'\n{dataset} Data Info after deletion')
        print('---------------------------------------------')
        print(df.info())
    
    print('\n')

    return df, rowCount



def dupeAnalysis(df, colGroupBy, colInhibition, dataset=''):   
    dupes_Count = df.shape[0]
    print(f'Duplicate count:  {dupes_Count}')

    #get min, max, quartiles, std for dupes on inhibition for a molecule name
    dupes_MolInfo = df.groupby([colGroupBy], as_index=False).agg({colInhibition:['count','min', 'max', 'mean', 'std']})

    #when you use the .agg function, it creates a hierarchy in the header, which complicates how you access columns in a df
    #below code takes care of that by flattening out the Hierarchical Column Indices
    dupes_MolInfo.columns = [
    '_'.join(col).rstrip('_').replace(colInhibition, 'Inhibit') for col in dupes_MolInfo.columns.values
    ]

    #get number of unique dupes
    dupe_UniqueMolCount = dupes_MolInfo.shape[0]
    print("# of unique molecuLes that are duplicated:  " + str(dupe_UniqueMolCount))


    #get how many molecules were duplicated 3 times, 2 times, etc...
    print('---------------------------------------------')
    print(dupes_MolInfo.value_counts(subset = ['Inhibit_count']).rename_axis('Duplicate count').reset_index(name='Molecule Count'))
    print('\n')

    return dupes_MolInfo, dupe_UniqueMolCount



def dedupeByOneColumn(df, columnGroupBy, columnDedupeOn, keep):    
    """
    Description
    ---------------------------------------------   
    Removes duplicates, keeping min or max of the specified column

    Parameters
    ---------------------------------------------
    df:  panda dataframe to dedupe
    columnGroupBy:  name of column in df that has the duplicates
    columnDedupeOn:  name of column in df whose value will determine which dupe to keep
    keep:  value to keep - min or max
    
    Returns
    ---------------------------------------------
    returns the df with duplicates removed

    Examples
    --------
    dataTAACF = cmn.dedupeByOneColumn(dataTAACF, 'MoleculeName', 'MtbH37Rv-Inhibition', 'min')
    """
    
    valDf = True
    valColumnGroupBy = True
    valColumnDedupeOn = True
    valKeep = True
    valMsg = "Invalid parameters: "
    keepFunction = ""
    

    #check valid params were sent
    if not isinstance(df, pd.DataFrame):
        isdf = False
        valMsg += "\nInvalid dataframe"
        
    if columnGroupBy == "":
        valColumnGroupBy = False
        valMsg += "\columnGroupBy:  column with duplicates, not provided"
        
    if columnDedupeOn == "":
        valColumnDedupeOn = False
        valMsg += "\ncolumnDedupeOn:  column whose value will determine which dupe to keep, not provided"

    if keep == "":
        valKeep = False
        valMsg += "\nInvalid value to keep.  Should be min or max."
    else:
        if keep != 'min' and keep != 'max':
            valKeep = False
            valMsg += "\nInvalid value to keep.  Should be min or max."
        else:
            if keep == 'min':
                keepFunction = "first"
            else:
                keepFunction = "last"

        
    #if invalid parameters sent, inform caller
    if (not valDf) or (not valColumnGroupBy) or (not valColumnDedupeOn) or (not valKeep):
        print(valMsg)
    else:  
        try:            
            #get molecule from smiles 
            df = df.sort_values(columnDedupeOn).drop_duplicates(columnGroupBy, keep=keepFunction)
         
        except ValueError as ve:
            raise ve
              
    
    return df 



def mergeDatasets(df1, df2, col_MergeOn, col_df1Inhibition, col_df2Inhibition, col_df1Active, col_df2Active, flag_df1, flag_df2, flag_both):   
    # _merge can be "both", 'left_only', 'right_only'

    #Inner Join Merge:  mols existing in both files, bring over df2 inhibition and active info
    dfInnerJoin = df1.merge(df2[[col_MergeOn, col_df2Inhibition, col_df2Active]], on=col_MergeOn, how="left", indicator=True)    

    # get count of mol that exist in both files
    countBoth = (dfInnerJoin[dfInnerJoin['_merge']=='both']).shape[0]
    countTAACFOnly = (dfInnerJoin[dfInnerJoin['_merge']=='left_only']).shape[0]
    countSRIKinase = (dfInnerJoin[dfInnerJoin['_merge']=='right_only']).shape[0]
    countTAACF = df1.shape[0]

    if countTAACF != (countBoth + countTAACFOnly + countSRIKinase):
        print("ERROR: Inner Join merge count & original TAACF file count do not match.")


    #Right Outer Join Merge:  mols existing only in SRIKinase, bring over their inhibition and active info
    dfRightJoin = df2.merge(df1[[col_MergeOn, col_df1Inhibition, col_df1Active]], on=col_MergeOn, how="left", indicator=True).query('_merge == "left_only"')

    # get count of mol that exist in both files
    countSRIKinaseONLY = (dfRightJoin[dfRightJoin['_merge']=='left_only']).shape[0]
    countSRIKinase = df2.shape[0]

    if countSRIKinase != (countBoth + countSRIKinaseONLY):
        print("ERROR: Right Join merge count & original SRIKinase file count do not match.")


    # reorder columns prior to final merge, and flag rows that came from second dataset
    column_names = list(dfInnerJoin.columns.values)
    dfRightJoin = dfRightJoin.reindex(columns=column_names)    
    
    #change datatype of _merge from category to string, so it can be updateable
    dfInnerJoin = dfInnerJoin.astype({"_merge": str})
    dfInnerJoin['_merge'].unique()

    #flag all rows with the dataset they came from
    dfRightJoin._merge = flag_df2
    dfInnerJoin['_merge'].loc[(dfInnerJoin['_merge'] == 'both')] = flag_both
    dfInnerJoin['_merge'].loc[(dfInnerJoin['_merge'] == 'left_only')] = flag_df1

    #combine the files
    lstDataFrames = [dfInnerJoin, dfRightJoin]
    dfMerged = pd.concat(lstDataFrames)
    countMerged = dfMerged.shape[0]
    
    print(f"\nMolecules in merged files:")
    print('------------------------------------------------------------')
    print("Molecules on both files:  " + str(countBoth))
    print("Molecules just in TAACF not in SRIKinase:  " + str(countTAACFOnly))
    print("Molecules just in SRIKinase not in TAAACF:  " + str(countSRIKinaseONLY))
    print("Molecules in merged df:  " + str(countMerged))

    if countMerged == (countSRIKinaseONLY + countTAACFOnly + countBoth):
        print("SUCCESS:  Merged file has all rows")
    else:
        print("ERROR: Merged file doesn't equal distinct molecules in each file, plus the shared molecules in both files")


    #Add activity column, set to 1 if either file has Active columns equal to 1
    dfMerged["Activity"] = (dfMerged[col_df1Active] == 1) | (dfMerged[col_df2Active] == 1).astype(int)
    
    taacf_inactive = (dfMerged[col_df1Active]==0).sum()
    taacf_active = (dfMerged[col_df1Active]==1).sum()
    sri_inactive = (dfMerged[col_df2Active]==0).sum()
    sri_active = (dfMerged[col_df2Active]==1).sum()
    active = (dfMerged['Activity']==1).sum()
    inactive = (dfMerged['Activity']==0).sum()

    print(f"\nActives in merged files:")
    print('------------------------------------------------------------')
    print("Merged df actives:  " + str(active))
    print("Merged df inactives:  " + str(inactive))

    if countMerged == ((taacf_inactive + taacf_active + sri_inactive + sri_active) - countBoth):
        print("SUCCESS:  Merged file Activity column added")
    else:
        print("ERROR: Merged file actives/inactives do not add up")


    #replace NaN to zero
    dfMerged[col_df1Inhibition] = dfMerged[col_df1Inhibition].fillna(0)
    dfMerged[col_df1Active] = dfMerged[col_df1Active].fillna(0)
    dfMerged[col_df2Inhibition] = dfMerged[col_df2Inhibition].fillna(0)
    dfMerged[col_df2Active] = dfMerged[col_df2Active].fillna(0)

    #convert active columns to integers
    dfMerged[col_df1Active] = dfMerged[col_df1Active].astype(int)
    dfMerged[col_df2Active] = dfMerged[col_df2Active].astype(int)
    dfMerged['Activity'] = dfMerged['Activity'].astype(int)

    return dfMerged