import pandas as pd
import numpy as np
import itertools
import sys


from sklearn.feature_selection import VarianceThreshold


#rdkit toolkit for similarity search on molecules
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import SaltRemover
from rdkit import DataStructs


from statsmodels.stats.outliers_influence import variance_inflation_factor



# rdkit functions
#-----------------------------------------------------------------------------------------------------
def rdkitMolFromSmiles(df, columnSmiles, column2update):    
    """
    Description
    ---------------------------------------------   
    Iterates through dataframe, reads the smiles string column and gets the molecule from rdkit. 
    If the smiles strain is valid, a molecule is returned, and the dataframe is updated with the molecule.
    Invalid smiles with errors are not returned

    Parameters
    ---------------------------------------------
    df:  panda dataframe to iterate through
    columnSmiles:  name of column in df that has the smiles string to check
    column2update:  name of column in df that will store the molecule info
    
    Returns
    ---------------------------------------------
    returns the df updated with all molecules found

    Examples
    --------
    dataTAACF_validSMILES = cmn.rdkitMolFromSmiles(dataTAACF, 'SMILES', 'molFromSmiles')
    """
    
    valDf = True
    valColumnSmiles = True
    valColumn2Update = True
    valMsg = "Invalid parameters: "
    
    #check valid params were sent
    if not isinstance(df, pd.DataFrame):
        isdf = False
        valMsg += "\nInvalid dataframe"
        
    if columnSmiles == "":
        valColumnSmiles = False
        valMsg += "\ncolumnSmiles:  column with Smiles, not provided"
        
    if column2update  == "" :
        valColumn2Update = False
        valMsg += "\ncolumn2update:  column to update with molecule info, not provided"

        
        
    #if invalid parameters sent, inform caller
    if (not valDf) or (not valColumnSmiles) or (not valColumn2Update):
        print(valMsg)
    else:  
        try:            
            #get molecule from smiles 
            for index, row in df.iterrows():
                molecule = Chem.MolFromSmiles(row[columnSmiles])

                #update dataframe if found
                if molecule is not None:
                    df.at[index,column2update] = molecule

            #return only those molecules that were found
            df = df[df[column2update] != None]             
        except ValueError as ve:
            raise ve
    
    return df 



def getSalts2Strip(filePath='salts_default.txt'):
    """
    Description
    ---------------------------------------------   
    Reads a text file containing salts to strip from a molecule

    Parameters
    ---------------------------------------------
    filePath -  string, path to file. if not passed, uses default text file 

    Returns
     ---------------------------------------------
    list - list of salts to process, along with their types
           list example:  salt = ["[Cl,Br,I]", "smarts"]

    Examples
    --------
    fileSequence = utilities.readFASTAFile(fileFasta, False)  
    """

    salts = []

    #open file in read mode
    with open(filePath, 'r') as fileSalts:
         #for each line that is read
        for line in fileSalts:
            #only process lines that don't have comments or aren't empty
            if  (not line.startswith('//')) and (len(line.strip())>0):
                #split line by delimeter into a list
                lineList = line.strip().split('~')

                #remove beginning/ending whitespace
                saltList = lineList[0].strip()               
                saltPattern = lineList[1].strip()

                #add processed salts to final list
                salt = [saltList, saltPattern]
                salts.append(salt)

        return salts



def rdkitStripSaltFromMol(df, columnMol, columnStripped, columnSaltsDeleted='', removeAllSalts=True, saveDeleted=False,filePath='salts_default.txt'):    
    """
    Description
    ---------------------------------------------   
    Iterates through dataframe, reads the smiles string column and strips the molecule using rdkit. 
    
    Parameters
    ---------------------------------------------
    df:  panda dataframe to iterate through
    columnMol:  name of column in df that has the molecule to strip
    columnStripped:  name of column in df that will store the stripped molecule
    columnSaltsDeleted: name of column to store delted salts
    removeAllSalts:  True if rdkit should remove all salts, False if you want to leave the last salt
    saveDeleted:  save deleted fragment
    filePath:  text file containing list of salts to strip from the molecule, if not passed, default is used   
    
    Returns
    ---------------------------------------------
    returns the df updated with stripped molecule and salts deleted if requested

    Examples
    --------
    rdkitStripSaltFromMol(dataTAACF, 'molFromSmiles', 'molStripped', 'strippedSalts', salts2Strip, False, True)
    """
    
    valDf = True
    valColumnMol = True
    valColumnStripped = True
    valColumnSaltsDeleted = True
    valMsg = "Invalid parameters: "

  
    #check valid params were sent
    if not isinstance(df, pd.DataFrame):
        valDf = False
        valMsg += "\nInvalid dataframe"
        
    if columnMol == "":
        valColumnMol = False
        valMsg += "\ncolumnMol:  column with molecule to strip, not provided"
        
    if columnStripped  == "" :
        valColumnStripped = False
        valMsg += "\ncolumnStripped:  column to update with stripped mol, not provided"

    if saveDeleted and valColumnSaltsDeleted == "":
        valColumnSaltsDeleted = False
        valMsg += "\ncolumnSaltsDeleted:  column to store deleted salts, not provided"

        
        
    #if invalid parameters sent, inform caller
    if (not valDf) or (not valColumnMol) or (not valColumnStripped) or (not valColumnSaltsDeleted):
        print(valMsg)
    else:  
        try:   
            #get salts to strip
            salts = getSalts2Strip(filePath)

            #iterate through each row of the df 
            for index, row in df.iterrows():                
                #blank out working variables
                deletedFragments = []
                deleted = ""
                res = None
                molecule = row[columnMol]
                
                #iterate through list of salts
                for i in range(len(salts)):
                    saltList = salts[i][0]
                    saltFormat = salts[i][1]
                    
                    #instantiate the salt remove
                    remover = SaltRemover.SaltRemover(defnData=saltList, defnFormat=saltFormat)

                    #if saving the stripped molecules and deleted fragments
                    if saveDeleted:
                        res, deleted = remover.StripMolWithDeleted(molecule, dontRemoveEverything=(~removeAllSalts))
                        if len(deleted) > 0:
                            deletedFragments.append([Chem.MolToSmarts(m) for m in deleted])
                    #if saving just the stripped molecule
                    else:
                        res = remover.StripMol(molecule, dontRemoveEverything=(~removeAllSalts))

                    molecule = res
                    
                #get the stripped molecule
                strippedMol = Chem.MolToSmiles(res)

                #update dataframe if found
                if res is not None:
                    df.at[index,columnStripped] = strippedMol
                    if saveDeleted and len(deletedFragments) > 0:
                        df.at[index,columnSaltsDeleted] = (mol for sublist in deletedFragments for mol in sublist)        
        
        except ValueError as ve:
            raise ve  
    
    return df 