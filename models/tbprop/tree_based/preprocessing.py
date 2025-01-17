import warnings
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def calc_high_corr_feats(df, remove_cols, threshold=0.9):
    reduced_df = df.drop(remove_cols, axis=1)
    corr_matrix = reduced_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop


def calc_low_variance_feats(df, remove_cols, threshold=0.2):
    selector = VarianceThreshold(threshold)
    reduced_df = df.drop(remove_cols, axis=1)
    try:
        selector.fit(reduced_df)
    except:
        warnings.warn("No features below required variance threshold.")
        return []

    return list(reduced_df.columns[selector.get_support(indices=True)])


def scale_columns_no_split(df, remove_cols, scaling='standard'):
    reduced_df = df.drop(remove_cols, axis=1)
    remove_df = df[remove_cols]
    
    if scaling == 'standard':
        scaler = StandardScaler()
    elif scaling == 'min_max':
        scaler = MinMaxScaler()
    elif scaling in ('minmax_scale', 'robust', 'power'):
        raise NotImplementedError(f"Scaling method '{scaling}' not implemented yet.")
    else:
        raise ValueError(f"Invalid value for `scaling`: '{scaling}'.")
    
    scale_cols = reduced_df.columns
    scaled_values = scaler.fit_transform(reduced_df)
    reduced_df = pd.DataFrame(scaled_values, columns=scale_cols)
    
    scaled_df = pd.concat([remove_df, reduced_df], axis=1)
    
    return scaled_df


def scale_columns(df_train, df_test, remove_cols, scaling='standard'):
    if scaling == 'standard':
        scaler = StandardScaler()
    elif scaling == 'min_max':
        scaler = MinMaxScaler()
    elif scaling in ('minmax_scale', 'robust', 'power'):
        raise NotImplementedError(f"Scaling method '{scaling}' not implemented yet.")
    else:
        raise ValueError(f"Invalid value for `scaling`: '{scaling}'.")
    
    # print(f"remove_cols = {remove_cols}")

    # Remove unnecessary columns from train and test sets
    reduced_df_trn = df_train.drop(remove_cols, axis=1)
    remove_df_trn = df_train[remove_cols]

    # print(f"\nScaler = {scaler}")
    # print(f"reduced_df_trn shape = {reduced_df_trn.shape}")

    reduced_df_tst = df_test.drop(remove_cols, axis=1)
    remove_df_tst = df_test[remove_cols]
    
    scale_cols = reduced_df_trn.columns
    
    # Fit scaler on training set
    scaler.fit(reduced_df_trn)
    
    # Scale train and test set values
    scaled_values_trn = scaler.transform(reduced_df_trn)
    scaled_values_tst = scaler.transform(reduced_df_tst)
    
    reduced_df_trn = pd.DataFrame(scaled_values_trn, columns=scale_cols)
    reduced_df_tst = pd.DataFrame(scaled_values_tst, columns=scale_cols)
    
    # Merge with removed columns
    scaled_df_trn = pd.concat([remove_df_trn, reduced_df_trn], axis=1)
    scaled_df_tst = pd.concat([remove_df_tst, reduced_df_tst], axis=1)
    
    return scaled_df_trn, scaled_df_tst, scaler
    

def reduce_dims(X_train, X_test, n_components=32):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test


def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n >= 22 and n <= 29) or (n >= 40 and n <= 47) or (n >= 72 and n <= 79)

def contains_transition_metal(mol):
    return any([is_transition_metal(at) for at in Chem.MolFromSmiles(mol, sanitize=False).GetAtoms()])

def standardize(smiles):
    """
    Follows the steps in
    https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    as described **excellently** (by Greg) in https://www.youtube.com/watch?v=eWTApNX8dJQ
    
    Borrowed from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        clean_mol = rdMolStandardize.Cleanup(mol)

        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenient method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    
    return np.nan


class VarianceThresholdPandas(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.16):
        self.threshold = threshold
        self.selector = VarianceThreshold(self.threshold)

    def fit(self, X, y=None):
        try:
            self.selector.fit(X)
        except:
            self.to_drop_ = []
        self.to_drop_ = list(X.columns[self.selector.get_support(indices=True)])
        return self
    
    def transform(self, X):
        return X.drop(self.to_drop_, axis=1)


class CorrelationThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] >= self.threshold)]
        return self

    def transform(self, X):
        return X.drop(self.to_drop_, axis=1)


class FeatureImportanceScoreThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0., classifier='XGBoost', n_folds=5, model_config=None):
        self.threshold = threshold
        self.classifier = classifier
        self.n_folds = n_folds
        self.model_config = model_config

        assert self.classifier in ('XGBoost', 'LightGBM', 'CatBoost', 'RandomForest'), \
            f"Classifier '{self.classifier}' not recognized!"

    def _init_model(self):
        init_handle_config = lambda f: f(**self.model_config) if self.model_config else f()

        if self.classifier == 'XGBoost':
            return init_handle_config(XGBClassifier)
        elif self.classifier == 'LightGBM':
            return init_handle_config(LGBMClassifier)
        elif self.classifier == 'CatBoost':
            return init_handle_config(CatBoostClassifier)
        elif self.classifier == 'RandomForest':
            return init_handle_config(RandomForestClassifier)
        else:
            raise ValueError(f"Classifier '{self.classifier}' not recognized!")

    def fit(self, X, y):
        estimator = self._init_model()
        res = cross_validate(estimator, X, y, cv=self.n_folds, return_estimator=True)
        mean_imps = np.array([x.feature_importances_ for x in res['estimator']]).mean(axis=0)
        self.to_drop_ = X.columns[np.where(mean_imps <= self.threshold)]
        return self
    
    def transform(self, X):
        return X.drop(self.to_drop_, axis=1)
    

class StandardScalerPandas(BaseEstimator, TransformerMixin):
    def __init__(self,):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.X_cols_ = X.columns
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=self.X_cols_)
