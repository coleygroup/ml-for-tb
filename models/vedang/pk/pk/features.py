import warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from pk.preprocessing import calc_high_corr_feats, calc_low_variance_feats, scale_columns
from pk.model_selection import CrossValidator

from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


def morgan_fingerprint(df, smiles_col='mol', n_bits=4096):
    if any([col.startswith('mfp_') for col in df.columns]):
        return df
    
    def smiles_to_morgan_fingerprint(smile_rep, nBits=4096):
        fp = Chem.MolFromSmiles(smile_rep)
        morgan_fp = GetMorganFingerprintAsBitVect(fp, radius=2, nBits=nBits)
        fp_array = np.zeros((1,))
        ConvertToNumpyArray(morgan_fp, fp_array)
        return fp_array
    
    df['morgan_fp'] = df[smiles_col].apply(lambda x: smiles_to_morgan_fingerprint(x, nBits=n_bits))
    fp_matrix = np.stack(df['morgan_fp'].values)
    
    headers = ['mfp_' + str(x+1) for x in np.arange(fp_matrix.shape[1])]
    df_fp = pd.DataFrame(fp_matrix, columns=headers)
    
    df.drop(['morgan_fp'], axis=1, inplace=True)
    df = pd.concat([df, df_fp], axis=1)
    
    return df


class FeatureEngineeringPipeline:
    
    def __init__(self, corr_thresh, var_thresh, fp_n_bits, smiles_col='mol', 
                 oversampling_ratio=0., undersampling_ratio=0., random_state=42, verbose=0):
        self.corr_thresh = corr_thresh
        self.var_thresh = var_thresh
        self.fp_n_bits = fp_n_bits
        self.smiles_col = smiles_col
        
        self.ycol = None
        self.scaler = None
        self.cross_validator = None  # Not really used during inference
        self.final_col_set = None
        self.is_fit = False
        self.over_sampler = None
        self.oversampling_ratio = oversampling_ratio
        self.undersampling_ratio = undersampling_ratio

        self.random_state = random_state
        self.verbose = verbose
        
    def fit_transform(self, df_train, df_test, remove_cols, ycol='AUC_bin'):
        warnings.filterwarnings('ignore')
        
        # Create a copy to avoid changing original data
        df_trn, df_tst = df_train.copy(), df_test.copy()
    
        # Remove high correlation features
        high_corr_feats = calc_high_corr_feats(df_trn, 
                                               remove_cols, 
                                               threshold=self.corr_thresh)
        remove_cols.update(high_corr_feats)

        # Remove low variance features
        low_var_feats = calc_low_variance_feats(df_trn,
                                                remove_cols,
                                                threshold=self.var_thresh)
        remove_cols.update(low_var_feats)

        # Scale columns
        df_train_scaled, df_test_scaled, scaler = scale_columns(df_trn, 
                                                                df_tst, 
                                                                list(remove_cols))

        # Add morgan fingerprints
        df_train_scaled = morgan_fingerprint(df_train_scaled, smiles_col=self.smiles_col, n_bits=self.fp_n_bits)
        df_test_scaled = morgan_fingerprint(df_test_scaled, smiles_col=self.smiles_col, n_bits=self.fp_n_bits)

        # Create train and test sets
        X_train, y_train = df_train_scaled.drop(remove_cols, axis=1), df_train_scaled[ycol]
        X_test, y_test = df_test_scaled.drop(remove_cols, axis=1), df_test_scaled[ycol]

        # Remove zero importance features
        cv = CrossValidator()
        cv.fit(X_train, y_train)
        
        remove_cols.update(cv.zero_imp_feats())
        X_train.drop(cv.zero_imp_feats(), axis=1, inplace=True)
        X_test.drop(cv.zero_imp_feats(), axis=1, inplace=True)

        if self.verbose:
            print(f"\nClass dist. before resampling: {np.unique(y_train, return_counts=True)[1]}.")

        # Majority undersampling
        sampling_strategy = 'auto' if self.undersampling_ratio == 0. else self.undersampling_ratio
        under_sampler = TomekLinks()
        X_train, y_train = under_sampler.fit_resample(X_train, y_train)
        self.under_sampler = under_sampler

        # Minority oversampling
        # sampling_strategy = 'auto' if self.oversampling_ratio == 0. else self.oversampling_ratio
        # over_sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=self.random_state)
        # X_train, y_train = over_sampler.fit_resample(X_train, y_train)
        # self.over_sampler = over_sampler

        if self.verbose:
            print(f"Class dist. after resampling: {np.unique(y_train, return_counts=True)[1]}.")

        # Set the is_fit flag
        self.is_fit = True
        
        # Save objects to memory
        self.ycol = ycol
        self.scaler = scaler
        self.cross_validator = cv
        self.final_col_set = set(X_train.columns)

        return X_train, y_train, X_test, y_test, remove_cols
    
    def transform(self, df_infer):
        if not self.is_fit:
            raise ValueError(f"Object is not fit yet!")
            
        if self.ycol not in df_infer.columns:
            warnings.warn(f"Label column, {self.ycol} not present in inference set.")
            
        # Copy data to avoid changing it
        df = df_infer.copy()
        
        # Add morgan fingerprint
        df = morgan_fingerprint(df, smiles_col=self.smiles_col, n_bits=self.fp_n_bits)
        
        # Check if all columns are present
        if self.final_col_set - set(df.columns):
            raise ValueError(f"Inference set doesn't have the following columns: {self.final_col_set - set(df_infer.columns)}.")
        
        to_scale = set(self.scaler.feature_names_in_).intersection(self.final_col_set)
        scale_extra = set(self.scaler.feature_names_in_) - self.final_col_set
        
        df_scale, df_not_scale = df[to_scale], df[self.final_col_set - to_scale]
        
        for col in scale_extra:
            df_scale[col] = 0.
            
        scale_cols = df_scale.columns
        
        df_scaled = self.scaler.transform(df_scale)
        df_scaled = pd.DataFrame(df_scaled, columns=scale_cols)
        
        df_scaled.drop(scale_extra, axis=1, inplace=True)
        
        df_combined = pd.concat([df_scaled, df_not_scale], axis=1)
        
        return df_combined