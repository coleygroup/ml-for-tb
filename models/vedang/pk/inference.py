import pickle
import warnings
import argparse
import pandas as pd

from pk.preprocessing import calc_high_corr_feats, calc_low_variance_feats, scale_columns
from pk.features import morgan_fingerprint
from pk.model_selection import CrossValidator

MODELS_PATH = 'ckpts/pk_all_p2_models.pkl'
FEATURE_ENGG_PIPELINE_PATH = 'ckpts/pk_feature_engineering_pipeline.pkl'

class FeatureEngineeringPipeline:
    
    def __init__(self, corr_thresh, var_thresh, fp_n_bits, smiles_col='SMILES'):
        self.corr_thresh = corr_thresh
        self.var_thresh = var_thresh
        self.fp_n_bits = fp_n_bits
        self.smiles_col = smiles_col
        
        self.ycol = None
        self.scaler = None
        self.cross_validator = None  # Not really used during inference
        self.final_col_set = None
        self.is_fit = False
        
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
        
        # Set the is_fit flag
        self.is_fit = True
        
        # Save objects to memory
        self.ycol = ycol
        self.scaler = scaler
        self.cross_validator = cv
        self.final_col_set = set(X_train.columns)

        return X_train, y_train, X_test, y_test, remove_cols
    
    def transform(self, df_infer, smiles_col):
        if not self.is_fit:
            raise ValueError(f"Object is not fit yet!")
            
        if self.ycol not in df_infer.columns:
            warnings.warn(f"Label column, {self.ycol} not present in inference set.")
        
        # Copy data to avoid changing it
        df = df_infer.copy()
        
        # Add morgan fingerprint
        df = morgan_fingerprint(df, smiles_col=smiles_col, n_bits=self.fp_n_bits)
        
        # Check if all columns are present
        if self.final_col_set - set(df.columns):
            raise ValueError(f"Inference set doesn't have the following columns: {self.final_col_set - set(df_infer.columns)}.")
        
        to_scale = set(self.scaler.feature_names_in_).intersection(self.final_col_set)
        scale_extra = set(self.scaler.feature_names_in_) - self.final_col_set
        
        df_scale, df_not_scale = df[to_scale], df[self.final_col_set - to_scale]
        
        for col in scale_extra:
            df_scale[col] = 0.
            
        scale_cols = df_scale.columns
        
        df_scaled = self.scaler.transform(df_scale[self.scaler.feature_names_in_])
        df_scaled = pd.DataFrame(df_scaled, columns=scale_cols)
        
        df_scaled.drop(scale_extra, axis=1, inplace=True)
        
        df_combined = pd.concat([df_scaled, df_not_scale], axis=1)
        
        return df_combined 

def get_args():
    parser = argparse.ArgumentParser(description='Inference script for PK')
    parser.add_argument('--infer_path', type=str, default='data/sample_inference_set.csv',  
                        help='Path to inference file (CSV)')
    parser.add_argument('--output_path', type=str, help='Path to output file')
    parser.add_argument('--model', type=str, default='RandomForestClassifier/optuna/P2', help='Path to model file')
    parser.add_argument('--smiles_col', type=str, help='Name of column containing SMILES strings')
    parser.add_argument('--verbose', action='store_true', help='Prints out logs')

    args = parser.parse_args()
    
    return args

def main():
    args = get_args()

    # Load models and pipeline
    if args.verbose: print(f"Loading models and pipeline...", end="")
    with open(FEATURE_ENGG_PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    with open(MODELS_PATH, 'rb') as f:
        all_models = pickle.load(f)
    if args.verbose: print(f"Done.")

    # Load inference data
    if args.verbose: print(f"Loading inference data...", end="")
    df_infer = pd.read_csv(args.infer_path)
    if args.verbose: print(f"Done.")

    # Perform inference
    if args.verbose: print(f"Performing inference using model {args.model}...", end="")
    model = all_models['trained_models'][args.model]
    X_infer = pipeline.transform(df_infer, smiles_col=args.smiles_col)
    inference_probs = model.predict_proba(X_infer[model.feature_names_in_])[:,1]
    if args.verbose: print(f"Done.")

    # Save inference results
    if args.verbose: print(f"Saving inference results...", end="")
    df_out = pd.DataFrame({'SMILES': df_infer[args.smiles_col], 
                           'model_probs': inference_probs})
    df_out.to_csv(args.output_path, index=False)
    if args.verbose: print(f"Done.")

if __name__ == '__main__':
    main()
