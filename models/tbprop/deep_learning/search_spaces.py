def build_neural_params(model_name: str, trial, dataset: str = None):
    """ Creates a new params dict for Optuna. """

    if model_name == 'FCNN':
        params = {
            'embed_size': trial.suggest_categorical('embed_size', [124, 68, 32]),
            'fc_num_layers': trial.suggest_int('fc_num_layers', 1, 3),
            'dropout_rate': trial.suggest_float('dropout_rate', 0., 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, 5e-1, log=True)
        }
    elif model_name == 'CNN':
        params = {
            'conv1_dim': trial.suggest_categorical('conv1_dim', [124, 64, 32]),
            'conv2_dim': trial.suggest_categorical('conv2_dim', [124, 64, 32]),
            'normalize': trial.suggest_categorical('normalize', [True, False]),
            'embed_size': trial.suggest_categorical('embed_size', [124, 68, 32]),
            'fc_num_layers': trial.suggest_int('fc_num_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0., 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, 5e-1, log=True)
        }
    elif model_name == 'LSTM':
        params = {
            'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [128, 64, 32]),
            'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
            'embed_size': trial.suggest_categorical('embed_size', [124, 68, 32]),
            'fc_num_layers': trial.suggest_int('fc_num_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0., 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        }
    elif model_name == 'ChemBERTa-v2':
        params = {
            'fc_num_layers': trial.suggest_int('n_layers', 1, 3),
            'learning_rate': trial.suggest_float('learning_rate', 5e-3, 5e-1, log=True),
        }
    elif model_name == 'MFBERT':
        raise NotImplementedError()
    else:
        raise ValueError(f"Model name '{model_name}' not recognized.")
        
    pos_weight_limit = 2 if dataset == "pk" else 50
    params['pos_weight'] = trial.suggest_float('pos_weight', 1, pos_weight_limit)

    return params
