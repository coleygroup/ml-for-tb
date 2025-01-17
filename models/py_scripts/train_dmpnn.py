import os
import json
import errno
import chemprop
import numpy as np
import pandas as pd

print(f"Using chemprop version {chemprop.__version__}")

from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from typing import Dict, List


class Args:
    def __init__(self):
        self.data_dir = "../data"
        self.dataset = "taacf"
        self.smiles_col = "smiles"
        self.target_col = "auc_bin" if self.dataset == "pk" else "inhibition_bin"
        self.use_gpu = True
        self.train_batch_size = 1048
        self.infer_batch_size = 1048


def make_dmpnn_dataset(args):
    df_trn = pd.read_csv(f"{args.data_dir}/{args.dataset}/{args.dataset}_trn.csv")
    df_val = pd.read_csv(f"{args.data_dir}/{args.dataset}/{args.dataset}_val.csv")
    df_tst = pd.read_csv(f"{args.data_dir}/{args.dataset}/{args.dataset}_tst.csv")

    df_trn_ = df_trn[[args.smiles_col, args.target_col]]
    df_val_ = df_val[[args.smiles_col, args.target_col]]
    df_tst_ = df_tst[[args.smiles_col, args.target_col]]

    df_trn_val_ = pd.concat([df_trn_, df_val_])

    df_trn_val_.to_csv(f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_trn.csv")
    df_tst_.to_csv(f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_tst.csv")


def max_f1_score(y_true, y_score):
    return max(
        [f1_score(y_true, (np.array(y_score) > t).astype(int)) for t in np.arange(0.0, 1.01, 0.02)]
    )


def make_params(params_dict: Dict[str, str]) -> List[str]:
    params_list = []
    for k, v in params_dict.items():
        params_list.append(f"--{k}")
        if v != "None":
            params_list.append(v)
    return params_list


def load_mpnn_config(args):

    config_path = f"{args.data_dir}/configs/{args.dataset}_D-MPNN_config.json"

    optim_data_path = f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_trn.csv"
    optim_config_save_path = f"{args.data_dir}/configs/{args.dataset}_D-MPNN_config_new.json"

    train_data_path = optim_data_path
    train_ckpt_dir = f"{args.data_dir}/checkpoints"

    test_data_path = f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_tst.csv"
    test_preds_path = f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_tst_preds.csv"
    test_ckpt_dir = train_ckpt_dir

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path)

    config["optim"]["data_path"] = optim_data_path
    config["optim"]["config_save_path"] = optim_config_save_path

    config["train"]["data_path"] = train_data_path
    config["train"]["save_dir"] = train_ckpt_dir

    config["test"]["test_path"] = test_data_path
    config["test"]["preds_path"] = test_preds_path
    config["test"]["checkpoint_dir"] = test_ckpt_dir

    for k in ["optim", "train", "test"]:
        config[k]["smiles_column"] = args.smiles_col
        if k != "test":
            config[k]["target_columns"] = args.target_col

    config["train"]["batch_size"] = str(args.train_batch_size)
    config["test"]["batch_size"] = str(args.infer_batch_size)

    # config['train']['num_workers'] = '7'
    # config['test']['num_workers'] = '7'

    return config


def test_evaluation(y_true, y_preds):
    print(f"AUROC = {round(roc_auc_score(y_true, y_preds), 5)*100}")
    print(f"AP = {round(average_precision_score(y_true, y_preds), 5)*100}")
    print(f"Max. F1 score = {round(max_f1_score(y_true, y_preds), 5)*100}")


def main():
    args = Args()

    if not os.path.exists(f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_trn.csv"):
        print("Making dataset.")
        make_dmpnn_dataset(args)
        print("Done.")
    else:
        print(f"Dataset already exists. Skipping.")

    params = load_mpnn_config(args)
    trn_params = make_params(params["train"])
    tst_params = make_params(params["test"])

    trn_args = chemprop.args.TrainArgs().parse_args(trn_params)
    mean_score, std_score = chemprop.train.cross_validate(
        args=trn_args, train_func=chemprop.train.run_training
    )

    tst_args = chemprop.args.PredictArgs().parse_args(tst_params)
    y_preds = np.array(chemprop.train.make_predictions(args=tst_args)).reshape(-1)
    y_true = pd.read_csv(f"{args.data_dir}/{args.dataset}/{args.dataset}_dmpnn_tst.csv")[
        args.target_col
    ].values

    test_evaluation(y_true=y_true, y_preds=y_preds)


if __name__ == "__main__":
    main()
