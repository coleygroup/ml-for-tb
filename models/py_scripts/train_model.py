import os
import time
import json
import errno
import warnings
import argparse
import numpy as np

import torch

from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_curve

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from tbprop.deep_learning.datasets import SmilesDataModule
from tbprop.deep_learning.models import init_neural_model


def setup():
    parser = argparse.ArgumentParser()

    exp_group = parser.add_argument_group("Experiment arguments")

    exp_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy, pytorch, and pytorch lightning.",
    )
    exp_group.add_argument(
        "--data_dir", type=str, default="./data", help="Location where all data is saved."
    )
    exp_group.add_argument(
        "--dataset",
        type=str,
        default="taacf",
        choices=["pk", "taacf", "mlsmr"],
        help="Name of dataset.",
    )
    exp_group.add_argument(
        "--target_col", type=str, default="inhibition_bin", help="Name of column containing labels."
    )
    exp_group.add_argument(
        "--smiles_col", type=str, default="smiles", help="Name of column containing SMILES strings."
    )
    exp_group.add_argument(
        "--hf_home",
        type=str,
        default="data/transformers_cache",
        help="Downloaded transformer models from HF saved here.",
    )
    exp_group.add_argument(
        "--find_threshold",
        type=str,
        default="f1",
        choices=["none", "f1", "j"],
        help="Finds threshold that maximizes the given metric on val set.",
    )
    exp_group.add_argument(
        "--tb_inc_timestamp",
        action="store_true",
        help="Whether to include timestamp in tensorboard study name.",
    )
    exp_group.add_argument(
        "--use_best_config", action="store_true", help="Uses best config saved at data_dir/configs"
    )
    exp_group.add_argument(
        "--use_saved_model",
        action="store_true",
        help="Uses a saved checkpoint, overrides best config flag.",
    )
    exp_group.add_argument("--verbose", action="store_true")

    model_group = parser.add_argument_group("Model training arguments")

    model_group.add_argument(
        "--model_name",
        type=str,
        default="FCNN",
        choices=["FCNN", "CNN", "LSTM", "ChemBERTa-v2"],
        help="Name of model.",
    )
    model_group.add_argument(
        "--n_devices", type=int, default=1, help="n_devices to train model on."
    )
    model_group.add_argument(
        "--num_workers", type=int, default=7, help="num_workers for torch dataloaders."
    )
    model_group.add_argument(
        "--patience", type=int, default=10, help="No. of epochs to wait for early stopping."
    )
    model_group.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=512,
        help="Pad/truncate sequences to this length.",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "gpu"],
        help="Device to train on.",
    )
    model_group.add_argument("--n_epochs", type=int, default=10, help="No. of epochs to train for.")
    model_group.add_argument(
        "--train_batch_size", type=int, default=512, help="Training batch size."
    )
    model_group.add_argument(
        "--infer_batch_size", type=int, default=512, help="Inference batch size."
    )
    model_group.add_argument(
        "--eval_metric",
        type=str,
        default="ptl/train_loss_epoch",
        help="Metric to monitor for early stopping and checkpointing.",
    )

    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    return args


def main():
    args = setup()

    if args.use_saved_model:
        raise NotImplementedError(f"Loading from saved models not implemented yet!")
    elif args.use_best_config:
        prefix = f"{args.data_dir}/configs/"
        no_thresh_path = prefix + f"{args.dataset}_{args.model_name}_config.json"
        thresh_path = prefix + f"{args.dataset}_{args.model_name}_w_thresh_config.json"

        best_config_path = thresh_path
        if not os.path.exists(thresh_path):
            best_config_path = no_thresh_path
            if not os.path.exists(no_thresh_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), no_thresh_path)

        if args.verbose:
            print(f"Using saved config: {best_config_path}")

        if not os.path.exists(best_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), best_config_path)

        with open(best_config_path, "r") as f:
            config = json.load(f)
    else:
        # Manually input the config for testing
        config = {
            "lstm_hidden_size": 32,
            "lstm_num_layers": 1,
            "embed_size": 32,
            "fc_num_layers": 2,
            "dropout_rate": 0.1,
            "learning_rate": 1e-1,
            "pos_weight": 1.1371344902865825,
            "scale_loss": True,
            "vocab_size": 128,
            "seq_len": args.tokenizer_max_length,
        }

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    if args.verbose:
        print(f"Loading data...")
    dm = SmilesDataModule(
        dataset=args.dataset,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        model_name=args.model_name,
        target_col=args.target_col,
        smiles_col=args.smiles_col,
        train_batch_size=args.train_batch_size,
        infer_batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    if args.verbose:
        print(
            f"""Done.
            Train shape = {dm.encoding_shape('train')}
            Validation shape = {dm.encoding_shape('validation')}
            Test shape = {dm.encoding_shape('test')}"""
        )

    if args.find_threshold != "none" and not (f"best_threshold_{args.find_threshold}" in config):
        if args.verbose:
            print(f"Finding best threshold according to '{args.find_threshold}' metric.")

        if args.find_threshold == "f1":
            evaluator_fn = f1_score
        elif args.find_threshold == "j":
            raise NotImplementedError(
                f"Threshold determination using Youden's J not implemented yet."
            )

        trainer = Trainer(
            max_epochs=args.n_epochs,
            accelerator=args.device,
            devices=args.n_devices,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor=args.eval_metric, patience=args.patience, mode="min")],
        )

        model = init_neural_model(args.model_name, config=config)

        trainer.fit(model, dm.trn_dataloader, dm.val_dataloader)
        val_preds = torch.cat(trainer.predict(model, dm.val_dataloader)).detach().cpu().numpy()
        y_val = dm.val_set.y.detach().cpu().numpy()

        t_metrics = [
            evaluator_fn(y_val, (val_preds > t).astype(int)) for t in np.arange(0.0, 1.01, 0.02)
        ]

        best_threshold = t_metrics[np.argmax(t_metrics)]

        config[f"best_threshold_{args.find_threshold}"] = best_threshold

        if args.verbose:
            print(
                f"Found best threshold {round(best_threshold, 2)}. Saving to best config path if exists."
            )

        if args.use_best_config:
            config_fname = f"{args.dataset}_{args.model_name}_w_thresh_config.json"
            config_path = f"{args.data_dir}/configs/{config_fname}"

            with open(config_path, "w") as f:
                json.dump(config, f)

    model = init_neural_model(args.model_name, config=config)

    if args.verbose:
        print(f"Model:\n {model}")

    # Including timestamp will separate out each run
    if args.tb_inc_timestamp:
        study_name = f'pl_{args.dataset}_{args.model_name}_{time.strftime("%Y%m%d-%H%M%S")}'
    else:
        study_name = f"pl_{args.dataset}_{args.model_name}"

    # Initialize tensorboard logger
    logger = TensorBoardLogger(f"{args.data_dir}/tensorboard_logs", name=study_name, log_graph=True)

    # Eval metric may have '/' in it which creates unnecessary subdirectories
    treated_eval_metric = "_".join(args.eval_metric.split("/"))
    ckpt_fname = f"{args.dataset}_{args.model_name}_best_{treated_eval_metric}"

    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.device,
        devices=args.n_devices,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor=args.eval_metric, patience=args.patience, mode="min"),
            ModelCheckpoint(
                dirpath=f"{args.data_dir}/checkpoints",
                filename=ckpt_fname,
                monitor=args.eval_metric,
                mode="min",
            ),
        ],
    )

    # Fit model and evaluate on the test set
    trainer.fit(model, dm.trn_val_dataloader)
    trainer.test(model, dm.tst_dataloader)

    # Create test predictions and save to disk
    if args.verbose:
        print(f"Predicting and saving to disk...")
    test_probs = torch.cat(trainer.predict(model, dm.tst_dataloader)).detach().cpu().numpy()
    test_preds = (test_probs > config[f"best_threshold_{args.find_threshold}"]).astype(int)

    with open(
        f"{args.data_dir}/predictions/{args.dataset}_{args.model_name}_tst_probs.npy", "wb"
    ) as f:
        np.save(f, test_probs)
    with open(
        f"{args.data_dir}/predictions/{args.dataset}_{args.model_name}_tst_preds.npy", "wb"
    ) as f:
        np.save(f, test_preds)
    if args.verbose:
        print("Done.")


if __name__ == "__main__":
    main()
