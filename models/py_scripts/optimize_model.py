import os
import json
import argparse

import torch
import numpy as np

from transformers import AutoTokenizer

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from tbprop.deep_learning.datasets import SmilesDataModule
from tbprop.deep_learning.optimizers import OptunaOptimizerForPyTorchLightning
from tbprop.deep_learning.models import init_neural_model


def setup():
    parser = argparse.ArgumentParser()

    exp_group = parser.add_argument_group("Experiment arguments")
    exp_group.add_argument("--seed", type=int, default=42)
    exp_group.add_argument("--dataset", type=str, default="taacf")
    exp_group.add_argument("--target_col", type=str, default="inhibition_bin")
    exp_group.add_argument("--smiles_col", type=str, default="smiles")
    exp_group.add_argument("--hf_home", type=str, default="data/transformers_cache")
    exp_group.add_argument("--data_dir", type=str, default="./data")
    exp_group.add_argument("--save_model", action="store_true")

    opt_group = parser.add_argument_group("Optuna arguments")
    opt_group.add_argument("--eval_metric", type=str, default="ptl/val_loss")
    opt_group.add_argument("--n_trials", type=int, default=10)
    opt_group.add_argument("--timeout", type=int, default=600)
    opt_group.add_argument("--n_warmup_steps", type=int, default=5)
    opt_group.add_argument("--patience", type=int, default=7)
    opt_group.add_argument("--optuna_verbosity", type=int, default=1)

    model_group = parser.add_argument_group("Model training arguments")
    model_group.add_argument("--model_name", type=str, default="FCNN")
    model_group.add_argument("--n_devices", type=int, default=1)
    model_group.add_argument("--num_workers", type=int, default=7)
    model_group.add_argument("--tokenizer_max_length", type=int, default=512)
    model_group.add_argument("--device", type=str, default="cpu")
    model_group.add_argument("--n_epochs", type=int, default=10)
    model_group.add_argument("--train_batch_size", type=int, default=512)
    model_group.add_argument("--infer_batch_size", type=int, default=512)

    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    return args


def main():
    args = setup()

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

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
    print(
        f"""Done.
        Train shape = {dm.encoding_shape('train')}
        Validation shape = {dm.encoding_shape('validation')}
        Test shape = {dm.encoding_shape('test')}"""
    )

    optimizer = OptunaOptimizerForPyTorchLightning(
        args,
        trn_dataloader=dm.trn_dataloader,
        val_dataloader=dm.val_dataloader,
        model_name=args.model_name,
        dataset=args.dataset,
        eval_metric=args.eval_metric,
        n_warmup_steps=args.n_warmup_steps,
        seq_len=args.tokenizer_max_length,
        verbose=args.optuna_verbosity,
        early_stopping_patience=args.patience,
    )
    best_trial = optimizer.optimize(n_trials=args.n_trials, timeout=args.timeout)
    best_config = best_trial.params
    best_config["seq_len"] = args.tokenizer_max_length
    print(f"Best trial: {best_config}")

    if args.save_model:
        fname = f"data/configs/{args.dataset}_{args.model_name}_config.json"
        with open(fname, "w") as f:
            json.dump(best_config, f)

    best_model = init_neural_model(args.model_name, best_config)

    # Eval metric may have '/' in it which creates unnecessary subdirectories
    treated_eval_metric = "_".join(args.eval_metric.split("/"))
    ckpt_fname = f"{args.dataset}_{args.model_name}_best_{treated_eval_metric}"

    trainer = Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.device,
        devices=args.n_devices,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="ptl/train_loss", mode="min"),
            ModelCheckpoint(
                dirpath=f"{args.data_dir}/checkpoints",
                filename=ckpt_fname,
                monitor=args.eval_metric,
                mode="min",
            ),
        ],
    )

    trainer.fit(best_model, train_dataloaders=dm.trn_val_dataloader)
    trainer.test(best_model, dataloaders=dm.tst_dataloader)

    # Create test predictions and save to disk
    if args.verbose:
        print(f"Predicting and saving to disk...")
    test_probs = torch.cat(trainer.predict(best_model, dm.tst_dataloader)).detach().cpu().numpy()
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
