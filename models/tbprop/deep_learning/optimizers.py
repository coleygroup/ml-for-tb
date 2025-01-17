import time
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from tbprop.deep_learning.models import init_neural_model
from tbprop.deep_learning.search_spaces import build_neural_params


class OptunaOptimizerForPyTorchLightning:
    def __init__(
        self,
        args,
        trn_dataloader,
        val_dataloader,
        model_name="",
        dataset="",
        eval_metric="ptl/val_loss",
        n_warmup_steps=5,
        seq_len=512,
        early_stopping_patience=7,
        verbose=0,
    ):
        self.args = args
        self.trn_dataloader = trn_dataloader
        self.val_dataloader = val_dataloader
        self.model_name = model_name
        self.dataset = dataset
        self.eval_metric = eval_metric
        self.n_warmup_steps = n_warmup_steps
        self.seq_len = seq_len
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

    def name_study(self):
        """Util function to name a study object. Useful while logging."""

        return f'pl_optuna_{self.model_name}_{time.strftime("%Y%m%d-%H%M%S")}'

    def objective(self, trial):
        """
        Suggest new hyperparameters, build the model, and calculate objective fn.
        """

        config = build_neural_params(self.model_name, trial, dataset=self.dataset)
        config["seq_len"] = self.seq_len
        model = init_neural_model(self.model_name, config)

        callbacks = [
            EarlyStopping(
                monitor=self.eval_metric, mode="min", patience=self.early_stopping_patience
            ),
            PyTorchLightningPruningCallback(trial, monitor=self.eval_metric),
        ]

        trainer = Trainer(
            max_epochs=self.args.n_epochs,
            accelerator=self.args.device,
            devices=1,  # Change in case we use GPU again
            log_every_n_steps=1,
            callbacks=callbacks,
        )

        trainer.logger.log_hyperparams(config)
        trainer.fit(model, self.trn_dataloader, self.val_dataloader)

        return trainer.callback_metrics[self.eval_metric].item()

    def optimize(self, n_trials=10, timeout=600, show_progress_bar=True):
        """
        Starts the Optuna study. Ends after either
            i. `n_trials` trials have happened or
            ii. `timeout` number of seconds have passed.
        """

        study = optuna.create_study(
            study_name=self.name_study(),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=self.n_warmup_steps),
            direction="minimize" if "loss" in self.eval_metric else "maximize",
        )

        study.optimize(
            self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress_bar
        )
        return study.best_trial
