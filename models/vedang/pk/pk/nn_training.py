import torch
import numpy as np
import pandas as pd
from time import time

from tqdm import tqdm
# from pk.utils import plot_losses
from pk.datasets import SmilesDataset
from pk.metrics import binary_classification_metrics
from pk.deep_models import MFBERTTransformer
from MFBERT.Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer


def finetune(model, train_dataloader, n_epochs=5,
            learning_rate=1e-4, momentum=0.9, early_stopping=True,
            tol=10, device='cpu', verbose=False):
    """
    Finetune a model on a training set.

    Parameters
    ----------
    model: torch.nn.Module
        Model to finetune.
    train_dataloader: torch.utils.data.DataLoader
        Dataloader for training set.
    n_epochs: int, optional
        Number of epochs to train for. Default is 5.
    learning_rate: float, optional
        Learning rate for optimizer. Default is 1e-4.
    momentum: float, optional
        Momentum for optimizer. Default is 0.9.
    early_stopping: bool, optional
        Whether to use early stopping. Default is True.
    tol: int, optional
        Number of epochs to wait before stopping if no improvement. Default is 10.
    verbose: bool, optional
        Whether to print progress. Default is False.

    Returns
    -------
    model: torch.nn.Module
        Trained model.
    train_losses: list
        List of training losses per epoch.
    """

    # Initialize criterion and optimizer
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # For logging
    train_losses = []
    epoch_iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)
    
    # Train
    for epoch in epoch_iterator:
        itr_losses, t0 = [], time()
        
        # Iterate over batches
        for batch_ndx, (X_batch, y_batch) in enumerate(train_dataloader):
            # if verbose and batch_ndx % 100 == 0:
            #     print(f"Batch {batch_ndx} of {len(train_dataloader)}.")

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            scores = model(X_batch).reshape(-1)

            # Compute loss
            loss = criterion(scores, y_batch.float())
            itr_losses.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Log epoch train loss
        mean_itr_loss = round(np.mean(itr_losses), 5)
        train_losses.append(mean_itr_loss)
        if verbose: print(f" Loss = {mean_itr_loss}. Epoch time = {round(time() - t0, 2)} s.")
        
        # Early stopping
        if early_stopping and \
           len(train_losses) > tol and \
           min(train_losses[-tol:]) == train_losses[-tol]:
            print("Stopping training early.")
            break
        
    return model, train_losses


def inference(model, test_dataloader, pred_threshold=0.5, device='cpu'):
    """ 
    Evaluate a trained model on a test set. 
    
    Parameters
    ----------
    model: torch.nn.Module
        A trained model.
    test_dataloader: torch.utils.data.DataLoader
        A dataloader for the test set.
    pred_threshold: float
        The threshold for classifying a sample as positive.
    device: str
        Device to run inference on. ('cuda' or 'cpu')

    Returns
    -------
    dict
    """
    model.eval()
    test_preds, test_scores, y_true = [], [], []
    for _, (X_batch, y_batch) in enumerate(tqdm(test_dataloader)):
        X_batch.to(device)
        outputs = model(X_batch)
        scores = outputs.reshape(-1)
        preds = torch.where(scores > pred_threshold, 1, 0)
        
        test_preds.append(preds.detach().cpu())
        test_scores.append(scores.detach().cpu())
        y_true.append(y_batch)

    accumulate = lambda x: torch.cat(x).numpy()
    
    test_preds = accumulate(test_preds)
    test_scores = accumulate(test_scores)
    y_true = accumulate(y_true)

    return binary_classification_metrics(y_true, test_preds, test_scores)


def evaluate_model(args):
    """ Train and evaluate a deep model. 
    
    Parameters
    ----------
    args should have the following attributes:
        train_path: str
            Path to training set.
        test_path: str
            Path to test set.
        ckpt_path: str
            Path to model checkpoint.
        train_batch_size: int
            Batch size for training.
        test_batch_size: int
            Batch size for testing.
        n_epochs: int
            Number of epochs to train for.
        learning_rate: float
            Learning rate for SGD.
        momentum: float
            Momentum for SGD.
        early_stopping: bool
            Whether to use early stopping.
        tol: int
            Number of epochs to wait before stopping.
        freeze_transformer: bool
            Whether to freeze the transformer layers.
        finetune: bool
            Whether to finetune the model.
        verbose: bool
            Whether to print progress.

    Returns
    -------
    pd.DataFrame
    """
    tokenizer = MFBERTTokenizer.from_pretrained('MFBERT/Tokenizer/Model/', 
                                            dict_file='MFBERT/Tokenizer/Model/dict.txt')

    # Load training and test sets
    train_set = SmilesDataset(args.train_path, tokenizer, smiles_col='smiles', label_col='hit_call')
    test_set = SmilesDataset(args.test_path, tokenizer, smiles_col='smiles', label_col='hit_call')

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

    # Initialize model
    model = MFBERTTransformer(args.ckpt_path, tokenizer.vocab_size, freeze_transformer=args.freeze_transformer)

    # Calculate metrics without any finetuning
    zero_shot_metrics = inference(model, test_dataloader)
    zero_shot_metrics.rename({'Value': 'Zero Shot'}, axis=1, inplace=True)

    # Finetune model
    if args.finetune:
        if args.verbose: print("Finetuning model...")
        model, train_losses = finetune(model, 
                                       train_dataloader,
                                       n_epochs=args.n_epochs,
                                       learning_rate=args.learning_rate,
                                       momentum=args.momentum,
                                       early_stopping=args.early_stopping,
                                       tol=args.tol,
                                       verbose=args.verbose)
        
        # Plot training losses
        # if args.verbose:
        #     plot_losses(train_losses, title='Train Loss vs. Epochs')
        
        # Calculate metrics after finetuning
        finetuned_metrics = inference(model, test_dataloader)
        finetuned_metrics.rename({'Value': 'Finetuned'}, axis=1, inplace=True)

        # Merge finetuned and zero-shot metrics
        test_metrics = pd.merge(zero_shot_metrics, finetuned_metrics, on='Metric', how='left')
    else:
        # No finetuning
        test_metrics = zero_shot_metrics
        
    # Save model
    if args.save_path:
        raise NotImplementedError('Saving not implemented yet.')
        
    return model, test_metrics


def embeddings(model, dl, to_dataframe=False, device='cpu', **kwargs):
    """ Calculate embeddings for a dataset using given model. """

    model.eval()

    batches, labels = [], []
    for _, (X_batch, y_batch) in enumerate(dl):
        X_batch = X_batch.to(device)
        batches.append(model.embed(X_batch).detach().cpu())
        labels.append(y_batch)
    X, y = torch.cat(batches, 0).numpy(), torch.cat(labels, 0).numpy()
    
    if to_dataframe:
        headers = [f"dim_{x}" for x in range(X.shape[1])]
        X = pd.DataFrame(X, columns=headers)
        
    return X, y