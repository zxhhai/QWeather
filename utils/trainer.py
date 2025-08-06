import os
import time
import logging
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm
from datetime import datetime

class Trainer:
    """A class to handle the training and evaluation of a PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        save_path: str = "checkpoints",
        logger: Optional[logging.Logger] = None,
        log_dir: str = "logs",
        save_logs: bool = True
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The model to train.
            criterion (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer for training.
            device (torch.device): The device to run the training on.
            scheduler (_LRScheduler, optional): Learning rate scheduler. Defaults to None.
            save_path (str, optional): Path to save checkpoints. Defaults to "checkpoints".
            logger (logging.Logger, optional): Logger for logging messages. Defaults to None.

        Returns:
            None
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_path = save_path
        self.logger = logger or self._setup_logger(log_dir, save_logs)

        # create save directory if it does not exist
        os.makedirs(self.save_path, exist_ok=True)

        # training state
        self.current_epoch = 0
        self.best_metric = float('inf')  # Assuming lower is better for the metric
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []


    def _setup_logger(self, log_dir: str = "logs", save_logs: bool = True) -> logging.Logger:
        """
        Set up a default logger.

        Args:
            None

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            
            if save_logs:
                os.makedirs(log_dir, exist_ok=True)
                log_filename = os.path.join(log_dir, f"trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                file_handler = logging.FileHandler(log_filename)
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger


    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
        
        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        with tqdm(train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                # forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                # print(output)
                loss = self.criterion(output, target)

                # backward pass
                loss.backward()
                self.optimizer.step()

                # update total loss
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                # update progress bar
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

        return total_loss / num_batches


    def validate(self, val_loader: DataLoader, metrics_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Validate the model on the validation dataset.

        Args:
            val_loader (DataLoader): DataLoader for the validation dataset.
            metrics_fn (Callable, optional): Function to compute additional metrics. Defaults to None.
        
        Returns:
            Dict[str, float]: Validation results including loss and any additional metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs= []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())

        avg_loss = total_loss / len(val_loader)
        results = {'val_loss': avg_loss}

        # compute additional metrics if provided
        if metrics_fn:
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)
            metrics = metrics_fn(all_outputs, all_targets)
            results.update(metrics)

        return results
    

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        metrics_fn: Optional[Callable] = None,
        early_stopping_patience: int = 10,
        save_every: int = 5
    ) -> None:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            num_epochs (int): Number of epochs to train.
            metrics_fn (Callable, optional): Function to compute additional metrics. Defaults to None.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
            save_every (int, optional): Save checkpoint every n epochs. Defaults to 5.
        
        Returns:
            None
        """
        self.logger.info(f'Starting training for {epochs} epochs...')

        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # validate the model
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader, metrics_fn)
                self.val_losses.append(val_metrics['val_loss'])
                self.metrics_history.append(val_metrics)

                # check if this is the best model so far
                current_metric = val_metrics.get('val_loss', float('inf'))
                is_best = current_metric < self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                is_best = False
                current_metric = train_loss

            # update learning rate scheduler if available
            if self.scheduler:
                self.scheduler.step()

            # log training and validation results
            log_info = f'Epoch {self.current_epoch} / {epochs} - Train Loss: {train_loss:.4f}'
            if val_metrics:
                for key, value in val_metrics.items():
                    log_info += f' - {key}: {value:.4f}'
            self.logger.info(log_info)

            # save checkpoint
            if self.current_epoch % save_every == 0 or is_best:
                self.save_checkpoint(val_metrics or {'train_loss': train_loss}, is_best)
            
            # early stopping check
            if val_loader and patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping triggered at epoch {self.current_epoch}.')
                break
        
        training_time = time.time() - start_time
        self.logger.info(f'Training completed in {training_time:.2f} seconds.')

        # save final model
        self.save_checkpoint(val_metrics or {'train_loss': train_loss})


    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Make predictions using the trained model.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to predict on.
        
        Returns:
            torch.Tensor: Predictions from the model.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc='Predicting'):
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.cpu())
        
        return torch.cat(predictions)


    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save the model checkpoint.

        Arg:
            metrics (Dict[str, float]): Metrics to save with the checkpoint.
            is_best (bool, optional): Whether this is the best model so far. Defaults to False.
        
        Returns:
            None
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # save checkpoint
        checkpoint_path = os.path.join(self.save_path, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # save best model if applicable
        if is_best:
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"\033[92mNew best model saved with metric: {metrics}\033[0m")



    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        
        Returns:
            None
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.metrics_history = checkpoint.get('metrics_history', [])

        self.logger.info(f'Checkpoint loaded from epoch {self.current_epoch}')


    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the training history including losses and metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing training history.
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric
        }
    
    def save_training_history(self, save_path: str) -> None:
        """
        Save the training history to a file.

        Args:
            save_path (str): Path to save the training history.
        
        Returns:
            None
        """
        import json
        history = self.get_training_history()
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=4)
        self.logger.info(f'Training history saved to {save_path}')
