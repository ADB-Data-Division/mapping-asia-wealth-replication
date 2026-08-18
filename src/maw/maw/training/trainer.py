import os
import torch
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from maw.models.resnet import unfreeze_modules
from .optim import build_optimizer, lin_warmup_exp_decay_scheduler


class EarlyStopping:
    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 0.0,
            checkpoint_dir: str = ".",
            filename: str = "model_checkpoint.pt"
    ):
        """
        Monitor loss and save checkpoint given a significant decrease. Assumes lower is better.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum decrease in loss qualify as an improvement.
            checkpoint_dir: Directory in which to save the best‐model checkpoint.
            filename: Name of the checkpoint file.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = os.path.join(checkpoint_dir, filename)
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, model: torch.nn.Module) -> bool:
        """
        Call this at the end of each epoch.

        Args:
            score: The current validation score (lower is better).
            model: Model whose weights may be checkpointed.

        Returns:
            bool: True if training should stop (patience exceeded), else False.
        """
        if self.best_score is None:                     # if first epoch, initialize best_score and save checkpoint
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score - self.min_delta:  # if no significant improvement
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:                                           # if improvement, reset counter and update best_score
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)

        return self.should_stop

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model when validation score decreases."""
        torch.save(model.state_dict(), self.checkpoint_path)
        logging.info(f"Validation score improved. Saving model to {self.checkpoint_path}...")

    def restore_best(self, model: torch.nn.Module) -> None:
        """Re-load the best‐saved weights back into model."""
        model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))
        logging.info(f"Restored best model weights from {self.checkpoint_path}")


@dataclass
class TrainConfig:
    model_type: str
    device: torch.device
    criterion: torch.nn.Module
    learning_rate: float
    weight_decay: float
    optim_class: type[Optimizer]
    optim_sf: dict | None = None
    epochs: int = 30
    min_epochs: int = 0
    scheduler_params: dict | None = None
    early_stopping_params: dict | None = None
    training_schedule: dict | None = None
    transforms_train: torch.nn.Module | None = None
    transforms_val: torch.nn.Module | None = None


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 cfg: TrainConfig,
                 log_dir: str | None = None
                 ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.cfg = cfg
        self.cfg.transforms_train = cfg.transforms_train or (lambda x: x)
        self.cfg.transforms_val = cfg.transforms_val or (lambda x: x)
        self.optimizer = build_optimizer(
            model=model,
            optim_class=cfg.optim_class,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            scale_factors=cfg.optim_sf
        )
        self.early_stopper = (
            EarlyStopping(**cfg.early_stopping_params)
            if cfg.early_stopping_params else None
        )
        self.scheduler = (
            lin_warmup_exp_decay_scheduler(optimizer=self.optimizer, **self.cfg.scheduler_params, last_epoch=-1)
            if cfg.scheduler_params else None
        )
        self.history = []

    def _run_loop(self,
                  loader: DataLoader,
                  transforms: torch.nn.Module,
                  train: bool
                  ) -> tuple[float, float, float, float]:

        total_loss = 0.0
        n_skipped = 0
        n = 0
        sum_abs_err = 0.0
        sum_sq_err = 0.0
        sum_y = 0.0
        sum_y_sq = 0.0
        desc = "Train" if train else "Validate"

        # Iterate over batches, record model predictions and labels, and update params if train == True
        non_blocking = self.cfg.device.type == "cuda"
        for batch in tqdm(loader, desc=desc):
            images = batch["image"].to(self.cfg.device, non_blocking=non_blocking)
            labels = batch["label"].to(self.cfg.device, non_blocking=non_blocking).float()

            # Skip batches with NaN labels (all-NODATA patch in wealth raster)
            if torch.isnan(labels).any():
                n_skipped += 1
                continue

            images = transforms(images)

            # If model expects images + covarits
            if self.cfg.model_type == "swinv2":
                covariates = batch["covariates"].unsqueeze(1).float().to(self.cfg.device, non_blocking=non_blocking)
                outputs = self.model(images, covariates)
            elif self.cfg.model_type == "resnet50":
                outputs = self.model(images)
            else:
                raise NotImplementedError("Issue in cfg.model_type: Only 'resnet50' or 'swinv2' are supported.")

            loss = self.cfg.criterion(outputs.view(-1), labels)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()

            with torch.no_grad():
                preds = outputs.view(-1).detach().float()
                lbls = labels.float()
                err = (preds - lbls).cpu()
                total_loss += loss.item() * images.size(0)
                n += len(lbls)
                sum_abs_err += err.abs().sum().item()
                sum_sq_err += err.pow(2).sum().item()
                sum_y += lbls.cpu().sum().item()
                sum_y_sq += lbls.cpu().pow(2).sum().item()

        if n_skipped:
            logging.warning(f"{desc}: skipped {n_skipped} batches with all-NODATA labels")

        # Compute and return performance metrics from running sums
        avg_loss = total_loss / n
        mae = sum_abs_err / n
        rmse = (sum_sq_err / n) ** 0.5
        mean_y = sum_y / n
        ss_tot = sum_y_sq - n * mean_y ** 2
        r2 = 1.0 - sum_sq_err / ss_tot
        return avg_loss, mae, rmse, r2

    def train_epoch(self) -> tuple[float, float, float, float]:
        """Train the model for one epoch and return epoch metrics."""
        self.model.train()
        return self._run_loop(loader=self.train_loader, transforms=self.cfg.transforms_train, train=True)

    def validate_epoch(self) -> tuple[float, float, float, float]:
        """Iterate over all batches on the validation set and return epoch metrics."""
        self.model.eval()
        with torch.no_grad():
            return self._run_loop(loader=self.val_loader, transforms=self.cfg.transforms_val, train=False)

    def fit(self, return_best: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, torch.nn.Module]:
        """Train the model using the supplied cfg when instantiating the Trainer."""

        writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None

        for epoch in range(self.cfg.epochs):

            # Unfreeze module if arrived at specified epoch; rebuild optimizer and scheduler with fresh warmup
            if self.cfg.training_schedule and epoch in self.cfg.training_schedule:
                unfreeze_modules(self.model, self.cfg.training_schedule[epoch])
                self.optimizer = build_optimizer(model=self.model, optim_class=self.cfg.optim_class, learning_rate=self.cfg.learning_rate,
                                                 weight_decay=self.cfg.weight_decay, scale_factors=self.cfg.optim_sf)
                if self.scheduler:
                    self.scheduler = lin_warmup_exp_decay_scheduler(optimizer=self.optimizer, **self.cfg.scheduler_params,
                                                                    last_epoch=-1)

            # Run training and validation passes
            logging.info(f"--- Epoch: {epoch}")
            train_loss, train_mae, train_rmse, train_r2 = self.train_epoch()
            val_loss, val_mae, val_rmse, val_r2 = self.validate_epoch()
            epoch_metrics = {
                "epoch": epoch,
                "train_avg_loss": train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "val_avg_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "lr_epoch": self.optimizer.param_groups[0]['lr'],
                "lr": self.cfg.learning_rate,
                "wd": self.cfg.weight_decay,
            }
            logging.info(epoch_metrics)
            self.history.append(epoch_metrics)

            if writer is not None:
                writer.add_scalars('Loss',{'train': train_loss, 'val': val_loss}, epoch)
                writer.add_scalars('MAE', {'train': train_mae, 'val': val_mae}, epoch)
                writer.add_scalars('RMSE',{'train': train_rmse, 'val': val_rmse}, epoch)
                writer.add_scalars('R2',{'train': train_r2, 'val': val_r2}, epoch)
                writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)  # current lr

            if self.scheduler:
                self.scheduler.step()

            if self.early_stopper:
                stop = self.early_stopper.step(val_loss, self.model)
                if stop and epoch > self.cfg.min_epochs:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        if writer is not None:
            writer.close()

        if return_best:
            if self.early_stopper is None:
                raise ValueError("return_best=True requires early_stopping_params to be configured")
            self.early_stopper.restore_best(self.model)
            return pd.DataFrame(self.history), self.model

        return pd.DataFrame(self.history)

    def save_history(self, filepath: str, **kwargs) -> None:
        """Save history to csv. **kwargs are used to add additional columns."""
        df = pd.DataFrame(self.history)
        for key, value in kwargs.items():
            df[key] = value
        df.to_csv(filepath, index=False)
