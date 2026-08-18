import torch
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: torch.Tensor | np.ndarray,
                    y_pred: torch.Tensor | np.ndarray
                    ) -> tuple[float, float, float]:
    """Compute MAE, RMSE, and R^2 given torch or numpy arrays."""

    # Convert tensors to numpy arrays, if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Check shape of arrays are identical
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # Calculate MAE, RMSE and R^2
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    rmse = math.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    r2 = r2_score(y_true=y_true, y_pred=y_pred)

    return mae, rmse, r2