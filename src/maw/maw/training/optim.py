import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
        model: torch.nn.Module,
        optim_class: type[Optimizer],
        learning_rate: float,
        weight_decay: float,
        scale_factors: dict[str, tuple[float, float]] | None = None
        ) -> Optimizer:
    """
    Builds an optimizer where for each child module,  (lr, wd) = (base_lr * lr_scale, base_wd * wd_scale).
    Args:
        model: Model being optimised.
        optim_class: Class of optimizer to build.
        learning_rate: Base learning rate.
        weight_decay: Base weight_decay.
        scale_factors: Factors to scale hyperparams, expecting (lr_scale, wd_scale).

    Returns:
        A torch.optim.Optimizer instance.
    """
    default_scales = (1.0, 1.0)
    sf = scale_factors or {}

    param_groups = []
    for name, module in model.named_children():
        if not any(p.requires_grad for p in module.parameters()):
            continue

        lr_scale, wd_scale = sf.get(name, default_scales)
        param_groups.append({
            "params": module.parameters(),
            "lr": learning_rate * lr_scale,
            "weight_decay": weight_decay * wd_scale
        })

    optimizer = optim_class(param_groups)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optimizer


def lin_warmup_exp_decay_scheduler(optimizer,
                                   warmup_epochs: int,
                                   decay_rate: float = 0.95,
                                   last_epoch: int = -1
                                   ) -> LambdaLR:
    """
    Creates a LambdaLR scheduler with linear warmup followed by exponential decay.
    Args:
        optimizer: Optimizer instance.
        warmup_epochs: Number of epochs for warmup.
        decay_rate: Decay rate per epoch after warmup.
        last_epoch: Index for epoch at which to initialise the scheduler.
    Returns:
        A LambdaLR scheduler.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linearly increase LR
            return float(current_epoch + 1) / warmup_epochs
        else:
            # After warmup, apply exponential decay
            return decay_rate ** (current_epoch - warmup_epochs)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
