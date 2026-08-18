from torch import nn
from torchgeo.models import resnet50, ResNet50_Weights


def freeze_backbone(model: nn.Module) -> None:
    """Set require_grad=False for all top-level modules not named 'fc'."""
    for name, module in model.named_children():
        if name != 'fc':
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_modules(model: nn.Module, layer_names: list[str]) -> None:
    """Unfreeze top-level modules in a PyTorch model given a list of names."""
    for name, module in model.named_children():
        if name in layer_names:
            for param in module.parameters():
                param.requires_grad = True
            print(f"Unfroze {name}")


def build_resnet50(hidden_dim: int | None = 1024,
                   dropout: float = 0.5,
                   frozen_backbone: bool = False,
                   weights: str | None = None,
                   ) -> nn.Module:
    """Instantiate a base Resnet50 for 6-band Landsat SR imagery."""
    if weights == "LANDSAT_ETM_SR_MOCO":
        model = resnet50(weights=ResNet50_Weights.LANDSAT_ETM_SR_MOCO, num_classes=1)
    elif weights == "SENTINEL2_SI_MS_SATLAS":
        model = resnet50(weights=ResNet50_Weights.SENTINEL2_SI_MS_SATLAS)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise NotImplementedError(weights)

    if hidden_dim is not None:
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    if frozen_backbone:
        freeze_backbone(model)

    return model