import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections.abc import Sequence


def compute_band_histograms(dataset: Dataset, batch_size: int = 32, num_bins: int = 1000, **kwargs) -> list:
    """
    Compute normalized histograms for each channel across the entire dataset.

    Args:
        dataset: Dataset returning images as torch.Tensor of shape (C, H, W).
        batch_size: Batch size for processing the dataset.
        num_bins: Number of histogram bins to use.
        **kwargs: Additional arguments passed to torch.histogram.

    Returns:
        A list of length C (number of channels) where each element is a 1D tensor of length num_bins.
    """
    hist_list = None

    for batch in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
        images = batch["image"]
        if hist_list is None:
            num_channels = images.shape[1]
            hist_list = [torch.zeros(num_bins, dtype=torch.float32) for _ in range(num_channels)]

        for c in range(images.shape[1]):
            channel_data = images[:, c, :, :].flatten().float()
            hist, _ = torch.histogram(channel_data, bins=num_bins, **kwargs)
            hist_list[c] += hist

    return hist_list


def compute_band_stats(dataset: Dataset,
                       batch_size: int = 32,
                       per_channel_min: Sequence[float] | None = None,
                       per_channel_max: Sequence[float] | None = None
                       ) -> dict:
    """
    Compute global per-channel statistics (min, max, mean, std) for the dataset, clipping at min/max.

    Args:
        dataset : Dataset returning images as torch.Tensor of shape (C, H, W).
        batch_size: Batch size for processing.
        per_channel_min: Sequence of minimum clipping value for each channel.
        per_channel_max: Sequence of maximum clipping value for each channel.

    Returns:
        A dictionary mapping channel index to its statistics (min, max, mean, std).
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Determine number of channels from the first image in the dataset
    first_sample = dataset[0]
    num_channels = first_sample["image"].squeeze().shape[0]

    # Initialize accumulators for each channel
    total_sum = [0.0 for _ in range(num_channels)]
    total_sum_sq = [0.0 for _ in range(num_channels)]
    total_count = [0 for _ in range(num_channels)]
    global_min = [float('inf') for _ in range(num_channels)]
    global_max = [float('-inf') for _ in range(num_channels)]

    for batch in tqdm(dataloader):
        # Convert to float for calculations; shape: (B, C, H, W)
        batch = batch["image"].float()
        B, C, H, W = batch.shape

        # If clipping values are provided, apply per-channel clipping.
        if per_channel_min is not None and per_channel_max is not None:
            clip_min = torch.tensor(per_channel_min, dtype=batch.dtype, device=batch.device).view(1, C, 1, 1)
            clip_max = torch.tensor(per_channel_max, dtype=batch.dtype, device=batch.device).view(1, C, 1, 1)
            batch = torch.clamp(batch, min=clip_min, max=clip_max)

        # Flatten each image per channel
        batch_flat = batch.view(B, C, -1)                 # shape: (B, C, H*W)

        # Sum and sum of squares along pixel dimensions (axis 2)
        sum_channels = batch_flat.sum(dim=2)              # shape: (B, C)
        sum_sq_channels = (batch_flat ** 2).sum(dim=2)    # shape: (B, C)
        count = batch_flat.shape[2]

        # Accumulate statistics per channel over the batch
        for c in range(C):
            total_sum[c] += sum_channels[:, c].sum().item()
            total_sum_sq[c] += sum_sq_channels[:, c].sum().item()
            total_count[c] += B * count

            # Update global min and max for channel c
            batch_min = batch_flat[:, c, :].min().item()
            batch_max = batch_flat[:, c, :].max().item()
            global_min[c] = min(global_min[c], batch_min)
            global_max[c] = max(global_max[c], batch_max)

    # Compute final statistics per channel
    stats = {}
    for c in range(num_channels):
        mean = total_sum[c] / total_count[c]
        variance = (total_sum_sq[c] / total_count[c]) - (mean ** 2)
        std = variance ** 0.5
        stats[c] = {'min': global_min[c],
                    'max': global_max[c],
                    'mean': mean,
                    'std': std}

    return stats


def plot_band_ecdf(histograms: list[torch.Tensor], hist_range: tuple[float, float]) -> None:
    """
    Plot the empirical cumulative distribution function (ECDF) for each channel based on the input histograms.

    Args:
        histograms: List of normalized histograms, per channel.
        hist_range: The (min, max) range over which the histogram was computed.
    """
    num_channels = len(histograms)
    num_bins = histograms[0].numel()
    x_vals = np.linspace(hist_range[0], hist_range[1], num_bins)    # compute bin centers given histogram range

    fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 4))

    # If only one channel, wrap axes in a list for consistency
    if num_channels == 1:
        axes = [axes]

    for i, hist in enumerate(histograms):
        hist = hist / hist.sum()                    # normalise histogram
        ecdf = torch.cumsum(hist, dim=0).numpy()    # compute the empirical CDF
        axes[i].plot(x_vals, ecdf)
        axes[i].set_title(f'Channel {i} ECDF')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.show()


def compute_percentiles(histograms: Sequence[torch.Tensor],
                        percentiles: Sequence[float],
                        hist_range: tuple[float, float]
                        ) -> list[dict]:
    """
    Compute the given percentiles from histograms of pixel intensities.

    Args:
        histograms: A sequence of 1D tensors (one per channel), all of equal length.
        percentiles: A sequence of percentile values (each in [0,100]).
        hist_range: (min, max) bounds used to build each histogram.

    Returns:
        A list of dictionaries, one per channel, mapping each percentile to its corresponding intensity value.
    """
    results = []
    min_val, max_val = hist_range
    num_bins = histograms[0].numel()
    bin_width = (max_val - min_val) / num_bins

    for hist in histograms:
        hist = hist / hist.sum()
        ecdf = torch.cumsum(hist, dim=0)
        percentile_values = {}
        for p in percentiles:
            threshold = p / 100.0
            idx = torch.searchsorted(ecdf, threshold)
            idx = min(idx.item(), num_bins - 1)                    # clamp index in case idx equals num_bins
            intensity_value = min_val + (idx + 0.5) * bin_width
            percentile_values[p] = intensity_value
        results.append(percentile_values)

    return results
