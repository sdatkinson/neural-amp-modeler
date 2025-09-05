import torch
import scipy.signal


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses

def get_window(win_type: str, win_length: int):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))

    return win
