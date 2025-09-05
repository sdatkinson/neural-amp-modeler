import torch
import numpy as np
from typing import List, Any

from .utils import apply_reduction, get_window
from .perceptual import SumAndDifference, FIRFilter


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    """STFT loss module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).

    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        mag_distance (str, optional): Distance function ["L1", "L2"] for the magnitude loss terms.
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        **kwargs
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        self.device = device

        self.phs_used = bool(self.w_phs)

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction,
            distance=mag_distance,
            **kwargs
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction,
            distance=mag_distance,
            **kwargs
        )

        # setup mel filterbank
        if scale is not None:
            try:
                import librosa.filters
            except Exception as e:
                print(e)
                print("Try `pip install auraloss[all]`.")

            if self.scale == "mel":
                assert sample_rate != None  # Must set sample rate to use mel scale
                assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
                fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = torch.tensor(fb).unsqueeze(0)

            elif self.scale == "chroma":
                assert sample_rate != None  # Must set sample rate to use chroma scale
                assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate, n_fft=fft_size, n_chroma=n_bins
                )

            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )

            self.register_buffer("fb", fb)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

        if self.perceptual_weighting:
            if sample_rate is None:
                raise ValueError(
                    f"`sample_rate` must be supplied when `perceptual_weighting = True`."
                )
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps)
        )

        # torch.angle is expensive, so it is only evaluated if the values are used in the loss
        if self.phs_used:
            x_phs = torch.angle(x_stft)
        else:
            x_phs = None

        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()

        if self.perceptual_weighting:  # apply optional A-weighting via FIR filter
            # since FIRFilter only support mono audio we will move channels to batch dim
            input = input.view(bs * chs, 1, -1)
            target = target.view(bs * chs, 1, -1)

            # now apply the filter to both
            self.prefilter.to(input.device)
            input, target = self.prefilter(input, target)

            # now move the channels back
            input = input.view(bs, chs, -1)
            target = target.view(bs, chs, -1)

        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(input.device)

        x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = torch.nn.functional.mse_loss(x_phs, y_phs) if self.phs_used else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class MelSTFTLoss(STFTLoss):
    """Mel-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_mels=128,
        **kwargs,
    ):
        super(MelSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "mel",
            n_mels,
            **kwargs,
        )


class ChromaSTFTLoss(STFTLoss):
    """Chroma-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_chroma=12,
        **kwargs,
    ):
        super(ChromaSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "chroma",
            n_chroma,
            **kwargs,
        )


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    perceptual_weighting,
                    scale_invariance,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class RandomResolutionSTFTLoss(torch.nn.Module):
    """Random resolution STFT loss module.

    See [Steinmetz & Reiss, 2020](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf)

    Args:
        resolutions (int): Total number of STFT resolutions.
        min_fft_size (int): Smallest FFT size.
        max_fft_size (int): Largest FFT size.
        min_hop_size (int): Smallest hop size as porportion of window size.
        min_hop_size (int): Largest hop size as porportion of window size.
        window (str): Window function type.
        randomize_rate (int): Number of forwards before STFTs are randomized.
    """

    def __init__(
        self,
        resolutions=3,
        min_fft_size=16,
        max_fft_size=32768,
        min_hop_size=0.1,
        max_hop_size=1.0,
        windows=[
            "hann_window",
            "bartlett_window",
            "blackman_window",
            "hamming_window",
            "kaiser_window",
        ],
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_mels=None,
        randomize_rate=1,
        **kwargs,
    ):
        super().__init__()
        self.resolutions = resolutions
        self.min_fft_size = min_fft_size
        self.max_fft_size = max_fft_size
        self.min_hop_size = min_hop_size
        self.max_hop_size = max_hop_size
        self.windows = windows
        self.randomize_rate = randomize_rate
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_mels = n_mels

        self.nforwards = 0
        self.randomize_losses()  # init the losses

    def randomize_losses(self):
        # clear the existing STFT losses
        self.stft_losses = torch.nn.ModuleList()
        for n in range(self.resolutions):
            frame_size = 2 ** np.random.randint(
                np.log2(self.min_fft_size), np.log2(self.max_fft_size)
            )
            hop_size = int(
                frame_size
                * (
                    self.min_hop_size
                    + (np.random.rand() * (self.max_hop_size - self.min_hop_size))
                )
            )
            window_length = int(frame_size * np.random.choice([1.0, 0.5, 0.25]))
            window = np.random.choice(self.windows)
            self.stft_losses += [
                STFTLoss(
                    frame_size,
                    hop_size,
                    window_length,
                    window,
                    self.w_sc,
                    self.w_log_mag,
                    self.w_lin_mag,
                    self.w_phs,
                    self.sample_rate,
                    self.scale,
                    self.n_mels,
                )
            ]

    def forward(self, input, target):
        if input.size(-1) <= self.max_fft_size:
            raise ValueError(
                f"Input length ({input.size(-1)}) must be larger than largest FFT size ({self.max_fft_size})."
            )
        elif target.size(-1) <= self.max_fft_size:
            raise ValueError(
                f"Target length ({target.size(-1)}) must be larger than largest FFT size ({self.max_fft_size})."
            )

        if self.nforwards % self.randomize_rate == 0:
            self.randomize_losses()

        loss = 0.0
        for f in self.stft_losses:
            loss += f(input, target)
        loss /= len(self.stft_losses)

        self.nforwards += 1

        return loss


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.

    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes (List[int]): List of FFT sizes.
        hop_sizes (List[int]): List of hop sizes.
        win_lengths (List[int]): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        mel_stft (bool, optional): Use Multi-resoltuion mel spectrograms. Default: False
        n_mel_bins (int, optional): Number of mel bins to use when mel_stft = True. Default: 128
        sample_rate (float, optional): Audio sample rate. Default: None
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        w_sum: float = 1.0,
        w_diff: float = 1.0,
        output: str = "loss",
        **kwargs,
    ):
        super().__init__()
        self.sd = SumAndDifference()
        self.w_sum = w_sum
        self.w_diff = w_diff
        self.output = output
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes,
            hop_sizes,
            win_lengths,
            window,
            **kwargs,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """This loss function assumes batched input of stereo audio in the time domain.

        Args:
            input (torch.Tensor): Input tensor with shape (batch size, 2, seq_len).
            target (torch.Tensor): Target tensor with shape (batch size, 2, seq_len).

        Returns:
            loss (torch.Tensor): Aggreate loss term. Only returned if output='loss'.
            loss (torch.Tensor), sum_loss (torch.Tensor), diff_loss (torch.Tensor):
                Aggregate and intermediate loss terms. Only returned if output='full'.
        """
        assert input.shape == target.shape  # must have same shape
        bs, chs, seq_len = input.size()

        # compute sum and difference signals for both
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        # compute error in STFT domain
        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss
