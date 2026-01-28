import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings


class EEGAugmentation:
    """
    Base class for EEG data augmentation
    All augmentation methods should return data with the same shape as input
    """

    def __init__(self, probability: float = 0.5, **kwargs):
        self.probability = probability
        self.config = kwargs

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probability:
            return self.apply(eeg_data)
        return eeg_data

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianNoiseAugmentation(EEGAugmentation):
    """
    Gaussian Noise Augmentation
    Add random Gaussian noise to EEG signals
    """

    def __init__(self, probability: float = 0.5, std: float = 0.01, **kwargs):
        super().__init__(probability, **kwargs)
        self.std = std

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(eeg_data) * self.std
        return eeg_data + noise


class TimeShiftAugmentation(EEGAugmentation):
    """
    Time Shift Augmentation
    Randomly shift signals along the time axis
    """

    def __init__(self, probability: float = 0.5, max_shift: int = 10, mode: str = 'circular', **kwargs):
        super().__init__(probability, **kwargs)
        self.max_shift = max_shift
        self.mode = mode  # 'circular', 'zero', 'reflect'

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return eeg_data

        if self.mode == 'circular':
            return torch.roll(eeg_data, shifts=shift, dims=-1)
        elif self.mode == 'zero':
            shifted = torch.zeros_like(eeg_data)
            if shift > 0:
                shifted[..., shift:] = eeg_data[..., :-shift]
            else:
                shifted[..., :shift] = eeg_data[..., -shift:]
            return shifted
        else:  # reflect
            return F.pad(eeg_data, (abs(shift), abs(shift)), mode='reflect')[..., abs(shift):-abs(shift)]


class ChannelDropoutAugmentation(EEGAugmentation):
    """
    Channel Dropout Augmentation
    Randomly dropout some EEG channels to simulate poor electrode contact
    """

    def __init__(self, probability: float = 0.5, dropout_rate: float = 0.1, **kwargs):
        super().__init__(probability, **kwargs)
        self.dropout_rate = dropout_rate

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_times = eeg_data.shape
        mask = torch.ones_like(eeg_data)

        for i in range(batch_size):
            # Randomly select channels to drop
            n_drop = max(1, int(n_channels * self.dropout_rate))
            channels_to_drop = random.sample(range(n_channels), n_drop)
            mask[i, channels_to_drop, :] = 0

        return eeg_data * mask


class FrequencyShiftAugmentation(EEGAugmentation):
    """
    Frequency Shift Augmentation
    Random shift in frequency domain to simulate frequency changes
    """

    def __init__(self, probability: float = 0.5, max_shift_hz: float = 2.0, sampling_rate: float = 1000, **kwargs):
        super().__init__(probability, **kwargs)
        self.max_shift_hz = max_shift_hz
        self.sampling_rate = sampling_rate

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for processing
        eeg_np = eeg_data.cpu().numpy()
        augmented = np.zeros_like(eeg_np)

        # Process each sample and channel
        for i in range(eeg_np.shape[0]):
            for j in range(eeg_np.shape[1]):
                # Compute Fourier Transform
                signal_fft = np.fft.fft(eeg_np[i, j])
                n = len(signal_fft)
                freq = np.fft.fftfreq(n, d=1 / self.sampling_rate)

                # Random frequency shift
                shift_hz = random.uniform(-self.max_shift_hz, self.max_shift_hz)
                shift_samples = int(shift_hz * n / self.sampling_rate)

                # Apply frequency domain shift
                shifted_fft = np.roll(signal_fft, shift_samples)

                # Maintain Hermitian symmetry (property of Fourier Transform for real signals)
                shifted_fft = self._maintain_hermitian_symmetry(shifted_fft)

                # Inverse Fourier Transform
                augmented[i, j] = np.real(np.fft.ifft(shifted_fft))

        return torch.from_numpy(augmented).to(eeg_data.device)

    def _maintain_hermitian_symmetry(self, fft_signal):
        """Maintain Hermitian symmetry of Fourier Transform"""
        n = len(fft_signal)
        half_n = n // 2

        # For real signals, FFT satisfies conjugate symmetry
        fft_signal[half_n + 1:] = np.conj(fft_signal[1:half_n][::-1])
        return fft_signal


class AmplitudeScalingAugmentation(EEGAugmentation):
    """
    Amplitude Scaling Augmentation
    Randomly scale signal amplitude
    """

    def __init__(self, probability: float = 0.5, scale_range: Tuple[float, float] = (0.8, 1.2), **kwargs):
        super().__init__(probability, **kwargs)
        self.scale_range = scale_range

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        return eeg_data * scale


class TimeWarpingAugmentation(EEGAugmentation):
    """
    Time Warping Augmentation
    Non-linear warping of the time axis
    """

    def __init__(self, probability: float = 0.5, num_knots: int = 4, max_warp: float = 0.2, **kwargs):
        super().__init__(probability, **kwargs)
        self.num_knots = num_knots
        self.max_warp = max_warp

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_times = eeg_data.shape
        augmented = torch.zeros_like(eeg_data)

        for i in range(batch_size):
            # Generate warp function
            warp_function = self._generate_warp_function(n_times)

            # Apply same warping to each channel
            for j in range(n_channels):
                # Use linear interpolation for time warping
                original_times = torch.linspace(0, 1, n_times, device=eeg_data.device)
                warped_times = torch.from_numpy(warp_function).float().to(eeg_data.device)

                # Resample signal
                signal = eeg_data[i, j]
                warped_signal = self._resample_signal(signal, original_times, warped_times)
                augmented[i, j] = warped_signal

        return augmented

    def _generate_warp_function(self, n_times: int) -> np.ndarray:
        """Generate time warp function"""
        # Control points
        knots = np.linspace(0, n_times - 1, self.num_knots).astype(int)
        knot_values = np.array([k + random.uniform(-self.max_warp, self.max_warp) * n_times
                                for k in knots])

        # Clamp to valid range
        knot_values = np.clip(knot_values, 0, n_times - 1)

        # Linear interpolation to generate complete warp function
        warp_function = np.interp(np.arange(n_times), knots, knot_values)
        return warp_function

    def _resample_signal(self, signal: torch.Tensor, original_t: torch.Tensor, warped_t: torch.Tensor) -> torch.Tensor:
        """Resample signal"""
        # Use linear interpolation
        return torch.from_numpy(np.interp(warped_t.cpu().numpy(),
                                          original_t.cpu().numpy(),
                                          signal.cpu().numpy())).to(signal.device)


class SmoothTimeMaskAugmentation(EEGAugmentation):
    """
    Smooth Time Mask Augmentation
    Randomly mask a time window with smooth transitions
    """

    def __init__(self, probability: float = 0.5, mask_len_samples: int = 100,
                 min_masks: int = 1, max_masks: int = 3, **kwargs):
        super().__init__(probability, **kwargs)
        self.mask_len_samples = mask_len_samples
        self.min_masks = min_masks
        self.max_masks = max_masks

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_times = eeg_data.shape
        augmented = eeg_data.clone()

        for i in range(batch_size):
            # Randomly determine number of masks
            n_masks = random.randint(self.min_masks, self.max_masks)

            for _ in range(n_masks):
                # Randomly select start position
                start_pos = random.randint(0, n_times - self.mask_len_samples - 1)
                end_pos = start_pos + self.mask_len_samples

                # Create smooth mask (using cosine window)
                mask = torch.ones(n_times, device=eeg_data.device)
                window = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, self.mask_len_samples,
                                                             device=eeg_data.device)))
                mask[start_pos:end_pos] = window

                # Apply mask to all channels
                augmented[i, :, :] *= mask.unsqueeze(0)

        return augmented


class FTSTAugmentation(EEGAugmentation):
    """
    Fourier Transform Surrogate Augmentation
    Randomize phase information in frequency domain while preserving magnitude spectrum
    """

    def __init__(self, probability: float = 0.5, phase_noise_magnitude: float = 1.0, **kwargs):
        super().__init__(probability, **kwargs)
        self.phase_noise_magnitude = phase_noise_magnitude

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for processing
        eeg_np = eeg_data.cpu().numpy()
        augmented = np.zeros_like(eeg_np)

        for i in range(eeg_np.shape[0]):
            for j in range(eeg_np.shape[1]):
                # Compute Fourier Transform
                signal_fft = np.fft.fft(eeg_np[i, j])
                magnitude = np.abs(signal_fft)
                phase = np.angle(signal_fft)

                # Add random phase noise
                phase_noise = np.random.uniform(-self.phase_noise_magnitude,
                                                self.phase_noise_magnitude,
                                                len(phase))
                new_phase = phase + phase_noise

                # Reconstruct signal
                new_fft = magnitude * np.exp(1j * new_phase)
                augmented[i, j] = np.real(np.fft.ifft(new_fft))

        return torch.from_numpy(augmented).to(eeg_data.device)


class SpatialMixupAugmentation(EEGAugmentation):
    """
    Spatial Mixup Augmentation
    Mix EEG signals from different samples to enhance spatial feature robustness
    """

    def __init__(self, probability: float = 0.5, alpha: float = 0.2, mix_channels: bool = True, **kwargs):
        super().__init__(probability, **kwargs)
        self.alpha = alpha
        self.mix_channels = mix_channels

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        batch_size = eeg_data.shape[0]

        if batch_size < 2:
            return eeg_data

        # Randomly select samples to mix
        idx = torch.randperm(batch_size)
        mixed_data = eeg_data.clone()

        if self.mix_channels:
            # Channel-level mixing
            mix_ratio = torch.rand(batch_size, eeg_data.shape[1], 1, device=eeg_data.device) * self.alpha
            mixed_data = mix_ratio * eeg_data + (1 - mix_ratio) * eeg_data[idx]
        else:
            # Sample-level mixing
            mix_ratio = torch.rand(batch_size, 1, 1, device=eeg_data.device) * self.alpha
            mixed_data = mix_ratio * eeg_data + (1 - mix_ratio) * eeg_data[idx]

        return mixed_data


class CutoutAugmentation(EEGAugmentation):
    """
    Cutout Augmentation
    Randomly remove a small segment of signal and fill with zeros or mean value
    """

    def __init__(self, probability: float = 0.5, cutout_len: int = 50,
                 fill_value: str = 'zero', **kwargs):
        super().__init__(probability, **kwargs)
        self.cutout_len = cutout_len
        self.fill_value = fill_value  # 'zero', 'mean', 'noise'

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_times = eeg_data.shape
        augmented = eeg_data.clone()

        for i in range(batch_size):
            # Randomly select start position
            start_pos = random.randint(0, n_times - self.cutout_len - 1)
            end_pos = start_pos + self.cutout_len

            # Determine fill value
            if self.fill_value == 'zero':
                fill_val = 0
            elif self.fill_value == 'mean':
                fill_val = eeg_data[i].mean()
            else:  # 'noise'
                fill_val = torch.randn_like(eeg_data[i, :, start_pos:end_pos]) * 0.01

            augmented[i, :, start_pos:end_pos] = fill_val

        return augmented


class SpecAugmentAugmentation(EEGAugmentation):
    """
    SpecAugment-style Augmentation
    Mask in time-frequency domain, suitable for time-frequency plots or raw signals
    """

    def __init__(self, probability: float = 0.5, time_mask_len: int = 20,
                 freq_mask_len: int = 5, num_masks: int = 2, **kwargs):
        super().__init__(probability, **kwargs)
        self.time_mask_len = time_mask_len
        self.freq_mask_len = freq_mask_len
        self.num_masks = num_masks

    def apply(self, eeg_data: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_times = eeg_data.shape
        augmented = eeg_data.clone()

        # Time domain masking
        for i in range(batch_size):
            for _ in range(self.num_masks):
                start_pos = random.randint(0, n_times - self.time_mask_len - 1)
                end_pos = start_pos + self.time_mask_len
                augmented[i, :, start_pos:end_pos] = 0

        return augmented


class AugmentationComposer:
    """
    Augmentation Composer
    Combine multiple augmentation methods, apply sequentially or randomly
    """

    def __init__(self, augmentations: List[EEGAugmentation],
                 mode: str = 'sequential', random_order: bool = False):
        """
        Args:
            augmentations: List of augmentation methods
            mode: 'sequential' apply in order, 'random' randomly select one, 'all' apply all
            random_order: Whether to randomize order (only effective when mode='sequential')
        """
        self.augmentations = augmentations
        self.mode = mode
        self.random_order = random_order

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        augmented = eeg_data.clone()

        if self.mode == 'sequential':
            order = list(range(len(self.augmentations)))
            if self.random_order:
                random.shuffle(order)

            for idx in order:
                augmented = self.augmentations[idx](augmented)

        elif self.mode == 'random':
            # Randomly select one augmentation method
            aug = random.choice(self.augmentations)
            augmented = aug(augmented)

        elif self.mode == 'all':
            # Apply all augmentation methods
            for aug in self.augmentations:
                augmented = aug(augmented)

        return augmented

    def __len__(self):
        return len(self.augmentations)


class OnlineAugmentationWrapper(nn.Module):
    """
    Online Augmentation Wrapper
    Wrap augmentation methods as PyTorch module for convenient use in training pipeline
    """

    def __init__(self, augmentation_composer: AugmentationComposer):
        super().__init__()
        self.augmentation_composer = augmentation_composer

    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.augmentation_composer(eeg_data)
        return eeg_data

    def train(self, mode: bool = True):
        """Override train method to ensure augmentation only applied during training"""
        super().train(mode)
        return self


def create_augmentation_from_config(config: Dict[str, Any]) -> Optional[AugmentationComposer]:
    """
    Create data augmenter from configuration

    Args:
        config: Configuration dictionary, should contain 'data_augmentation' section

    Returns:
        AugmentationComposer instance, returns None if not enabled
    """
    aug_config = config.get('advanced', {}).get('data_augmentation', {})

    if not aug_config.get('enabled', False):
        return None

    augmentation_methods = []
    methods_config = aug_config.get('methods', [])

    for method_config in methods_config:
        method_name = method_config.get('name', '')
        probability = method_config.get('probability', 0.5)

        if method_name == 'gaussian_noise':
            augmentation = GaussianNoiseAugmentation(
                probability=probability,
                std=method_config.get('std', 0.01)
            )

        elif method_name == 'time_shift':
            augmentation = TimeShiftAugmentation(
                probability=probability,
                max_shift=method_config.get('max_shift', 10),
                mode=method_config.get('mode', 'circular')
            )

        elif method_name == 'channel_dropout':
            augmentation = ChannelDropoutAugmentation(
                probability=probability,
                dropout_rate=method_config.get('dropout_rate', 0.1)
            )

        elif method_name == 'amplitude_scaling':
            augmentation = AmplitudeScalingAugmentation(
                probability=probability,
                scale_range=method_config.get('scale_range', (0.8, 1.2))
            )

        elif method_name == 'smooth_time_mask':
            augmentation = SmoothTimeMaskAugmentation(
                probability=probability,
                mask_len_samples=method_config.get('mask_len_samples', 100),
                min_masks=method_config.get('min_masks', 1),
                max_masks=method_config.get('max_masks', 3)
            )

        elif method_name == 'cutout':
            augmentation = CutoutAugmentation(
                probability=probability,
                cutout_len=method_config.get('cutout_len', 50),
                fill_value=method_config.get('fill_value', 'zero')
            )

        elif method_name == 'freq_shift':
            augmentation = FrequencyShiftAugmentation(
                probability=probability,
                max_shift_hz=method_config.get('max_shift_hz', 2.0),
                sampling_rate=method_config.get('sampling_rate', 1000)
            )

        elif method_name == 'time_warping':
            augmentation = TimeWarpingAugmentation(
                probability=probability,
                num_knots=method_config.get('num_knots', 4),
                max_warp=method_config.get('max_warp', 0.2)
            )

        elif method_name == 'fts':
            augmentation = FTSTAugmentation(
                probability=probability,
                phase_noise_magnitude=method_config.get('phase_noise_magnitude', 1.0)
            )

        elif method_name == 'spatial_mixup':
            augmentation = SpatialMixupAugmentation(
                probability=probability,
                alpha=method_config.get('alpha', 0.2),
                mix_channels=method_config.get('mix_channels', True)
            )

        elif method_name == 'spec_augment':
            augmentation = SpecAugmentAugmentation(
                probability=probability,
                time_mask_len=method_config.get('time_mask_len', 20),
                freq_mask_len=method_config.get('freq_mask_len', 5),
                num_masks=method_config.get('num_masks', 2)
            )

        else:
            print(f"Warning: Unknown augmentation method '{method_name}', skipping")
            continue

        augmentation_methods.append(augmentation)
        print(f"Created augmentation method: {method_name} (probability: {probability})")

    if not augmentation_methods:
        return None

    composer_mode = aug_config.get('composer_mode', 'sequential')
    random_order = aug_config.get('random_order', False)

    composer = AugmentationComposer(
        augmentations=augmentation_methods,
        mode=composer_mode,
        random_order=random_order
    )

    return composer