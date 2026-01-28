import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from braindecode.models import (
    ShallowFBCSPNet, Deep4Net, EEGNet, EEGConformer, TCN,
    SleepStagerChambon2018, TIDNet, EEGInceptionERP, EEGInceptionMI,
    EEGTCNet, ATCNet, FBCNet, TSception
)
from braindecode.util import set_random_seeds


class EEGEncoder(nn.Module):
    """Base class for EEG encoders."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        embedding_dim = config.get('embedding_dim', 512)
        self.embedding_dim = int(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BraindecodeWrapperEncoder(EEGEncoder):
    """Wrapper for Braindecode models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        n_channels = int(config.get('n_channels', 128))
        n_timepoints = int(config.get('n_timepoints', 300))
        model_type = config.get('model_type', 'shallow').lower()
        dropout = float(config.get('dropout', 0.5))
        sfreq = float(config.get('sampling_rate', 100))

        model_params = config.get('model_params', {})

        if model_type == 'shallow':
            # ShallowFBCSPNet
            self.model = ShallowFBCSPNet(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                final_conv_length='auto',
                pool_time_stride=2
            )

        elif model_type == 'deep':
            # Deep4Net
            self.model = Deep4Net(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                final_conv_length='auto',
                pool_time_stride=2
            )

        elif model_type == 'eegnet':
            # EEGNet
            self.model = EEGNet(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                drop_prob=dropout
            )

        elif model_type == 'conformer':
            # EEGConformer
            self.model = EEGConformer(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                drop_prob=dropout,
                att_depth=model_params.get('att_depth', 3),
                att_heads=model_params.get('att_heads', 8),
                pool_time_stride=model_params.get('pool_time_stride', 2),
            )


        elif model_type == 'eegconformer':
            # EEGConformer
            self.model = EEGConformer(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                drop_prob=dropout,
                att_depth=model_params.get('att_depth', 4),  # Deep
                att_heads=model_params.get('att_heads', 8),
                pool_time_stride=model_params.get('pool_time_stride', 2),
            )


        elif model_type == 'tcn':
            # TCN
            self.model = TCN(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                n_filters=model_params.get('n_filters', 64),
                n_blocks=model_params.get('n_blocks', 6),
                kernel_size=model_params.get('kernel_size', 5),
                drop_prob=dropout
            )

        elif model_type == 'eegtcnet':
            from braindecode.models import EEGTCNet
            self.model = EEGTCNet(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                n_filters=model_params.get('n_filters', 32),
                kernel_size=model_params.get('kernel_size', 4),
                drop_prob=dropout,
                pooling=model_params.get('pooling', 2)
            )

        elif model_type == 'atcnet':
            # ATCNet (Attention Temporal Convolutional Network)
            from braindecode.models import ATCNet
            self.model = ATCNet(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                # ATCNet specific parameters
                n_windows=model_params.get('n_windows', 3),
                attention=model_params.get('attention', 'se'),  # 'se' or 'mhsa'
                drop_prob=dropout
            )

        elif model_type == 'contrawr':
            # ContraWR
            try:
                from braindecode.models import ContraWR
                base_model = ShallowFBCSPNet(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    final_conv_length='auto'
                )
                # Wrap with ContraWR
                self.model = ContraWR(
                    encoder=base_model,
                    projection_head_layers=model_params.get('projection_head_layers', 2),
                    projection_head_dim=model_params.get('projection_head_dim', 256)
                )
                print("Using ContraWR (contrastive learning framework)")
            except ImportError:
                print("Warning: ContraWR not available, using ShallowFBCSPNet")
                self.model = ShallowFBCSPNet(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    final_conv_length='auto'
                )

        elif model_type == 'fbcnet':
            # FBCNet (Filter Bank Common Spatial Pattern)
            try:
                from braindecode.models import FBCNet
                self.model = FBCNet(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    n_filterbanks=model_params.get('n_filterbanks', 9),
                    kernel_length=model_params.get('kernel_length', 32),
                    drop_prob=dropout
                )
                print("Using FBCNet (Filter Bank + CSP)")
            except ImportError:
                print("Warning: FBCNet not available, using Deep4Net")
                self.model = Deep4Net(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    final_conv_length='auto'
                )

        elif model_type == 'tsception':
            # TSception (multi-scale spatio-temporal feature extraction)
            try:
                from braindecode.models import TSception
                self.model = TSception(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    num_T=model_params.get('num_T', 7),
                    num_S=model_params.get('num_S', 6),
                    drop_prob=dropout
                )
                print("Using TSception (multi-scale spatio-temporal convolution)")
            except ImportError:
                print("Warning: TSception not available, using EEGInceptionERP")
                self.model = EEGInceptionERP(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    drop_prob=dropout
                )

        elif model_type == 'eeginception':
            # EEGInception (general version)
            try:
                from braindecode.models import EEGInceptionERP, EEGInceptionMI
                # Use ERP version as general encoder
                self.model = EEGInceptionERP(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    drop_prob=dropout,
                    sfreq=sfreq
                )
                print("Using EEGInception (Inception-style architecture)")
            except ImportError:
                print("Warning: EEGInception not available, using Deep4Net")
                self.model = Deep4Net(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    final_conv_length='auto'
                )

        elif model_type == 'tidnet':
            # TIDNet (Temporal and Spatial Convolution)
            try:
                from braindecode.models import TIDNet
                self.model = TIDNet(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    drop_prob=dropout,
                    temp_layers=model_params.get('temp_layers', 2),
                    spat_layers=model_params.get('spat_layers', 2)
                )
                print("Using TIDNet (temporal-spatial factorized convolution)")
            except ImportError:
                print("Warning: TIDNet not available, using EEGNet")
                self.model = EEGNet(
                    n_chans=n_channels,
                    n_outputs=self.embedding_dim,
                    n_times=n_timepoints,
                    drop_prob=dropout
                )

        elif model_type == 'sleep':
            # SleepStagerChambon2018 (sleep staging model, transferable)
            self.model = SleepStagerChambon2018(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                sfreq=sfreq,
                n_times=n_timepoints
            )

        else:

            print(f"Unsupported model type: {model_type}, using default EEGConformer")
            self.model = EEGConformer(
                n_chans=n_channels,
                n_outputs=self.embedding_dim,
                n_times=n_timepoints,
                drop_prob=dropout,
                att_depth=3,
                att_heads=8
            )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Using Braindecode {model_type} encoder")
        print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

        self._setup_output_projection()

    def _setup_output_projection(self):
        test_input = torch.randn(2, self.config.get('n_channels', 128),
                                 self.config.get('n_timepoints', 250))

        if next(self.model.parameters()).is_cuda:
            test_input = test_input.cuda()

        with torch.no_grad():
            try:
                test_output = self.model(test_input)
                output_dim = test_output.shape[-1]
            except RuntimeError as e:
                print(f"Warning: Model forward pass test failed ({e}), trying to infer output dimension")
                output_dim = self.embedding_dim

        if output_dim != self.embedding_dim:
            print(
                f"Model output dimension ({output_dim}) doesn't match target dimension ({self.embedding_dim}), adding projection layer")
            self.output_projection = nn.Linear(output_dim, self.embedding_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Note: braindecode models expect input as [batch, channels, timepoints]
        """
        # braindecode models handle raw EEG format directly
        features = self.model(x)

        # Flatten features (if needed)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        output = self.output_projection(features)
        return output


class PretrainedEncoderWrapper(EEGEncoder):
    """Wrapper for pretrained encoders with warmup strategy."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get pretrained configuration
        pretrained_config = config.get('pretrained', {})
        self.pretrained_enabled = pretrained_config.get('enabled', False)
        self.checkpoint_path = pretrained_config.get('checkpoint_path')
        self.freeze_pretrained = pretrained_config.get('freeze_pretrained', True)

        # Create base encoder - using only BraindecodeWrapperEncoder as other classes don't exist
        encoder_type = config.get('type', 'braindecode')

        # Use BraindecodeWrapperEncoder for all types since other encoder classes don't exist
        print(f"Note: Using BraindecodeWrapperEncoder as base encoder (original type: {encoder_type})")
        self.base_encoder = BraindecodeWrapperEncoder(config)

        base_dim = self.base_encoder.embedding_dim
        # Adapter layer (if dimension adjustment is needed)
        if base_dim != self.embedding_dim:
            self.adapter = nn.Linear(base_dim, self.embedding_dim)
            print(f"Adding adapter layer: {base_dim} -> {self.embedding_dim}")
        else:
            self.adapter = nn.Identity()


        self.loaded_pretrained = False
        if self.pretrained_enabled and self.checkpoint_path:
            resolved_path = self._resolve_checkpoint_path(self.checkpoint_path)
            if resolved_path:
                self.loaded_pretrained = self._load_pretrained_weights(resolved_path)
            else:
                print(f"Warning: Pretrained checkpoint not found: {self.checkpoint_path}")
                print("Will use random initialization + warmup strategy")
                self.loaded_pretrained = False
        else:
            if self.pretrained_enabled:
                print("Pretraining enabled but no checkpoint_path provided, using random initialization")
            else:
                print("Pretraining not enabled, using random initialization")
            self.loaded_pretrained = False

        # Freeze pretrained layers if requested
        if self.loaded_pretrained and self.freeze_pretrained:
            for param in self.base_encoder.parameters():
                param.requires_grad = False
            print("Frozen pretrained encoder parameters")

        # If no pretraining, use warmup strategy
        if not self.loaded_pretrained:
            self.use_warmup = config.get('initialization', {}).get('use_warmup', True)
            if self.use_warmup:
                print("Will use warmup strategy for encoder training")
                self._initialize_warmup_state()

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Optional[str]:

        if os.path.isabs(checkpoint_path) and os.path.exists(checkpoint_path):
            return checkpoint_path

        # Check multiple possible relative locations
        possible_locations = [
            checkpoint_path,  # Original path
            os.path.join(os.getcwd(), checkpoint_path),  # Current working directory
            os.path.join(os.path.dirname(__file__), checkpoint_path),  # Script directory
            os.path.join('experiments', checkpoint_path),  # experiments directory
            os.path.join('checkpoints', checkpoint_path),  # checkpoints directory
        ]

        # Add .pt extension if missing
        if not checkpoint_path.endswith(('.pt', '.pth', '.ckpt')):
            possible_locations.extend([
                checkpoint_path + '.pt',
                checkpoint_path + '.pth',
                checkpoint_path + '.ckpt'
            ])

        # Check each possible location
        for location in possible_locations:
            if os.path.exists(location):
                print(f"Found checkpoint: {location}")
                return location

        # If none found, return None
        return None

    def _load_pretrained_weights(self, checkpoint_path: str) -> bool:
        """Load pretrained weights"""
        try:
            print(f"Loading pretrained weights from: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Try different state dict key names
            state_dict_keys = ['model_state_dict', 'state_dict', 'eeg_encoder_state_dict',
                               'encoder_state_dict', 'eeg_encoder', 'encoder']
            state_dict = None

            for key in state_dict_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"Using state dict key: {key}")
                    break

            if state_dict is None:
                # If no standard key found, try loading the entire checkpoint
                print("Warning: No standard state dict key found, trying to load checkpoint directly")
                state_dict = checkpoint

            # Filter encoder-related weights
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # Match various possible key patterns
                if any(prefix in key for prefix in [
                    'eeg_encoder.', 'encoder.', 'base_encoder.',
                    'model.eeg_encoder.', 'model.encoder.',
                    'eeg_encoder_and_projection.'
                ]):
                    # Remove prefixes
                    new_key = key
                    for prefix in ['eeg_encoder.', 'encoder.', 'base_encoder.',
                                   'model.eeg_encoder.', 'model.encoder.',
                                   'eeg_encoder_and_projection.']:
                        if key.startswith(prefix):
                            new_key = key[len(prefix):]
                            break
                    encoder_state_dict[new_key] = value
                # If no prefix, might be the entire encoder state
                elif 'eeg' in key.lower() or 'encoder' in key.lower():
                    # Try using directly
                    encoder_state_dict[key] = value

            if encoder_state_dict:
                # Load weights, allowing non-strict matching
                try:
                    missing_keys, unexpected_keys = self.base_encoder.load_state_dict(
                        encoder_state_dict, strict=False
                    )

                    if missing_keys:
                        print(f"Missing keys: {missing_keys[:5]}... (total: {len(missing_keys)})")
                    if unexpected_keys:
                        print(f"Unexpected keys: {unexpected_keys[:5]}... (total: {len(unexpected_keys)})")

                    print(f"Successfully loaded pretrained weights, matched parameters: {len(encoder_state_dict)}")
                    return True
                except Exception as e:
                    print(f"Failed to load weights, trying name matching: {e}")
                    # Try manual name matching
                    return self._load_weights_by_name(encoder_state_dict)
            else:
                print("Warning: No encoder weights found in checkpoint")
                return False

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_weights_by_name(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Manually match weights by name"""
        try:
            model_state_dict = self.base_encoder.state_dict()

            # Create mapping dictionary
            loaded_keys = set()
            for name, param in model_state_dict.items():
                # Try to find matching key
                if name in state_dict:
                    # Direct match
                    model_state_dict[name] = state_dict[name]
                    loaded_keys.add(name)
                else:
                    # Try partial matching
                    for key in state_dict.keys():
                        if name in key or key in name:
                            if state_dict[key].shape == param.shape:
                                model_state_dict[name] = state_dict[key]
                                loaded_keys.add(name)
                                break

            self.base_encoder.load_state_dict(model_state_dict, strict=False)
            print(f"Manually matched and loaded {len(loaded_keys)}/{len(model_state_dict)} parameters")
            return len(loaded_keys) > 0

        except Exception as e:
            print(f"Manual matching failed: {e}")
            return False

    def _initialize_warmup_state(self):
        """Initialize warmup training state"""
        self.num_layers = sum(1 for _ in self.base_encoder.modules() if
                              isinstance(_, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.TransformerEncoderLayer)))

        # Initially freeze all base encoder parameters
        for param in self.base_encoder.parameters():
            param.requires_grad = False

        # Adapter layer is always trainable
        for param in self.adapter.parameters():
            param.requires_grad = True

        print(f"Warmup initialization complete, total layers: {self.num_layers}")

    def apply_warmup(self, current_epoch: int, warmup_epochs: int):
        """Apply warmup strategy based on current epoch"""
        if not hasattr(self, 'use_warmup') or not self.use_warmup:
            return

        if current_epoch < warmup_epochs:
            unfreeze_ratio = (current_epoch + 1) / warmup_epochs
            if current_epoch == 0:
                print(f"Epoch {current_epoch}: Training only adapter layer")
            elif current_epoch < warmup_epochs // 2:
                # Gradually unfreeze layers from the end
                layers_to_unfreeze = int(self.num_layers * unfreeze_ratio)
                # Collect all layers
                all_layers = []
                for module in self.base_encoder.modules():
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.TransformerEncoderLayer)):
                        all_layers.extend(list(module.parameters()))

                if layers_to_unfreeze > 0 and layers_to_unfreeze <= len(all_layers):
                    for param in all_layers[-layers_to_unfreeze:]:
                        param.requires_grad = True

                    print(f"Epoch {current_epoch}: Unfrozen last {layers_to_unfreeze} layers")
            else:
                # Unfreeze all layers
                for param in self.base_encoder.parameters():
                    param.requires_grad = True

                print(f"Epoch {current_epoch}: Unfrozen all layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_encoder(x)
        embedding = self.adapter(features)
        return embedding


def create_eeg_encoder(config: Dict[str, Any]) -> EEGEncoder:
    """Factory function to create EEG encoders based on configuration."""
    encoder_config = config.get('eeg_encoder', {})
    encoder_type = encoder_config.get('type', 'braindecode').lower()

    if encoder_type == 'pretrained':
        return PretrainedEncoderWrapper(encoder_config)
    elif encoder_type == 'braindecode':
        return BraindecodeWrapperEncoder(encoder_config)
    else:
        # Default to BraindecodeWrapperEncoder
        print(f"Unsupported encoder type: {encoder_type}, defaulting to BraindecodeWrapperEncoder")
        return BraindecodeWrapperEncoder(encoder_config)


def save_encoder_checkpoint(encoder: EEGEncoder, save_path: str, config: Dict[str, Any] = None):

    checkpoint = {
        'model_state_dict': encoder.state_dict(),
        'embedding_dim': encoder.embedding_dim if hasattr(encoder, 'embedding_dim') else 768,
        'config': config if config is not None else encoder.config if hasattr(encoder, 'config') else {},
        'encoder_type': encoder.__class__.__name__
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_encoder_checkpoint(checkpoint_path: str, config: Dict[str, Any]) -> EEGEncoder:

    encoder = create_eeg_encoder(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['state_dict'])
    else:
        encoder.load_state_dict(checkpoint)

    return encoder


__all__ = [
    'EEGEncoder',
    'BraindecodeWrapperEncoder',
    'PretrainedEncoderWrapper',
    'create_eeg_encoder',
    'save_encoder_checkpoint',
    'load_encoder_checkpoint',
]