import os
import json
import yaml
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# =============================================================================
# Setup Utilities
# =============================================================================

def setup_device(gpu_num: int, gpu_ids: list = None) -> torch.device:
    """Setup computing device (CPU/GPU)"""
    if gpu_num <= 0:
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return torch.device('cpu')
    
    if gpu_ids:
        if isinstance(gpu_ids, list):
            gpu_ids_str = ",".join(str(id) for id in gpu_ids)
        else:
            gpu_ids_str = str(gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    
    return torch.device('cuda:0')

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_env_and_dirs(config: Dict[str, Any]) -> tuple:
    """Setup device, seed, and directories"""
    gpu_num = int(config.get('gpu_num', 1))
    gpu_ids = config.get('gpu_ids', [])
    if isinstance(gpu_ids, str):
        gpu_ids = [int(id.strip()) for id in gpu_ids.split(',') if id.strip()]
    device = setup_device(gpu_num, gpu_ids)
    print(f"device: {device}")
    
    # Seed
    seed = int(config.get('seed', 42))
    set_seed(seed)
    
    # Directories
    exp_dir = Path(config.get('exp_dir', 'experiments'))
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"experiment direction: {exp_dir}")
    
    return device, exp_dir

def setup_parallel_processing(model: nn.Module, classifier: Optional[nn.Module], gpu_num: int) -> tuple:
    """Setup DataParallel if multiple GPUs are available"""
    if torch.cuda.is_available() and gpu_num > 1:
        model = nn.DataParallel(model)
        if classifier:
            classifier = nn.DataParallel(classifier)
    return model, classifier

# =============================================================================
# File Utilities
# =============================================================================

def process_checkpoint_config(config: Dict[str, Any]):
        """Process pretrained checkpoint configuration"""
        if 'eeg_encoder' in config:
            eeg_config = config['eeg_encoder']
            if 'pretrained' in eeg_config:
                pretrained_config = eeg_config['pretrained']
                checkpoint_path = pretrained_config.get('checkpoint_path')
                
                if checkpoint_path and pretrained_config.get('enabled', False):
                    pretrained_config['checkpoint_path'] = checkpoint_path
                else:
                    print(f"警告: 未找到checkpoint: {checkpoint_path}，将使用随机初始化")
                    pretrained_config['enabled'] = False

# =============================================================================
# Model Utilities
# =============================================================================

def count_params(model: nn.Module):
    """Count total and trainable parameters of a model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_state_to_file(state: Dict[str, Any], path: Path, verbose: bool = True):
    """Save training state to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if verbose:
        print(f"saving checkpoint to: {path}")

def load_state_from_file(path: str, device: torch.device, verbose: bool = True) -> Dict[str, Any]:
    """Load training state from file"""
    if verbose:
        print(f"loading checkpoint: {path}")
    return torch.load(path, map_location=device)


def calculate_class_weights(
    dataset, 
    task_instance, 
    device: torch.device, 
    batch_size: int = 128, 
    num_workers: int = 4,
    collate_fn = None
) -> Optional[torch.Tensor]:

    # Create a temporary loader
    temp_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    label_counts = {}
    total_samples = 0
    
    for batch in tqdm(temp_loader, desc="Computing Class Weights"):
        # Extract labels using task instance
        if hasattr(task_instance, 'get_labels'):
            labels = task_instance.get_labels(batch)
            valid_labels = labels[labels != -1]
            
            for label in valid_labels:
                l = label.item()
                label_counts[l] = label_counts.get(l, 0) + 1
                total_samples += 1
    
    if total_samples > 0:
        if hasattr(task_instance, 'tone_to_idx'):
            num_classes = len(task_instance.tone_to_idx)
        else:
            max_label = max(label_counts.keys()) if label_counts else 4
            num_classes = max(5, max_label + 1)
            
        weights_list = []
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            if count > 0:
                w = total_samples / (num_classes * count)
                weights_list.append(w)
            else:
                weights_list.append(1.0)
                
        weights_tensor = torch.tensor(weights_list, device=device, dtype=torch.float)
        # Normalize
        weights_tensor = weights_tensor / weights_tensor.mean()

        
        return weights_tensor
    else:
        return None
