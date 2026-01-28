import torch
import gc
import logging
import time
import psutil
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from functools import wraps


class MemoryManager:
    
    def __init__(self, device: Optional[torch.device] = None, 
                 cleanup_interval: int = 100,
                 log_level: str = 'INFO'):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.cleanup_interval = cleanup_interval
        self.batch_count = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.memory_history = []
        self.peak_memory = 0
        
    def get_memory_info(self) -> Dict[str, Any]:
        info = {
            'device': str(self.device),
            'timestamp': time.time()
        }
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            info.update({
                'allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
                'free_gb': (torch.cuda.get_device_properties(self.device).total_memory - 
                           torch.cuda.memory_allocated(self.device)) / 1e9,
                'total_gb': torch.cuda.get_device_properties(self.device).total_memory / 1e9,
                'memory_fraction': torch.cuda.memory_allocated(self.device) / 
                                 torch.cuda.get_device_properties(self.device).total_memory
            })
        else:
            memory_info = psutil.virtual_memory()
            info.update({
                'used_gb': memory_info.used / 1e9,
                'available_gb': memory_info.available / 1e9,
                'total_gb': memory_info.total / 1e9,
                'memory_percent': memory_info.percent
            })
            
        return info
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        before_info = self.get_memory_info()
        
        if self.device.type == 'cuda' and torch.cuda.is_available():

            torch.cuda.empty_cache()
            if force:
                torch.cuda.synchronize(self.device)
                gc.collect()
        else:
            gc.collect()
            
        after_info = self.get_memory_info()
        memory_saved = before_info.get('allocated_gb', 0) - after_info.get('allocated_gb', 0)
        self.logger.info(f"内存清理完成: 释放 {memory_saved:.2f} GB")
        
        return {
            'before': before_info,
            'after': after_info,
            'memory_saved_gb': memory_saved
        }
    
    def check_batch_cleanup(self) -> bool:

        self.batch_count += 1
        if self.batch_count >= self.cleanup_interval:
            self.batch_count = 0
            return True
        return False
    
    def batch_cleanup(self) -> Optional[Dict[str, Any]]:

        if self.check_batch_cleanup():
            return self.cleanup_memory(force=False)
        return None
    
    def log_memory_usage(self, stage: str = "unknown") -> Dict[str, Any]:
        info = self.get_memory_info()
        self.memory_history.append(info)

        if self.device.type == 'cuda':
            current_allocated = info.get('allocated_gb', 0)
            self.peak_memory = max(self.peak_memory, current_allocated)
            
        self.logger.info(f"[{stage}] resource allocation: {info.get('allocated_gb', info.get('used_gb', 0)):.2f} GB")
        
        return info
    
    def get_memory_summary(self) -> Dict[str, Any]:
        if not self.memory_history:
            return {}
            
        if self.device.type == 'cuda':
            allocated_values = [h.get('allocated_gb', 0) for h in self.memory_history]
            return {
                'peak_memory_gb': self.peak_memory,
                'avg_memory_gb': sum(allocated_values) / len(allocated_values),
                'final_memory_gb': allocated_values[-1],
                'memory_growth_gb': allocated_values[-1] - allocated_values[0] if len(allocated_values) > 1 else 0,
                'total_cleanups': len([h for h in self.memory_history if 'memory_saved_gb' in h])
            }
        else:
            used_values = [h.get('used_gb', 0) for h in self.memory_history]
            return {
                'avg_memory_gb': sum(used_values) / len(used_values),
                'final_memory_gb': used_values[-1],
                'memory_growth_gb': used_values[-1] - used_values[0] if len(used_values) > 1 else 0
            }


@contextmanager
def memory_context(stage: str = "operation", device: Optional[torch.device] = None):
    manager = MemoryManager(device=device)
    
    try:
        before_info = manager.log_memory_usage(f"{stage}_start")
        yield manager
        
    finally:
        after_info = manager.log_memory_usage(f"{stage}_end")
        manager.cleanup_memory(force=True)


def memory_monitor(interval: int = 10):

    def decorator(func):
        manager = MemoryManager(cleanup_interval=interval)
        @wraps(func)
        def wrapper(*args, **kwargs):
            cleanup_info = manager.batch_cleanup()
            result = func(*args, **kwargs)
            manager.log_memory_usage(func.__name__)
            return result

        wrapper.get_memory_summary = manager.get_memory_summary
        return wrapper
    
    return decorator


def safe_tensor_operation(operation_name: str = "tensor_op"):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with memory_context(operation_name) as manager:
                    result = func(*args, **kwargs)
                    return result
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"[{operation_name}] CUDA内存不足错误: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                raise
                
            except Exception as e:
                print(f"[{operation_name}] 操作失败: {e}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise
                
        return wrapper
    
    return decorator


def gradient_accumulation_safe(loss: torch.Tensor, accumulation_steps: int) -> torch.Tensor:

    if accumulation_steps > 1:
        loss_value = loss.item() if loss.numel() == 1 else loss.mean().item()
        loss = loss / accumulation_steps
        return loss
    else:
        return loss


def clear_model_cache(model: torch.nn.Module):
    if hasattr(model, 'zero_grad'):
        model.zero_grad(set_to_none=True)
    if hasattr(model, 'clear_cache'):
        model.clear_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    summary_lines = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        summary_lines.append(
            f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
            f"{total:.2f}GB total ({allocated/total*100:.1f}%)"
        )
    
    return "\n".join(summary_lines)

_global_memory_manager = None

def get_global_memory_manager() -> MemoryManager:
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def cleanup_all_memory():
    manager = get_global_memory_manager()
    return manager.cleanup_memory(force=True)


def log_memory_usage(stage: str = "unknown"):
    manager = get_global_memory_manager()
    return manager.log_memory_usage(stage)