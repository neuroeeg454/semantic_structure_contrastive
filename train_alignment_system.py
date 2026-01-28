import sys
import os
import argparse
import warnings
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models.alignment_model import EEGTextAlignmentModel
from utils.common_utils import (
    setup_device, set_seed, process_checkpoint_config, count_params,
    save_state_to_file, load_state_from_file, setup_env_and_dirs,
    setup_parallel_processing, calculate_class_weights
)
from utils.wandb_manager import WandBManager, EarlyStopping
from chineseeeg2 import EEGTextAlignmentDataModule

class AlignmentTrainer:
    """
    EEG-Text Alignment Trainer
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = config.get('task_type', 'retrieval')
        
        # Setup environment, directories, and random seed
        self.device, self.exp_dir = setup_env_and_dirs(self.config)
        self.gpu_num = int(self.config.get('gpu_num', 1))
        
        # Initialize Task Class
        self._init_task_class()
        
        # Initialize WandB
        self._init_wandb()
        
        # Process Checkpoint Config
        process_checkpoint_config(self.config)
        
        # Initialize Placeholders
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.classifier = None
        self.task_instance = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = None
        self.best_epoch = 0
        self.best_checkpoint_path = None
        
        # Training config
        training_config = config.get('training', {})
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.use_amp = training_config.get('use_amp', False)
        self.amp_dtype = training_config.get('amp_dtype', 'float16')
        
        if self.use_amp:
            self.scaler = GradScaler()


    def _init_task_class(self):
        """Initialize task class based on configuration"""
        elif self.task_type == 'binary_retrieval':
            from task.binary_retrieval_task import BinaryRetrievalTask
            self.task_class = BinaryRetrievalTask
        elif self.task_type == 'pos_aware_retrieval':
            from task.pos_aware_retrieval_task import POSAwareRetrievalTask
            self.task_class = POSAwareRetrievalTask
        elif self.task_type == 'binary_retrieval_threshold':
            from task.binary_retrieval_task_threshold import BinaryRetrievalTaskThreshold
            self.task_class = BinaryRetrievalTaskThreshold
        else:
            raise ValueError(f"unavailable task type: {self.task_type}. ")

    def _init_wandb(self):
        """Initialize WandB if enabled"""
        self.wandb_manager = None
        if self.config.get('use_wandb', True):
            self.wandb_manager = WandBManager(
                self.config,
                project_name=self.config.get('wandb_project', 'eeg-text-alignment'),
                task_type=self.task_type
            )

    def setup_data(self):
        """Setup data loaders"""
        try:
            data_module = EEGTextAlignmentDataModule(self.config)
            
            training_config = self.config.get('training', {})
            batch_size = int(training_config.get('batch_size', 32))
            num_workers = int(training_config.get('num_workers', 4))
            
            self.train_loader, self.val_loader, self.test_loader = data_module.create_dataloaders(
                batch_size=batch_size, num_workers=num_workers
            )
        except Exception as e:
            raise

    def _init_model_components(self):
        """Initialize Model, Task Instance, and Classifier"""
        self.model = EEGTextAlignmentModel(self.config)
        self.task_instance = self.task_class(self.config, self.device)
        self.task_instance.model = self.model
        self.model = self.model.to(self.device)
        
        # Classifier setup
        if hasattr(self.task_instance, 'classifier'):
            classifier_config = self.config.get('classifier', {})
            self.task_instance.create_classifier(
                input_dim=None,
                hidden_dim=int(classifier_config.get('hidden_dim', 256)),
                num_layers=int(classifier_config.get('num_layers', 1)),
                dropout_rate=float(classifier_config.get('dropout', 0.2))
            )
            self.classifier = self.task_instance.classifier
            if self.classifier:
                self.classifier = self.classifier.to(self.device)

    def _log_model_stats(self):
        """Log model parameters and structure"""
        total_params, trainable_params = count_params(self.model)
        if self.classifier:
            c_total, c_train = count_params(self.classifier)
            total_params += c_total
            trainable_params += c_train

        if self.wandb_manager:
            self.wandb_manager.log_model_summary(self.model, name="Main Model")
            if self.classifier:
                self.wandb_manager.log_model_summary(self.classifier, name="Classifier")

    def _setup_criterion(self):
        """Setup Loss Function and Class Weights"""
        self.criterion = self.task_instance.criterion
        
        # Dynamic Class Weights Calculation
        if self.config.get('use_class_weights', False) and \
           self.config.get('class_weights') == 'balanced':
            
            training_config = self.config.get('training', {})
            batch_size = int(training_config.get('batch_size', 128))
            
            weights_tensor = calculate_class_weights(
                self.train_loader.dataset,
                self.task_instance,
                self.device,
                batch_size=batch_size,
                num_workers=self.train_loader.num_workers,
                collate_fn=self.train_loader.collate_fn
            )
            
            if weights_tensor is not None:
                if hasattr(self.criterion, 'weight'):
                    self.criterion.weight = weights_tensor
                else:
                    label_smoothing = self.config.get('label_smoothing', 0.1)
                    self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
                    self.task_instance.criterion = self.criterion

    def _get_optimized_parameters(self, lr):
        """Get parameters to optimize based on configuration"""
        params_to_optimize = []
        freeze_text_encoder = bool(self.config.get('freeze_text_encoder', False))
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        
        if freeze_text_encoder and hasattr(model_ref, 'text_encoder'):
            for param in model_ref.text_encoder.parameters():
                param.requires_grad = False
                
        if self.classifier:
            train_classifier_only = self.config.get('train_classifier_only', True)
            if train_classifier_only:
                for param in model_ref.parameters():
                    param.requires_grad = False
                params_to_optimize.extend(self.classifier.parameters())
            else:
                classifier_config = self.config.get('classifier', {})
                alignment_lr = float(classifier_config.get('alignment_lr', lr * 0.1))
                classifier_lr = float(classifier_config.get('classifier_lr', lr))
                
                params_to_optimize.append({'params': model_ref.eeg_encoder.parameters(), 'lr': alignment_lr})
                if not freeze_text_encoder:
                    params_to_optimize.append({'params': model_ref.text_encoder.parameters(), 'lr': alignment_lr})
                
                other_params = [p for n, p in model_ref.named_parameters() 
                              if p.requires_grad and 
                              not any(p is pp for pp in model_ref.eeg_encoder.parameters()) and
                              not any(p is pp for pp in model_ref.text_encoder.parameters())]
                if other_params:
                    params_to_optimize.append({'params': other_params, 'lr': alignment_lr})
                    
                params_to_optimize.append({'params': self.classifier.parameters(), 'lr': classifier_lr})
        else:
            params_to_optimize.extend([p for p in model_ref.parameters() if p.requires_grad])
            
        return params_to_optimize

    def _setup_optimizer(self):
        """Setup optimizer, scheduler, and early stopping"""
        training_config = self.config.get('training', {})
        lr = float(training_config.get('learning_rate', 1e-4))
        weight_decay = float(training_config.get('weight_decay', 0.01))
        
        # Get parameters
        params_to_optimize = self._get_optimized_parameters(lr)
            
        self.optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        epochs = int(training_config.get('epochs', 100))
        min_lr = float(training_config.get('min_lr', 1e-6))
        
        self.monitor_metric = training_config.get('monitor_metric', 'loss')
        self.monitor_mode = training_config.get('mode', 'min' if self.monitor_metric == 'loss' else 'max')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader) * epochs, eta_min=min_lr)
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode=self.monitor_mode, factor=0.5, patience=5)
            
        # Early Stopping
        if training_config.get('use_early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=int(training_config.get('early_stopping_patience', 10)),
                mode=self.monitor_mode
            )
            
        self.grad_clip = float(training_config.get('grad_clip', 1.0))

    def _prepare_inputs(self, batch):
        """Prepare inputs for training/evaluation"""
        eeg_data = batch['eeg'].to(self.device)

        if self.classifier:
            return eeg_data, batch
            
        if 'text' in batch and isinstance(batch['text'], dict):
            text_data = {
                'input_ids': batch['text']['input_ids'].to(self.device),
                'attention_mask': batch['text']['attention_mask'].to(self.device),
                'text_str': batch.get('text_str', [])
            }
        elif 'text_str' in batch:
            text_data = {'text_str': batch['text_str']}
        else:
            text_data = {'text_str': [''] * len(eeg_data)}
            
        return eeg_data, text_data

    def _apply_encoder_warmup(self, current_epoch: int):
        """Apply warmup to encoder if applicable"""
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_ref, 'eeg_encoder') and hasattr(model_ref.eeg_encoder, 'apply_warmup'):
            eeg_config = self.config.get('eeg_encoder', {})
            warmup_epochs = eeg_config.get('initialization', {}).get('warmup_epochs', 10)
            if hasattr(model_ref.eeg_encoder, 'loaded_pretrained') and not model_ref.eeg_encoder.loaded_pretrained:
                model_ref.eeg_encoder.apply_warmup(current_epoch, warmup_epochs)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self._apply_encoder_warmup(epoch)
        self.model.train()
        if self.classifier:
            self.classifier.train()

        total_loss = 0.0
        total_batches = 0
        total_accuracy = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(pbar):
            eeg_data, second_input = self._prepare_inputs(batch)
            
            if self.classifier:
                # Classification Task
                labels = self.task_instance.get_labels(second_input)
                model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                eeg_features = model_ref.eeg_encoder(eeg_data)
                
                logits = self.classifier(eeg_features)
                loss = self.criterion(logits, labels)
                
                batch_metrics = self.task_instance.compute_metrics(logits, labels)
                batch_accuracy = batch_metrics.get('accuracy', 0.0)
            else:
                # Retrieval/Alignment Task
                model_for_forward = self.model.module if hasattr(self.model, 'module') else self.model
                outputs = model_for_forward(eeg_data, second_input)
                
                # Check if task instance has custom loss computation (e.g., POS-aware)
                if hasattr(self.task_instance, 'compute_loss'):
                    loss = self.task_instance.compute_loss(outputs['eeg_embedding'], outputs['text_embedding'], second_input.get('text_str', []))
                else:
                    loss_output = self.criterion(outputs['eeg_embedding'], outputs['text_embedding'])
                    loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                
                batch_metrics = self.task_instance.compute_metrics(outputs['eeg_embedding'], outputs['text_embedding'])
                batch_accuracy = float(batch_metrics.get('binary_acc', 0.0))

            loss = loss / self.gradient_accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_steps += 1
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                if self.grad_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.classifier:
                        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clip)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_steps = 0
                
                if isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            total_accuracy += batch_accuracy
            total_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'acc': batch_accuracy,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            if self.wandb_manager and batch_idx % 100 == 0:
                self.wandb_manager.log_gradient_flow(self.model, epoch)

        # Handle remaining steps
        if accumulation_steps > 0:
            if self.grad_clip > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self.classifier:
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clip)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()

        return {
            'loss': total_loss / max(total_batches, 1),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'accuracy': total_accuracy / max(total_batches, 1)
        }

    def validate(self) -> Dict[str, float]:
        val_metrics = self.task_instance.evaluate_on_dataloader(self.val_loader)
        

        return val_metrics

    def test(self) -> Dict[str, float]:
        test_results = self.task_instance.evaluate_on_dataloader(self.test_loader, return_embeddings=True, return_predictions=True)
        test_metrics = {k: v for k, v in test_results.items() if not isinstance(v, (torch.Tensor, list, np.ndarray, dict))}
        
        with open(self.exp_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
            
        self.task_instance.run_analysis(test_results)
        for k, v in test_metrics.items():
            print(f"{k}: {v}")
        return test_metrics

    def save_checkpoint(self, name: str, metrics: Dict[str, float]):
        """Save training checkpoint"""
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': model_ref.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        if self.classifier:
            classifier_ref = self.classifier.module if hasattr(self.classifier, 'module') else self.classifier
            state['classifier_state_dict'] = classifier_ref.state_dict()
            
        save_state_to_file(state, self.exp_dir / 'checkpoints' / f"{name}.pt")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = load_state_from_file(path, self.device)
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        model_ref.load_state_dict(checkpoint['model_state_dict'])
        
        if self.classifier and 'classifier_state_dict' in checkpoint:
            classifier_ref = self.classifier.module if hasattr(self.classifier, 'module') else self.classifier
            classifier_ref.load_state_dict(checkpoint['classifier_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']

    def train(self):
        """Main training loop"""
        self.setup_data()
        self._init_model_components()
        self.model, self.classifier = setup_parallel_processing(
            self.model, self.classifier, self.gpu_num
        )
        self._log_model_stats()
        self._setup_criterion()
        self._setup_optimizer()
        self.task_instance.model = self.model
        if hasattr(self.task_instance, 'classifier'):
            self.task_instance.classifier = self.classifier
        self.task_instance.criterion = self.criterion
        
        epochs = int(self.config.get('training', {}).get('epochs', 100))
        save_interval = int(self.config.get('training', {}).get('save_interval', 5))
        
        for epoch in range(self.current_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            # Scheduler Step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get(self.monitor_metric, val_metrics.get('loss')))
                
            # Logging
            if self.wandb_manager:
                self.wandb_manager.log_training_metrics(epoch, train_metrics, val_metrics, train_metrics['learning_rate'])
                
            # Save Best
            metric_val = val_metrics.get(self.monitor_metric, val_metrics.get('loss'))
            is_improved = False
            if self.best_val_score is None:
                is_improved = True
            elif self.monitor_mode == 'min':
                is_improved = metric_val < self.best_val_score
            else:
                is_improved = metric_val > self.best_val_score
                
            if is_improved:
                self.best_val_score = metric_val
                self.best_epoch = epoch
                self.save_checkpoint('best', val_metrics)
                
            # Regular Save
            if save_interval > 0 and epoch % save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}', val_metrics)
                
            # Early Stopping
            if self.early_stopping and self.early_stopping(metric_val):
                break
                
        # Final Test
        best_path = self.exp_dir / 'checkpoints' / 'best.pt'
        if best_path.exists():
            self.load_checkpoint(str(best_path))
        self.test()
        if self.wandb_manager:
            self.wandb_manager.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task_type', type=str)
    parser.add_argument('--gpu_ids', type=str)
    parser.add_argument('--pos_weights', type=str, help='JSON string of POS weights')
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.task_type:
        config['task_type'] = args.task_type
        
    if args.gpu_ids:
        config['gpu_ids'] = args.gpu_ids
        
    if args.pos_weights:
        try:
            weights = json.loads(args.pos_weights)
            config['pos_weights'] = weights
            # Force switch task type if weights are provided
            config['task_type'] = 'pos_aware_retrieval'
            print(f"Using POS Weights from command line: {weights}")
        except json.JSONDecodeError:
            print("Error: Failed to parse --pos_weights JSON string")
            
    trainer = AlignmentTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
