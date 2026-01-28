import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_TMPDIR"] = "/tmp"
os.environ["WANDB_CACHE_DIR"] = "/tmp/wandb_cache"
os.environ["WANDB_DIR"] = "/tmp"

import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
import json
import io
import torch
from datetime import datetime
import tempfile
import base64

class WandBManager:
    def __init__(self, config: Dict[str, Any], 
                 project_name: str = "eeg-text-alignment",
                 offline_mode: bool = False,
                 task_type: str = "retrieval"):
        self.config = config
        self.project_name = project_name
        self.offline_mode = offline_mode
        self.task_type = task_type
        self.experiment_name = config.get('experiment_name', 
                                         f"eeg-text-align-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.temp_dir = tempfile.mkdtemp(prefix="wandb_temp_")
        self._init_wandb()
        self.train_history = []
        self.val_history = []
        self.test_metrics = {}
        
        # Track global step for gradient logging to avoid conflicts
        self.global_step = 0

    
    def _init_wandb(self):
        wandb_kwargs = {
            'project': self.project_name,
            'name': self.experiment_name,
            'config': self.config,
            'group': self.config.get('experiment_group', 'default'),
            'tags': self.config.get('tags', ['eeg', 'text', 'alignment']),
            'notes': self.config.get('experiment_notes', 'EEG-Text Embedding Alignment Experiment'),
        }
        
        if self.offline_mode:
            wandb_kwargs['mode'] = 'offline'

        try:
            self.run = wandb.init(**wandb_kwargs)
        except Exception as e:
            print(f"WandB initialization failed, using offline mode: {e}")
            wandb_kwargs['mode'] = 'offline'
            self.run = wandb.init(**wandb_kwargs)

        self.exp_dir = Path(self.temp_dir)
    
    def log_config(self, config: Dict[str, Any]):
        wandb.config.update(config)

    def log_training_metrics(self, 
                           epoch: int, 
                           train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float],
                           learning_rate: Optional[float] = None):

        log_dict = {
            'epoch': epoch,
            'train/loss': train_metrics.get('loss', 0.0),
            'val/loss': val_metrics.get('loss', 0.0),
        }

        for key, value in train_metrics.items():
            if key != 'loss':
                log_dict[f'train/{key}'] = value

        for key, value in val_metrics.items():
            if key != 'loss':
                log_dict[f'val/{key}'] = value

        if learning_rate is not None:
            log_dict['train/lr'] = learning_rate

        wandb.log(log_dict, step=epoch)

        self.train_history.append({
            'epoch': epoch,
            **train_metrics
        })
        self.val_history.append({
            'epoch': epoch,
            **val_metrics
        })
    
    def log_test_metrics(self, test_metrics: Dict[str, float]):
        self.test_metrics = test_metrics
        test_log_dict = {
            'test/final_loss': test_metrics.get('loss', 0.0),
        }

        test_log_dict.update({
            'test/top1_acc': test_metrics.get('retrieval_top1', 0.0),
            'test/top5_acc': test_metrics.get('retrieval_top5', 0.0),
            'test/mrr': test_metrics.get('mrr', 0.0),
        })

        for key, value in test_metrics.items():
            if key not in ['loss', 'retrieval_top1', 'retrieval_top5', 'mrr', 
                          'accuracy', 'macro_f1', 'precision', 'recall', 'f1', 'auc']:
                test_log_dict[f'test/{key}'] = value
        
        wandb.log(test_log_dict)
        
        print(f"测试结果已记录: {test_metrics}")
    
    def log_model_checkpoint(self, 
                           model_state_dict: Dict[str, torch.Tensor],
                           optimizer_state_dict: Dict[str, Any],
                           epoch: int,
                           metrics: Dict[str, float],
                           filename: str = 'checkpoint.pt'):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'metrics': metrics,
            'config': self.config
        }

        checkpoints_dir = self.exp_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoints_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        print(f"模型检查点已保存到本地: {checkpoint_path}")
    
    def log_embeddings(self, 
                      eeg_embeddings: torch.Tensor,
                      text_embeddings: torch.Tensor,
                      text_labels: List[str],
                      epoch: int,
                      name: str = 'embeddings'):
        sample_size = min(1000, len(eeg_embeddings))
        indices = np.random.choice(len(eeg_embeddings), sample_size, replace=False)
        eeg_emb = eeg_embeddings.cpu().numpy()[indices]
        text_emb = text_embeddings.cpu().numpy()[indices]
        sample_labels = [text_labels[i] for i in indices]
        data = []
        for i in range(sample_size):
            data.append([
                sample_labels[i],
                eeg_emb[i].tolist(),
                text_emb[i].tolist()
            ])

        columns = ['text', 'eeg_embedding', 'text_embedding']
        table = wandb.Table(data=data, columns=columns)

        wandb.log({
            f'{name}/epoch_{epoch}': table
        })
        
        print(f"Embedding已记录: {sample_size} 个样本")
    
    def log_confusion_matrix(self, 
                           y_true: List[str],
                           y_pred: List[str],
                           class_names: List[str],
                           title: str = "Confusion Matrix"):

        from sklearn.metrics import confusion_matrix
        import numpy as np

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        unique_y_true = np.unique(y_true)
        unique_y_pred = np.unique(y_pred)
        all_unique = set(unique_y_true).union(set(unique_y_pred))
        ordered_unique = sorted(list(all_unique))
        
        # Select appropriate class names based on actual labels found
        available_class_names = []
        for label in ordered_unique:
            label_int = int(label)
            if label_int < len(class_names):
                available_class_names.append(class_names[label_int])
            else:
                available_class_names.append(str(label))

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true.tolist(),
                preds=y_pred.tolist(),
                class_names=available_class_names,
                title=title
            )
        })
    
    def log_retrieval_analysis(self,
                         similarity_matrix,
                         text_labels: List[str],
                         epoch: int):

        try:
            if isinstance(similarity_matrix, torch.Tensor):
                sim_matrix = similarity_matrix.cpu().numpy()
            elif isinstance(similarity_matrix, np.ndarray):
                sim_matrix = similarity_matrix
            else:
                try:
                    sim_matrix = np.array(similarity_matrix)
                except:
                    print(f"error: unable to process {type(similarity_matrix)}")
                    return

            n = min(1000, sim_matrix.shape[0])
            sim_matrix = sim_matrix[:n, :n]
            
            ranks = []
            for i in range(n):
                sim_scores = sim_matrix[i]
                sorted_indices = np.argsort(sim_scores)[::-1]
                rank = np.where(sorted_indices == i)[0][0] + 1 if i in sorted_indices else n + 1
                ranks.append(rank)

            mean_rank = np.mean(ranks)
            median_rank = np.median(ranks)
            top1_acc = np.mean([1 if r == 1 else 0 for r in ranks])
            top5_acc = np.mean([1 if r <= 5 else 0 for r in ranks])
            top10_acc = np.mean([1 if r <= 10 else 0 for r in ranks])

            wandb.log({
                f'retrieval/epoch_{epoch}/mean_rank': mean_rank,
                f'retrieval/epoch_{epoch}/median_rank': median_rank,
                f'retrieval/epoch_{epoch}/top1_acc': top1_acc,
                f'retrieval/epoch_{epoch}/top5_acc': top5_acc,
                f'retrieval/epoch_{epoch}/top10_acc': top10_acc,
            })

            if n <= 100:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(sim_matrix, cmap='viridis')
                ax.set_xlabel('Text Index')
                ax.set_ylabel('EEG Index')
                ax.set_title(f'Similarity Matrix - Epoch {epoch}')
                plt.colorbar(im, ax=ax)
                wandb.log({
                    f"retrieval/similarity_matrix_epoch_{epoch}": wandb.Image(fig)
                })
                plt.close(fig)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
    def log_training_curves(self):
        if not self.train_history or not self.val_history:
            return
        epochs = [h['epoch'] for h in self.train_history]
        if 'loss' in self.train_history[0] and 'loss' in self.val_history[0]:
            train_loss = [h.get('loss', 0) for h in self.train_history]
            val_loss = [h.get('loss', 0) for h in self.val_history]
            
            # LinePlot
            data = [[epoch, tloss, vloss] for epoch, tloss, vloss in zip(epochs, train_loss, val_loss)]
            table = wandb.Table(data=data, columns=["epoch", "train_loss", "val_loss"])
            
            wandb.log({
                "loss_curves": wandb.plot.line(
                    table, "epoch", ["train_loss", "val_loss"],
                    title="Training and Validation Loss"
                )
            })

        if self.task_type == 'retrieval' and 'retrieval_top1' in self.val_history[0] and 'retrieval_top5' in self.val_history[0]:
            val_top1 = [h.get('retrieval_top1', 0) for h in self.val_history]
            val_top5 = [h.get('retrieval_top5', 0) for h in self.val_history]
            
            data = [[epoch, top1, top5] for epoch, top1, top5 in zip(epochs, val_top1, val_top5)]
            table = wandb.Table(data=data, columns=["epoch", "top1_acc", "top5_acc"])
            
            wandb.log({
                "accuracy_curves": wandb.plot.line(
                    table, "epoch", ["top1_acc", "top5_acc"],
                    title="Retrieval Accuracy"
                )
            })
        elif self.task_type == 'classification' and 'accuracy' in self.val_history[0] and 'macro_f1' in self.val_history[0]:
            val_accuracy = [h.get('accuracy', 0) for h in self.val_history]
            val_macro_f1 = [h.get('macro_f1', 0) for h in self.val_history]
            
            data = [[epoch, acc, f1] for epoch, acc, f1 in zip(epochs, val_accuracy, val_macro_f1)]
            table = wandb.Table(data=data, columns=["epoch", "accuracy", "macro_f1"])
            
            wandb.log({
                "accuracy_curves": wandb.plot.line(
                    table, "epoch", ["accuracy", "macro_f1"],
                    title="Classification Accuracy and F1"
                )
            })
        elif self.task_type in ['binary_retrieval', 'vad'] and 'accuracy' in self.val_history[0] and 'f1' in self.val_history[0]:
            val_accuracy = [h.get('accuracy', 0) for h in self.val_history]
            val_f1 = [h.get('f1', 0) for h in self.val_history]
            
            data = [[epoch, acc, f1] for epoch, acc, f1 in zip(epochs, val_accuracy, val_f1)]
            table = wandb.Table(data=data, columns=["epoch", "accuracy", "f1"])
            
            wandb.log({
                "accuracy_curves": wandb.plot.line(
                    table, "epoch", ["accuracy", "f1"],
                    title="Binary Classification Accuracy and F1"
                )
            })
    
    def log_model_summary(self, model: torch.nn.Module, name: str = "Model"):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        clean_name = name.lower().replace(' ', '_').replace('-', '_')
        wandb.log({
            f'{clean_name}/total_parameters': total_params,
            f'{clean_name}/trainable_parameters': trainable_params,
            f'{clean_name}/non_trainable_parameters': total_params - trainable_params,
        })
        

        summary_lines = [
            f"model parameter summary:",
            f"total parameter: {total_params:,}",
            f"trainable parameter: {trainable_params:,}",
            f"frozen: {total_params - trainable_params:,}",
        ]
        

        summary_lines.append("\n10 layers info:")
        layer_count = 0
        for name, param in model.named_parameters():
            if layer_count < 10:
                summary_lines.append(f"{name}: {param.numel():,} params, shape={list(param.shape)}, trainable={param.requires_grad}")
                layer_count += 1
        
        summary_text = "\n".join(summary_lines)

        wandb.log({
            'model/summary': wandb.Html(f"<pre>{summary_text}</pre>")
        })
        
        print(f"模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    def log_gradient_flow(self, model: torch.nn.Module, epoch: int):

        gradients = []
        grad_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients.append(grad_norm)
                grad_names.append(name)
        
        if gradients:

            wandb.log({
                f'gradients/epoch_{epoch}/mean': np.mean(gradients),
                f'gradients/epoch_{epoch}/std': np.std(gradients),
                f'gradients/epoch_{epoch}/max': np.max(gradients),
                f'gradients/epoch_{epoch}/min': np.min(gradients),
                f'gradients/epoch_{epoch}/median': np.median(gradients),
            }, step=epoch)

            if len(gradients) > 10:
                indices = np.argsort(gradients)[-10:]
                for idx in indices:
                    wandb.log({
                        f'gradients/epoch_{epoch}/top_gradients/{grad_names[idx]}': gradients[idx]
                    }, step=epoch)
    
    def create_summary_report(self):

        report_lines = [
            f"# EEG-Text Alignment Experiment Report",
            f"",
            f"## Experiment Information",
            f"- **Experiment Name**: {self.experiment_name}",
            f"- **Project**: {self.project_name}",
            f"- **Task Type**: {self.task_type}",
            f"- **Status**: {'Running' if wandb.run else 'Completed'}",
            f"",
            f"## Configuration Summary",
            f"- Task Level: {self.config.get('task_level', 'word')}",
            f"- Modality: {self.config.get('modality', 'reading')}",
            f"- Model: EEG-Text Alignment Model",
            f"- Embedding Dimension: {self.config.get('embedding_dim', 768)}",
            f"- Batch Size: {self.config.get('batch_size', 32)}",
            f"- Learning Rate: {self.config.get('learning_rate', 1e-4)}",
            f"- Epochs: {self.config.get('epochs', 100)}",
            f"",
            f"## Training Statistics",
        ]
        
        if self.train_history:
            report_lines.append(f"- Total Training Epochs: {len(self.train_history)}")

        if self.val_history:
            if self.task_type == 'retrieval':
                val_top1 = [h.get('retrieval_top1', 0) for h in self.val_history]
                val_top5 = [h.get('retrieval_top5', 0) for h in self.val_history]
                if val_top1:
                    report_lines.append(f"- Best Validation Top-1: {max(val_top1):.4f}")
                if val_top5:
                    report_lines.append(f"- Best Validation Top-5: {max(val_top5):.4f}")
            elif self.task_type == 'classification':
                val_accuracy = [h.get('accuracy', 0) for h in self.val_history]
                val_macro_f1 = [h.get('macro_f1', 0) for h in self.val_history]
                if val_accuracy:
                    report_lines.append(f"- Best Validation Accuracy: {max(val_accuracy):.4f}")
                if val_macro_f1:
                    report_lines.append(f"- Best Validation Macro F1: {max(val_macro_f1):.4f}")
            elif self.task_type in ['binary_retrieval', 'vad']:
                val_accuracy = [h.get('accuracy', 0) for h in self.val_history]
                val_f1 = [h.get('f1', 0) for h in self.val_history]
                if val_accuracy:
                    report_lines.append(f"- Best Validation Accuracy: {max(val_accuracy):.4f}")
                if val_f1:
                    report_lines.append(f"- Best Validation F1 Score: {max(val_f1):.4f}")
        
        report_lines.append(f"")
        report_lines.append(f"## Test Results")
        if self.test_metrics:
            for metric, value in self.test_metrics.items():
                try:
                    num_value = float(value)
                    report_lines.append(f"- {metric}: {num_value:.6f}")
                except (ValueError, TypeError):
                    report_lines.append(f"- {metric}: {value}")
        
        report = "\n".join(report_lines)

        wandb.log({
            "experiment_report": wandb.Html(f"<pre>{report}</pre>")
        })
        
        return report
    
    def finish(self):
        try:
            self.log_training_curves()
            self.create_summary_report()
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            wandb.finish()
            print(f" {self.experiment_name} finished")
            
        except Exception as e:
            print(f"Wandb error: {e}")
            try:
                wandb.finish()
            except:
                pass


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.compare = lambda x, y: x < y - min_delta
            self.best_score = float('inf')
        else:  # mode == 'max'
            self.compare = lambda x, y: x > y + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            if self.verbose:
                print(f"EarlyStopping: {self.best_score:.6f} -> {score:.6f}")
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping")
        
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.early_stop = False
        if self.verbose:
            print("EarlyStopping")

