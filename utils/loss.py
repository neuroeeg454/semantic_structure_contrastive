
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
import argparse
from text_encoder import CachedTextEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    InfoNCE
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_semantic_positives: bool = False,
        semantic_threshold: float = 0.8,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_semantic_positives = use_semantic_positives
        self.semantic_threshold = semantic_threshold
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, eeg_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = eeg_embeddings.size(0)
        logits = torch.matmul(eeg_embeddings, text_embeddings.T) / self.temperature
        logits = torch.clamp(logits, min=-100, max=100)
        labels = torch.arange(batch_size, device=eeg_embeddings.device)
        
        if not self.use_semantic_positives:
            loss_eeg = self.cross_entropy(logits, labels)
            loss_text = self.cross_entropy(logits.T, labels)
            loss = (loss_eeg + loss_text) / 2
            return loss, {}

        text_norm = F.normalize(text_embeddings, p=2, dim=-1)
        text_sim = torch.matmul(text_norm, text_norm.T)
        pos_mask = text_sim > float(self.semantic_threshold)

        eye = torch.eye(batch_size, dtype=torch.bool, device=eeg_embeddings.device)
        pos_mask = pos_mask | eye

        row_log_denom = torch.logsumexp(logits, dim=1)
        row_log_num = torch.logsumexp(logits.masked_fill(~pos_mask, float("-inf")), dim=1)
        loss_eeg = -(row_log_num - row_log_denom).mean()

        logits_t = logits.T
        pos_mask_t = pos_mask.T
        col_log_denom = torch.logsumexp(logits_t, dim=1)
        col_log_num = torch.logsumexp(logits_t.masked_fill(~pos_mask_t, float("-inf")), dim=1)
        loss_text = -(col_log_num - col_log_denom).mean()

        loss = (loss_eeg + loss_text) / 2
        metrics = {
            "avg_pos_per_query": pos_mask.float().sum(dim=1).mean().item(),
        }
        return loss, metrics

    
class BinaryCrossEntropyLoss(nn.Module):

    
    def __init__(self, temperature: float = 0.07, pos_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    
    def forward(self, eeg_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_embeddings: EEG embedding [batch, dim]
            text_embeddings: 文本embedding [batch, dim]
            
        Returns:
            Binary Cross Entropy Loss
        """
        batch_size = eeg_embeddings.size(0)
        logits = torch.matmul(eeg_embeddings, text_embeddings.T) / self.temperature
        labels = torch.eye(batch_size, device=eeg_embeddings.device)
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        loss = self.bce_loss(logits_flat, labels_flat)
        
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PairwiseBCELoss(nn.Module):

    def __init__(self, classifier_head: Optional[nn.Module] = None):
        super().__init__()
        self.classifier_head = classifier_head
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, eeg_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            eeg_embeddings: [batch, dim]
            text_embeddings: [batch, dim]
            
        Returns:
            loss: scalar tensor
            metrics: dict containing accuracy details
        """
        batch_size = eeg_embeddings.size(0)
        device = eeg_embeddings.device
        pos_sim = torch.sum(eeg_embeddings * text_embeddings, dim=-1, keepdim=True)

        neg_indices = (torch.arange(batch_size, device=device).unsqueeze(1) + 
                       torch.randint(1, batch_size, (batch_size, 1), device=device)) % batch_size
        neg_text_embeddings = text_embeddings[neg_indices.view(-1)]
        neg_sim = torch.sum(eeg_embeddings * neg_text_embeddings, dim=-1, keepdim=True)

        if self.classifier_head is not None and hasattr(self.classifier_head, 'scale'):
            pos_logits = pos_sim * self.classifier_head.scale + self.classifier_head.bias
            neg_logits = neg_sim * self.classifier_head.scale + self.classifier_head.bias
        else:

            pos_logits = pos_sim * 10.0
            neg_logits = neg_sim * 10.0

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        final_logits = torch.cat([pos_logits, neg_logits], dim=0)
        final_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        loss = self.bce(final_logits, final_labels)
        with torch.no_grad():
            probs = torch.sigmoid(final_logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == final_labels).float().mean()
            pos_acc = (preds[:batch_size] == 1).float().mean()
            neg_acc = (preds[batch_size:] == 0).float().mean()
            preds_np = preds.cpu().numpy()
            labels_np = final_labels.cpu().numpy()
            if len(np.unique(labels_np)) > 1:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels_np, preds_np, average='binary', zero_division=0
                )
            else:
                precision = recall = f1 = 0.0
            
        metrics = {
            'accuracy': accuracy.item(),
            'pos_acc': pos_acc.item(),
            'neg_acc': neg_acc.item(),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'logits': final_logits,
            'labels': final_labels,
            'probs': probs
        }
        
        return loss, metrics
