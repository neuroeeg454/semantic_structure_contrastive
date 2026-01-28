import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from utils.loss import ContrastiveLoss
from sklearn.metrics import roc_auc_score, f1_score

class BinaryRetrievalTask:
    """
    Traditional Binary Retrieval
    """
    def __init__(self, config: Dict[str, Any], device: str = None):
        self.config = config
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model = None
        self.criterion = None
        self.exp_dir = Path(config.get('exp_dir', 'experiments/binary_retrieval'))
        self.logger = logging.getLogger(__name__)
        temperature = float(self.config.get('model', {}).get('contrastive', {}).get('temperature', 0.07))
        self.criterion = ContrastiveLoss(temperature=temperature).to(self.device)

    def _extract_pos_neg(self, similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels_matrix = eye_mask.long()
        pos_sim = torch.diagonal(similarity_matrix)
        neg_mask = ~eye_mask
        neg_sim = similarity_matrix[neg_mask]
        
        return pos_sim, neg_sim, neg_mask, labels_matrix


    def compute_metrics(self, eeg_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Accuracy, F1, AUC, Top-K
        """
        with torch.no_grad():
            batch_size = eeg_embeddings.size(0)
            temperature = getattr(self.criterion, 'temperature', 0.07)
            logits = torch.matmul(eeg_embeddings, text_embeddings.T) / temperature
            labels = torch.arange(batch_size, device=self.device)
            
            pos_sim_vec, _, mask, labels_matrix = self._extract_pos_neg(logits)
            # ==============================
            # Binary Accuracy (Pairwise)
            # ==============================
            pos_sim = pos_sim_vec.unsqueeze(1) # [batch, 1]
            
            # EEG -> Text
            row_compare = (pos_sim > logits) & mask
            if batch_size > 1:
                binary_acc_eeg = row_compare.sum(dim=1).float() / (batch_size - 1)
                binary_acc_eeg = binary_acc_eeg.mean()
            else:
                binary_acc_eeg = torch.tensor(1.0, device=self.device)

            # Text -> EEG
            logits_T = logits.T
            col_compare = (pos_sim > logits_T) & mask
            if batch_size > 1:
                binary_acc_text = col_compare.sum(dim=1).float() / (batch_size - 1)
                binary_acc_text = binary_acc_text.mean()
            else:
                binary_acc_text = torch.tensor(1.0, device=self.device)
            
            binary_acc = (binary_acc_eeg + binary_acc_text) / 2
            
            # ==============================
            # Top-K Accuracy (Retrieval)
            # ==============================
            # EEG -> Text Retrieval
            # argsort descending
            _, indices = torch.sort(logits, dim=1, descending=True) # [batch, batch]
            targets = labels.view(-1, 1).expand_as(indices)
            hits = (targets == indices) # [batch, batch]
            
            r1 = hits[:, :1].sum().float() / batch_size
            r5 = hits[:, :5].sum().float() / batch_size
            r10 = hits[:, :10].sum().float() / batch_size
            
            # ==============================
            # AUC & F1 (Treat as Binary Classification)
            # ==============================
            flat_logits = logits.view(-1).cpu().numpy()
            flat_labels = labels_matrix.view(-1).cpu().numpy()
            flat_probs = 1 / (1 + np.exp(-flat_logits))
            flat_preds = (flat_probs > 0.5).astype(int)
            
            try:
                auc_score = roc_auc_score(flat_labels, flat_probs)
                f1 = f1_score(flat_labels, flat_preds)
            except ValueError:
                auc_score = 0.0
                f1 = 0.0

            # ==============================
            # UMA, MUS, SRS (Text-Text Metrics)
            # ==============================
            # Normalize for cosine similarity metrics
            eeg_norm = F.normalize(eeg_embeddings, p=2, dim=-1)
            text_norm = F.normalize(text_embeddings, p=2, dim=-1)
            
            # 1. Retrieval Step (using EEG vs Text similarity)
            # Find the text that is most similar to the EEG
            sim_matrix = torch.matmul(eeg_norm, text_norm.T)
            pred_indices = torch.argmax(sim_matrix, dim=1)
            retrieved_emb = text_norm[pred_indices]
            
            # 2. Calculate Similarity (Text vs Text)
            # Compare Retrieved Text Embedding with Ground Truth Text Embedding
            text_sim = torch.sum(retrieved_emb * text_norm, dim=1)
            
            # MUS: Mean Unit Similarity (Mean of Text-Text similarity)
            mus = text_sim.mean()
            text_sim_matrix = torch.matmul(text_norm, text_norm.T)
            eye_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
            avg_text_sim = text_sim_matrix[~eye_mask].mean()
            
            # UMA: Unit Matching Accuracy (Fraction of Text-Text similarity > threshold)
            uma_threshold = 0.8
            uma = (text_sim > uma_threshold).float().mean()
            
            # SRS: Sentence Reconstruction Similarity (Same as MUS in this context)
            srs = text_sim.mean()

            return {
                'binary_acc': binary_acc.item(),
                'binary_acc_eeg': binary_acc_eeg.item(),
                'binary_acc_text': binary_acc_text.item(),
                'R@1': r1.item(),
                'R@5': r5.item(),
                'R@10': r10.item(),
                'auc': auc_score,
                'f1': f1,
                'mus': mus.item(),
                'uma': uma.item(),
                'srs': srs.item(),
                'avg_text_sim': avg_text_sim.item()
            }


    def evaluate_on_dataloader(self, dataloader, return_embeddings: bool = False, return_predictions: bool = False, **kwargs) -> Dict[str, Any]:

        if self.model is None:
            raise ValueError("model loading failed")
        
        self.model.eval()
        model_for_forward = self.model.module if hasattr(self.model, 'module') else self.model
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = []
        
        all_eeg_embeddings = []
        all_text_embeddings = []
        all_text_strings = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="binary evaluation")
            for batch_idx, batch in enumerate(pbar):
                eeg_data = batch['eeg'].to(self.device)
                text_data = {
                    'input_ids': batch['text']['input_ids'].to(self.device),
                    'attention_mask': batch['text']['attention_mask'].to(self.device),
                    'text_str': batch.get('text_str', [''] * len(eeg_data))
                }

                outputs = model_for_forward(eeg_data, text_data)

                if isinstance(self.criterion, ContrastiveLoss):
                    batch_loss, _ = self.criterion(outputs['eeg_embedding'], outputs['text_embedding'])
                    batch_metrics = self.compute_metrics(outputs['eeg_embedding'], outputs['text_embedding'])
                else:
                     batch_loss = torch.tensor(0.0, device=self.device)
                     batch_metrics = {}

                total_loss += batch_loss.item()
                num_batches += 1
                
                all_metrics.append(batch_metrics)
                
                if return_embeddings:
                    all_eeg_embeddings.append(outputs['eeg_embedding'].cpu())
                    all_text_embeddings.append(outputs['text_embedding'].cpu())
                    all_text_strings.extend(text_data.get('text_str', []))
                
                pbar.set_postfix({
                    'loss': batch_loss.item(), 
                    'R@1': f"{batch_metrics.get('R@1', 0):.3f}",
                    'Acc': f"{batch_metrics.get('binary_acc', 0):.3f}"
                })

        avg_metrics = {}
        if all_metrics:
            keys = all_metrics[0].keys()
            for k in keys:
                avg_metrics[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
        
        avg_metrics['loss'] = total_loss / max(num_batches, 1)
        results = {**avg_metrics}
        
        if return_embeddings:
            if all_eeg_embeddings:
                results['eeg_embeddings'] = torch.cat(all_eeg_embeddings, dim=0)
                results['text_embeddings'] = torch.cat(all_text_embeddings, dim=0)
                results['text_strings'] = all_text_strings
        
        return results

    def run_analysis(self, test_results: Dict[str, Any], top_n: int = 20):
        import json

        if 'eeg_embeddings' not in test_results or 'text_embeddings' not in test_results:
            print("no embeddings data!!!")
            return

        eeg_embeddings = test_results['eeg_embeddings']
        text_embeddings = test_results['text_embeddings']
        text_strings = test_results.get('text_strings', [])
        
        analysis = {}
        with torch.no_grad():
            similarity = torch.matmul(eeg_embeddings, text_embeddings.T)
            topk_sim, topk_indices = torch.topk(similarity, min(top_n, similarity.size(1)), dim=1)
            
            batch_size = similarity.size(0)
            
            pos_similarities, neg_similarities, _, _ = self._extract_pos_neg(similarity)
            
            analysis = {
                'positive_mean_similarity': float(pos_similarities.mean().item()),
                'negative_mean_similarity': float(neg_similarities.mean().item()),
                'separation_score': float(pos_similarities.mean().item() - neg_similarities.mean().item()),
                'top_k_examples': []
            }

            for i in range(min(10, batch_size)):
                top_examples = []
                for j in range(min(5, topk_sim.size(1))):
                    idx = topk_indices[i, j].item()
                    top_examples.append({
                        'text': text_strings[idx] if idx < len(text_strings) else "",
                        'similarity': float(topk_sim[i, j].item()),
                        'is_positive': (idx == i)
                    })
                analysis['top_k_examples'].append({
                    'query_index': i,
                    'top_matches': top_examples
                })

        save_path = self.exp_dir / 'test_analysis.json'
        metrics = {k: v for k, v in test_results.items() if not isinstance(v, (torch.Tensor, list, np.ndarray, dict))}
        
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'analysis': analysis}, f, indent=2)
            
        print(f"results saved: {save_path}")
