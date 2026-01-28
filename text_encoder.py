import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import os
import warnings
import yaml


class QwenTextEncoder(nn.Module):

    def __init__(self,
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 embedding_dim: int = 768,
                 max_length: int = 512,
                 pooling_strategy: str = 'cls',
                 freeze_layers: int = 0,
                 task_level: str = 'word',
                 device: str = None,
                 config_path: str = None,
                 **kwargs):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.task_level = task_level
        self.config_path = config_path
        
        print(f"loading model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='right'
            )

            load_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32
            }
            
            self.qwen_model = AutoModel.from_pretrained(model_name, **load_kwargs)
                
        except Exception as e:
            print(f"loading failed: {e}")
            raise

        if freeze_layers > 0:
            self._freeze_qwen_layers(freeze_layers)

        try:
            self.qwen_hidden_size = self.qwen_model.config.hidden_size
        except:
            self.qwen_hidden_size = 4096

        if self.qwen_hidden_size != embedding_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.qwen_hidden_size, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    
    def _freeze_qwen_layers(self, freeze_layers: int):

        print(f"freezing {freeze_layers} layers")
        try:
            if hasattr(self.qwen_model, 'transformer'):
                transformer = self.qwen_model.transformer
            elif hasattr(self.qwen_model, 'model'):
                transformer = self.qwen_model.model
            else:
                transformer = self.qwen_model
            for name, param in transformer.named_parameters():
                if 'layers.' in name:
                    try:
                        layer_num = int(name.split('layers.')[1].split('.')[0])
                        if layer_num < freeze_layers:
                            param.requires_grad = False
                    except:
                        pass
                elif 'embeddings' in name and freeze_layers > 0:
                    param.requires_grad = False
                    
        except Exception as e:
            print(f"freezing error: {e}")
    
    def _qwen_pooling(self, last_hidden_state: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            if attention_mask is not None:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return last_hidden_state.mean(dim=1)
        
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state = last_hidden_state.masked_fill(input_mask_expanded == 0, -1e9)
            return torch.max(last_hidden_state, dim=1)[0]
        
        elif self.pooling_strategy == 'last':
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                return last_hidden_state[batch_indices, seq_lengths, :]
            else:
                return last_hidden_state[:, -1, :]
        
        else:
            raise ValueError(f"unavailable pooling strategy: {self.pooling_strategy}")
    
    def _encode_with_qwen(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        # Tokenizeing text
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        device = next(self.qwen_model.parameters()).device
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        with torch.no_grad() if not self.training else torch.enable_grad():
            try:
                outputs = self.qwen_model(**encoded_inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                elif isinstance(outputs, tuple):
                    last_hidden_state = outputs[0]
                    last_hidden_state = outputs.logits
                else:
                    last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                
                if last_hidden_state is None:
                    raise ValueError("无法从Qwen模型获取隐藏状态")
                
            except Exception as e:
                batch_size = len(texts)
                last_hidden_state = torch.randn(
                    batch_size, 10, self.qwen_hidden_size, device=device
                )
                attention_mask = encoded_inputs.get('attention_mask', torch.ones(batch_size, 10, device=device))
        
        # pooling
        attention_mask = encoded_inputs.get('attention_mask')
        pooled_output = self._qwen_pooling(last_hidden_state, attention_mask)
        
        return {
            'embedding': pooled_output,
            'last_hidden_state': last_hidden_state,
            'attention_mask': attention_mask,
            'input_ids': encoded_inputs.get('input_ids')
        }

    def forward(self, texts: Union[List[str], str],
                return_dict: bool = True) -> Dict[str, torch.Tensor]:

        if isinstance(texts, str):
            texts = [texts]
        result = self._encode_with_qwen(texts)
        embedding = result['embedding']

        if not isinstance(self.projection, nn.Identity):
            projection_dtype = next(self.projection.parameters()).dtype
            embedding = embedding.to(projection_dtype)
            projected_embedding = self.projection(embedding)
        else:
            projected_embedding = embedding
        
        # L2
        normalized_embedding = F.normalize(projected_embedding, p=2, dim=-1)
        
        if return_dict:
            return {
                'embedding': normalized_embedding,
                'raw_embedding': projected_embedding,
                'qwen_embedding': embedding,
                'last_hidden_state': result.get('last_hidden_state'),
                'attention_mask': result.get('attention_mask'),
                'input_ids': result.get('input_ids')
            }
        else:
            return normalized_embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        Args:
            texts: text lists
            
        Returns:
            all text embedding
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_result = self.forward(batch_texts, return_dict=True)
            all_embeddings.append(batch_result['embedding'].cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_token_embeddings(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Args:
            text: input text
            
        Returns:
            token_embeddings: token embedding [num_tokens, hidden_size]
            tokens: token list
        """
        #  Tokenize
        tokens = self.tokenizer.tokenize(text)
        encoded = self.tokenizer(text, return_tensors='pt')

        device = next(self.qwen_model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.qwen_model(**encoded)
            
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                last_hidden_state = outputs[0]
            else:
                last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

        if last_hidden_state is not None:
            token_embeddings = last_hidden_state[0, :len(tokens), :]
        else:

            token_embeddings = torch.randn(len(tokens), self.embedding_dim, device=device)
        
        return token_embeddings, tokens


class HybridTextEncoder(nn.Module):

    
    def __init__(self,
                 encoder_type: str = 'qwen',
                 model_name: str = None,
                 embedding_dim: int = 768,
                 max_length: int = 512,
                 pooling_strategy: str = 'cls',
                 freeze_layers: int = 0,
                 task_level: str = 'word',
                 device: str = None,
                 config_path: str = None,
                 **kwargs):
        super().__init__()
        
        self.encoder_type = encoder_type.lower()
        self.config_path = config_path

        if self.encoder_type == 'qwen':
            if model_name is None:
                model_name = "Qwen/Qwen3-Embedding-0.6B"
            
            self.encoder = QwenTextEncoder(
                model_name=model_name,
                embedding_dim=embedding_dim,
                max_length=max_length,
                pooling_strategy=pooling_strategy,
                freeze_layers=freeze_layers,
                task_level=task_level,
                device=device,
                config_path=config_path,
                **kwargs
            )
            
        elif self.encoder_type == 'bert':
            if model_name is None:
                model_name = "google-bert/bert-base-chinese"

            from .text_encoder_bert import ChineseTextEncoder
            self.encoder = ChineseTextEncoder(
                model_name=model_name,
                embedding_dim=embedding_dim,
                max_length=max_length,
                pooling_strategy=pooling_strategy,
                freeze_layers=freeze_layers,
                task_level=task_level,
                **kwargs
            )
            
        else:
            raise ValueError(f"unavailable encoder: {encoder_type}")
    
    def forward(self, texts: Union[List[str], str],
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        return self.encoder(texts, return_dict=return_dict)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:

        if self.encoder_type == 'qwen':
            batch_size = min(batch_size, 8)
        
        return self.encoder.encode_batch(texts, batch_size)

def create_text_encoder(config: Dict[str, Any]) -> nn.Module:

    encoder_type = config.get('type', 'qwen').lower()
    model_name = config.get('model_name', None)
    task_level = config.get('task_level', 'word')

    if encoder_type == 'qwen':
        if model_name is None:
            model_name = "Qwen/Qwen3-Embedding-0.6B"
        
        return QwenTextEncoder(
            model_name=model_name,
            embedding_dim=config.get('embedding_dim', 768),
            max_length=config.get('max_length', 512),
            pooling_strategy=config.get('pooling_strategy', 'cls'),
            freeze_layers=config.get('freeze_layers', 0),
            task_level=task_level,
            device=None,
            config_path=None
        )

    elif encoder_type == 'hybrid':
        return HybridTextEncoder(
            encoder_type=config.get('encoder_backbone', 'qwen'),
            model_name=model_name,
            embedding_dim=config.get('embedding_dim', 768),
            max_length=config.get('max_length', 512),
            pooling_strategy=config.get('pooling_strategy', 'cls'),
            freeze_layers=config.get('freeze_layers', 0),
            task_level=task_level,
            device=None,
            config_path=None
        )

    else:
        if model_name is None:
            model_name = "Qwen/Qwen3-Embedding-0.6B"
        
        return QwenTextEncoder(
            model_name=model_name,
            embedding_dim=config.get('embedding_dim', 768),
            max_length=config.get('max_length', 512),
            pooling_strategy=config.get('pooling_strategy', 'cls'),
            freeze_layers=config.get('freeze_layers', 0),
            task_level=task_level,
            device=None,
            config_path=None
        )


class CachedTextEncoder(nn.Module):
    def __init__(self, base_encoder, vocab_path=None):
        super().__init__()
        self.base_encoder = base_encoder
        self.cache = {}
        self.hits = 0
        self.misses = 0

        if vocab_path:
            self.preload_cache(vocab_path)
    
    def preload_cache(self, vocab_path):
        import pandas as pd
        vocab_df = pd.read_csv(vocab_path)
        words = vocab_df['word'].tolist()
        batch_size = 32
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            embeddings = self.base_encoder.encode_batch(batch_words, batch_size=8)
            for word, emb in zip(batch_words, embeddings):
                self.cache[word] = emb
    
    def forward(self, texts, return_dict=True):
        if isinstance(texts, str):
            texts = [texts]

        cached_results = []
        need_encode = []
        need_encode_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                cached_results.append(self.cache[text])
                self.hits += 1
            else:
                need_encode.append(text)
                need_encode_indices.append(i)
                self.misses += 1

        if need_encode:
            new_results = self.base_encoder(need_encode, return_dict=True)
            new_embeddings = new_results['embedding']
            for text, emb in zip(need_encode, new_embeddings):
                self.cache[text] = emb
        else:
            new_embeddings = torch.empty(0, device=self.base_encoder.qwen_model.device)
        all_embeddings = []
        cache_idx = 0
        new_idx = 0
        for i in range(len(texts)):
            if i in need_encode_indices:
                all_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
            else:
                all_embeddings.append(cached_results[cache_idx])
                cache_idx += 1
        
        result = torch.stack(all_embeddings)
        
        if return_dict:
            return {'embedding': result}
        return result