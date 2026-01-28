import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from eeg_encoder import create_eeg_encoder
from text_encoder import create_text_encoder
from eeg_augmentation import create_augmentation_from_config, OnlineAugmentationWrapper

class EEGTextAlignmentModel(nn.Module):
    """
    EEG-Text alignment model
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__() 
        self.config = config
        self.task_type = config.get('task_type', 'retrieval')

        self.eeg_encoder = create_eeg_encoder(config)
        eeg_embedding_dim = self.eeg_encoder.embedding_dim

        self.augmentation = create_augmentation_from_config(config)
        if self.augmentation:
            self.augmentation = OnlineAugmentationWrapper(self.augmentation)

        self.is_single_modal = self.task_type in ['vad', 'tone']
        
        if not self.is_single_modal:
            text_config = config.get('text_encoder', {})

            if config.get('skip_text_encoder', False):
                self.text_encoder = nn.Identity()
                text_embedding_dim = int(text_config.get('embedding_dim', 768))
                self.text_encoder.embedding_dim = text_embedding_dim
            else:
                self.text_encoder = create_text_encoder(text_config)
                text_embedding_dim = self.text_encoder.embedding_dim
            model_config = config.get('model', {})
            contrastive_config = model_config.get('contrastive', {})
            temp_value = float(contrastive_config.get('temperature', 0.07))
            self.temperature = nn.Parameter(torch.tensor(temp_value))

            projection_config = model_config.get('projection', {})
            shared_dim = int(projection_config.get('shared_dim', 256))
            dropout_rate = float(projection_config.get('dropout', 0.1))

            self.eeg_projection = nn.Sequential(
                nn.Linear(eeg_embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, shared_dim),
                nn.LayerNorm(shared_dim)
            )

            self.text_projection = nn.Sequential(
                nn.Linear(text_embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, shared_dim),
                nn.LayerNorm(shared_dim)
            )
        else:
            print(f"{self.task_type} task: No text")

        self._init_weights()
        
    def _init_weights(self):
        modules_to_init = []
        if hasattr(self, 'eeg_projection'):
            modules_to_init.append(self.eeg_projection)
        if hasattr(self, 'text_projection'):
            modules_to_init.append(self.text_projection)
        if hasattr(self, 'text_encoder') and hasattr(self.text_encoder, 'parameters'):
            modules_to_init.append(self.text_encoder)
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1)
    
    def forward(self, eeg_data: torch.Tensor, text_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        eeg_data = eeg_data.float()
        if hasattr(self, 'augmentation') and self.augmentation is not None:
            eeg_data = self.augmentation(eeg_data)
        eeg_features = self.eeg_encoder(eeg_data)  # [batch, embedding_dim]
        if self.is_single_modal:
            if hasattr(self, 'eeg_projection'):
                eeg_projected = self.eeg_projection(eeg_features)
                return {
                    'eeg_features': eeg_features,
                    'eeg_embedding': F.normalize(eeg_projected, p=2, dim=-1),
                    'task_type': self.task_type
                }
            else:
                return {
                    'eeg_features': eeg_features,
                    'eeg_embedding': eeg_features,
                    'task_type': self.task_type
                }
        else:
            text_result = self.text_encoder.forward(text_data['text_str'], return_dict=True)
            text_features = text_result['embedding']
            
            if text_features.dtype != self.text_projection[0].weight.dtype:
                text_features = text_features.to(self.text_projection[0].weight.dtype)
            
            eeg_projected = self.eeg_projection(eeg_features)
            text_projected = self.text_projection(text_features)
            
            eeg_embedding = F.normalize(eeg_projected, p=2, dim=-1)
            text_embedding = F.normalize(text_projected, p=2, dim=-1)
            
            return {
                'eeg_embedding': eeg_embedding,
                'text_embedding': text_embedding,
                'eeg_features': eeg_features,
                'text_features': text_features,
                'temperature': self.temperature
            }
