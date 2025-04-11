"""
Pre-trained encoder models for different modalities:
- Text: RoBERTa
- Audio: AST (Audio Spectrogram Transformer)
- Image: ViT (Vision Transformer)
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, AutoModel
from transformers import ViTModel
import torchaudio


class TextEncoder(nn.Module):
    """
    Text encoder based on RoBERTa model.
    """
    def __init__(self, model_name="roberta-base", freeze_base=True, output_dim=768):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Token ids (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]


class AudioEncoder(nn.Module):
    """
    Audio encoder based on AST (Audio Spectrogram Transformer) model.
    """
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", freeze_base=True, output_dim=768):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_values):
        """
        Forward pass through the audio encoder.
        
        Args:
            input_values: Audio features (batch_size, seq_len, feat_dim)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        outputs = self.model(input_values=input_values)
        
        # Use [CLS] token representation or mean pooling
        return outputs.last_hidden_state[:, 0, :]


class ImageEncoder(nn.Module):
    """
    Image encoder based on ViT (Vision Transformer) model.
    """
    def __init__(self, model_name="google/vit-base-patch16-224", freeze_base=True, output_dim=768):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values):
        """
        Forward pass through the image encoder.
        
        Args:
            pixel_values: Image pixel values (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        outputs = self.model(pixel_values=pixel_values)
        
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]