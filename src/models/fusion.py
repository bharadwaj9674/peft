"""
Fusion and projection modules for cross-modal retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    """
    Projection layer that maps encoder outputs to a common embedding space.
    """
    def __init__(self, input_dim, output_dim, dropout=0.1, norm=True):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        self.norm = norm
        
    def forward(self, x):
        """
        Forward pass through the projection layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Projected tensor of shape (batch_size, output_dim)
        """
        x = self.projection(x)
        
        # Normalize embeddings if requested
        if self.norm:
            x = F.normalize(x, p=2, dim=-1)
            
        return x


class CrossModalModel(nn.Module):
    """
    Cross-modal retrieval model that combines encoders with PEFT and projection layers.
    """
    def __init__(
        self,
        text_encoder,
        audio_encoder,
        image_encoder,
        embedding_dim=512,
        dropout=0.1,
        peft_config=None
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        
        # Create projection layers for each modality
        self.text_projection = ProjectionLayer(
            input_dim=text_encoder.output_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.audio_projection = ProjectionLayer(
            input_dim=audio_encoder.output_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.image_projection = ProjectionLayer(
            input_dim=image_encoder.output_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Store embedding dimension
        self.embedding_dim = embedding_dim
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode text inputs to the shared embedding space."""
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_features)
    
    def encode_audio(self, input_values):
        """Encode audio inputs to the shared embedding space."""
        audio_features = self.audio_encoder(input_values)
        return self.audio_projection(audio_features)
    
    def encode_image(self, pixel_values):
        """Encode image inputs to the shared embedding space."""
        image_features = self.image_encoder(pixel_values)
        return self.image_projection(image_features)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_values=None,
        pixel_values=None,
        return_loss=False
    ):
        """
        Forward pass through the cross-modal model.
        
        Args:
            input_ids: Text token ids
            attention_mask: Text attention mask
            input_values: Audio features
            pixel_values: Image pixel values
            return_loss: Whether to return the contrastive loss
            
        Returns:
            Dictionary of embeddings for each provided modality
            Optionally returns the contrastive loss if return_loss=True
        """
        outputs = {}
        
        # Encode text if provided
        if input_ids is not None:
            outputs["text_embeddings"] = self.encode_text(input_ids, attention_mask)
        
        # Encode audio if provided
        if input_values is not None:
            outputs["audio_embeddings"] = self.encode_audio(input_values)
        
        # Encode image if provided
        if pixel_values is not None:
            outputs["image_embeddings"] = self.encode_image(pixel_values)
        
        # Compute contrastive loss if requested and if we have multiple modalities
        if return_loss and len(outputs) > 1:
            loss = self._compute_contrastive_loss(outputs)
            outputs["loss"] = loss
        
        return outputs
    
    def _compute_contrastive_loss(self, embeddings_dict):
        """
        Compute contrastive loss between different modality embeddings.
        
        Args:
            embeddings_dict: Dictionary of embeddings for each modality
            
        Returns:
            Contrastive loss value
        """
        # Extract the embeddings
        embeddings_list = list(embeddings_dict.values())
        
        # Initialize loss
        total_loss = 0.0
        pairs = 0
        
        # Compute pairwise losses between all modalities
        for i in range(len(embeddings_list)):
            for j in range(i+1, len(embeddings_list)):
                # Compute the loss between these two modalities
                pair_loss = self._nce_loss(embeddings_list[i], embeddings_list[j])
                total_loss += pair_loss
                pairs += 1
        
        # Return average loss across all pairs
        return total_loss / max(1, pairs)
    
    def _nce_loss(self, embeddings1, embeddings2, temperature=0.07):
        """
        Noise Contrastive Estimation (NCE) loss, also known as InfoNCE.
        
        Args:
            embeddings1: First set of embeddings (batch_size, embedding_dim)
            embeddings2: Second set of embeddings (batch_size, embedding_dim)
            temperature: Temperature parameter for the softmax
            
        Returns:
            NCE loss value
        """
        # Compute similarity matrix
        logits = torch.matmul(embeddings1, embeddings2.T) / temperature
        
        # Targets are the diagonal elements (positive pairs)
        batch_size = embeddings1.size(0)
        targets = torch.arange(batch_size, device=embeddings1.device)
        
        # Compute cross-entropy loss in both directions
        loss1 = F.cross_entropy(logits, targets)
        loss2 = F.cross_entropy(logits.T, targets)
        
        # Return the average loss
        return (loss1 + loss2) / 2.0