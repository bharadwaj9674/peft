"""
Loss functions for cross-modal contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent),
    also known as InfoNCE or contrastive loss.
    
    This is a popular loss function for contrastive learning.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings1, embeddings2):
        """
        Compute NT-Xent loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (batch_size, embedding_dim)
            embeddings2: Second set of embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss value
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Get batch size
        batch_size = embeddings1.size(0)
        
        # Compute similarity matrix
        # Each element [i, j] is the dot product between normalized embeddings
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(batch_size, device=embeddings1.device)
        
        # Compute loss for both directions
        loss1 = self.criterion(similarity_matrix, labels)
        loss2 = self.criterion(similarity_matrix.T, labels)
        
        # Return the average loss
        return (loss1 + loss2) / 2.0


class MultimodalContrastiveLoss(nn.Module):
    """
    Contrastive loss for multiple modalities.
    
    Computes pairwise contrastive loss between all provided modality embeddings.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ntxent = NTXentLoss(temperature=temperature)
    
    def forward(self, embeddings_dict):
        """
        Compute contrastive loss between multiple modality embeddings.
        
        Args:
            embeddings_dict: Dictionary of embeddings for each modality
                             {modality_name: embeddings}
                             
        Returns:
            Total contrastive loss
        """
        # Extract modalities and embeddings
        modalities = list(embeddings_dict.keys())
        
        # Check if we have at least two modalities
        if len(modalities) < 2:
            return torch.tensor(0.0, device=next(iter(embeddings_dict.values())).device)
        
        # Initialize loss
        total_loss = 0.0
        pair_count = 0
        
        # Compute loss for each pair of modalities
        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                mod1 = modalities[i]
                mod2 = modalities[j]
                
                # Get embeddings
                emb1 = embeddings_dict[mod1]
                emb2 = embeddings_dict[mod2]
                
                # Ensure both embeddings have the same batch size
                if emb1.size(0) != emb2.size(0):
                    # Skip this pair if batch sizes don't match
                    continue
                
                # Compute loss for this pair
                pair_loss = self.ntxent(emb1, emb2)
                
                # Add to total loss
                total_loss += pair_loss
                pair_count += 1
        
        # Return average loss
        return total_loss / max(pair_count, 1)


class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss for cross-modal embeddings.
    
    This loss encourages the embeddings of corresponding pairs to be closer
    than non-corresponding pairs by a margin.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss between anchor, positive, and negative embeddings.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss value
        """
        return self.triplet_loss(anchor, positive, negative)
    
    def compute_batch_all(self, embeddings1, embeddings2):
        """
        Compute triplet loss over all possible triplets in the batch.
        
        Args:
            embeddings1: First set of embeddings (batch_size, embedding_dim)
            embeddings2: Second set of embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss value
        """
        # Get batch size
        batch_size = embeddings1.size(0)
        
        # Compute all pairwise distances
        distances = torch.cdist(embeddings1, embeddings2, p=2)
        
        # Compute loss for each anchor
        loss = 0.0
        valid_triplets = 0
        
        for i in range(batch_size):
            # Positive distance is the distance to the corresponding embedding
            pos_dist = distances[i, i]
            
            # Negative distances are distances to non-corresponding embeddings
            neg_dists = torch.cat([distances[i, :i], distances[i, i+1:]])
            
            # Compute loss for valid triplets (where negative distance < positive distance + margin)
            valid_mask = neg_dists < pos_dist + self.margin
            
            if valid_mask.sum() > 0:
                triplet_loss = torch.relu(pos_dist - neg_dists[valid_mask] + self.margin)
                loss += triplet_loss.mean()
                valid_triplets += 1
        
        # Return average loss
        return loss / max(valid_triplets, 1)