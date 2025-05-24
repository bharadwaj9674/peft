"""
Cross-Modal Attention Network (CMAN) with Unified Modality Fusion
Implements multi-layer cross-modal attention with holistic modality interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedFusionLayer(nn.Module):
    """Stacked unified cross-modal attention layer with depth"""
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([self._create_layer(embed_dim, num_heads, ff_dim, dropout) 
                                   for _ in range(num_layers)])
        
    def _create_layer(self, embed_dim, num_heads, ff_dim, dropout):
        return nn.ModuleDict({
            'attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
            'ffn': nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout)
            ),
            'norm1': nn.LayerNorm(embed_dim),
            'norm2': nn.LayerNorm(embed_dim)
        })

    def forward(self, x):
        for layer in self.layers:
            # Multi-head attention
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        return x

class CMAN(nn.Module):
    """
    Enhanced CMAN with unified cross-modal fusion
    Handles dynamic number of input modalities
    """
    def __init__(
        self,
        text_encoder,
        audio_encoder,
        image_encoder,
        embedding_dim=512,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        
        # Input projections
        self.text_proj = nn.Linear(text_encoder.output_dim, embedding_dim)
        self.audio_proj = nn.Linear(audio_encoder.output_dim, embedding_dim)
        self.image_proj = nn.Linear(image_encoder.output_dim, embedding_dim)
        
        # Unified fusion transformer
        self.fusion_transformer = UnifiedFusionLayer(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Learnable temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Output projections
        self.output_proj = nn.ModuleDict({
            'text': nn.Linear(embedding_dim, embedding_dim),
            'audio': nn.Linear(embedding_dim, embedding_dim),
            'image': nn.Linear(embedding_dim, embedding_dim)
        })

    def _get_modality_features(self, inputs):
        """Extract and project features for present modalities"""
        features = {}
        if inputs.get('input_ids') is not None and inputs.get('attention_mask') is not None:
            features['text'] = self.text_proj(self.text_encoder(inputs['input_ids'], inputs['attention_mask']))
        if inputs.get('input_values') is not None:
            features['audio'] = self.audio_proj(self.audio_encoder(inputs['input_values']))
        if inputs.get('pixel_values') is not None:
            features['image'] = self.image_proj(self.image_encoder(inputs['pixel_values']))
        return features


    def forward(self, return_loss=False, **kwargs):
        """
        Forward pass with dynamic modality handling
        Kwargs can contain: 
        - text: (input_ids, attention_mask)
        - audio: input_values
        - image: pixel_values
        """
        # 1. Extract and project features for present modalities
        modality_features = self._get_modality_features(kwargs)
        modality_order = list(modality_features.keys())
        
        if len(modality_order) == 0:
            raise ValueError("No input modalities provided")
            
        # 2. Process single modality case
        if len(modality_order) == 1:
            mod = modality_order[0]
            emb = self.output_proj[mod](modality_features[mod])
            return {f"{mod}_embeddings": F.normalize(emb, p=2, dim=-1)}
        
        # 3. Unified fusion for multiple modalities
        # Stack features in consistent order: text -> audio -> image
        ordered_features = [modality_features[mod] for mod in ['text', 'audio', 'image'] if mod in modality_features]
        stacked_features = torch.stack(ordered_features, dim=1)  # (B, num_mods, D)
        
        # Apply deep fusion
        fused_features = self.fusion_transformer(stacked_features)

        # 4. Split and project outputs
        outputs = {}
        embeddings_dict = {}
        for idx, mod in enumerate([m for m in ['text', 'audio', 'image'] if m in modality_order]):
            emb = self.output_proj[mod](fused_features[:, idx])
            normalized_emb = F.normalize(emb, p=2, dim=-1)
            outputs[f"{mod}_embeddings"] = normalized_emb
            embeddings_dict[mod] = normalized_emb
            
        # 5. Calculate max-margin loss if requested
        if return_loss:
            outputs["loss"] = self._compute_bidirectional_max_margin_loss(embeddings_dict)
            
        return outputs
    
    def _compute_bidirectional_max_margin_loss(self, embeddings_dict, margin=0.2):
        """
        Bidirectional Max-Margin Ranking Loss for cross-modal alignment.
        This loss explicitly pushes positive pairs closer while separating 
        negative pairs by a margin.
        
        Args:
        embeddings_dict: Dictionary of {modality_name: embeddings}
        margin: Margin for triplet loss
        
        Returns:
        Loss tensor
        """
        
        losses = []
        modalities = list(embeddings_dict.keys())
        
        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                emb1, emb2 = embeddings_dict[mod1], embeddings_dict[mod2]
                
                # Compute similarity matrix (cosine similarity since embeddings are normalized)
                sim_matrix = torch.matmul(emb1, emb2.T)
                batch_size = emb1.size(0)
                
                # Diagonal elements are positive pairs
                pos_scores = torch.diag(sim_matrix)
                
                # Create mask for negative pairs (all non-diagonal elements)
                negative_mask = 1.0 - torch.eye(batch_size, device=sim_matrix.device)
                
                # Hard negative mining in both directions
                # For each anchor in mod1, find hardest negative in mod2
                hard_negatives_1 = (sim_matrix * negative_mask).max(dim=1)[0]
                # For each anchor in mod2, find hardest negative in mod1
                hard_negatives_2 = (sim_matrix.T * negative_mask).max(dim=1)[0]
                
                # Compute hinge loss in both directions
                loss_1 = torch.clamp(margin - pos_scores + hard_negatives_1, min=0).mean()
                loss_2 = torch.clamp(margin - pos_scores + hard_negatives_2, min=0).mean()
                
                # Combine losses
                losses.append((loss_1 + loss_2) / 2)
                
            # Average over all modality pairs
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=next(iter(embeddings_dict.values())).device)
        
    #     # 4. Split and project outputs
    #     outputs = {}
    #     for idx, mod in enumerate([m for m in ['text', 'audio', 'image'] if m in modality_order]):
    #         emb = self.output_proj[mod](fused_features[:, idx])
    #         outputs[f"{mod}_embeddings"] = F.normalize(emb, p=2, dim=-1)
        
    #     # 5. Calculate contrastive loss if requested
    #     if return_loss:
    #         outputs["loss"] = self._compute_contrastive_loss(list(outputs.values()))
        
    #     return outputs

    # def _compute_contrastive_loss(self, embeddings_list):
    #     """Dynamic NCE loss for variable modalities"""
    #     total_loss = 0.0
    #     pairs = 0
        
    #     for i in range(len(embeddings_list)):
    #         for j in range(i+1, len(embeddings_list)):
    #             loss = self._pairwise_nce_loss(embeddings_list[i], embeddings_list[j])
    #             total_loss += loss
    #             pairs += 1
                
    #     return total_loss / pairs if pairs > 0 else torch.tensor(0.0)

    # def _pairwise_nce_loss(self, emb1, emb2):
    #     """Symmetrical contrastive loss with learnable temperature"""
    #     logits = torch.matmul(emb1, emb2.T) / self.temperature.clamp(min=1e-8)
    #     targets = torch.arange(emb1.size(0), device=emb1.device)
    #     return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2