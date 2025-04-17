"""
Enhanced multimodal encoders with performance optimizations:
- Improved text, audio, and image encoders
- Cross-modal alignment techniques
- Projection and fusion mechanisms
- Contrastive learning support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaModel, AutoModel, ViTModel
import torchaudio


class TextEncoder(nn.Module):
    """
    Enhanced text encoder based on RoBERTa model with improved pooling
    and optional adapter layers.
    """
    def __init__(self, 
                 model_name="roberta-base", 
                 freeze_base=True, 
                 output_dim=768,
                 pooling_strategy="cls",
                 use_adapter=True,
                 adapter_dim=128,
                 num_layers_to_use=4,
                 dropout=0.1):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name)
        self.output_dim = output_dim
        self.pooling_strategy = pooling_strategy
        self.use_adapter = use_adapter
        self.num_layers_to_use = num_layers_to_use
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Enable output_hidden_states for access to all layer outputs
        self.model.config.output_hidden_states = True
        
        # Adapters for the last few layers if enabled
        if use_adapter:
            self.adapters = nn.ModuleList([
                AdapterLayer(self.model.config.hidden_size, adapter_dim, dropout)
                for _ in range(num_layers_to_use)
            ])
        
        # Projection to desired output dimension if different from model dimension
        if self.model.config.hidden_size != output_dim:
            self.projector = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projector = nn.Identity()
        
        # Layer weights for weighted pooling across multiple layers
        if num_layers_to_use > 1:
            self.layer_weights = nn.Parameter(torch.ones(num_layers_to_use) / num_layers_to_use)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the text encoder with enhanced pooling strategies.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get outputs from the last N layers
        if self.num_layers_to_use > 1:
            # Get hidden states from all layers
            hidden_states = outputs.hidden_states
            # Take the last num_layers_to_use layers
            last_layers = hidden_states[-self.num_layers_to_use:]
            
            # Apply adapters if enabled
            if self.use_adapter:
                adapted_layers = []
                for i, layer_output in enumerate(last_layers):
                    adapted_layers.append(self.adapters[i](layer_output))
                last_layers = adapted_layers
            
            # Weighted sum of layer outputs
            norm_weights = F.softmax(self.layer_weights, dim=0)
            weighted_sum = torch.zeros_like(last_layers[0])
            for i, layer_output in enumerate(last_layers):
                weighted_sum += norm_weights[i] * layer_output
            
            # Apply pooling strategy
            if self.pooling_strategy == "cls":
                pooled_output = weighted_sum[:, 0]
            elif self.pooling_strategy == "mean":
                # Mean pooling (use attention mask to ignore padding)
                expanded_mask = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(weighted_sum * expanded_mask, 1)
                sum_mask = torch.sum(expanded_mask, 1)
                pooled_output = sum_embeddings / (sum_mask + 1e-9)
            elif self.pooling_strategy == "max":
                # Max pooling (use attention mask to ignore padding)
                expanded_mask = attention_mask.unsqueeze(-1).float()
                weighted_sum = weighted_sum * expanded_mask - 1e9 * (1 - expanded_mask)
                pooled_output = torch.max(weighted_sum, dim=1)[0]
            else:
                pooled_output = weighted_sum[:, 0]  # Default to CLS
        else:
            # Use only the last layer with simple pooling
            last_hidden = outputs.last_hidden_state
            
            # Apply adapter if enabled
            if self.use_adapter:
                last_hidden = self.adapters[0](last_hidden)
            
            # Apply pooling strategy
            if self.pooling_strategy == "cls":
                pooled_output = last_hidden[:, 0]
            elif self.pooling_strategy == "mean":
                # Mean pooling (use attention mask to ignore padding)
                expanded_mask = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(last_hidden * expanded_mask, 1)
                sum_mask = torch.sum(expanded_mask, 1)
                pooled_output = sum_embeddings / (sum_mask + 1e-9)
            elif self.pooling_strategy == "max":
                # Max pooling (use attention mask to ignore padding)
                expanded_mask = attention_mask.unsqueeze(-1).float()
                last_hidden = last_hidden * expanded_mask - 1e9 * (1 - expanded_mask)
                pooled_output = torch.max(last_hidden, dim=1)[0]
            else:
                pooled_output = last_hidden[:, 0]  # Default to CLS
        
        # Project to desired output dimension
        return self.projector(pooled_output)


class AudioEncoder(nn.Module):
    """
    Enhanced audio encoder based on AST with additional processing
    and spectral augmentation during training.
    """
    def __init__(self, 
                 model_name="MIT/ast-finetuned-audioset-10-10-0.4593", 
                 freeze_base=True, 
                 output_dim=768,
                 pooling_strategy="cls",
                 use_adapter=True,
                 adapter_dim=128,
                 use_specaug=True,
                 dropout=0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim
        self.pooling_strategy = pooling_strategy
        self.use_adapter = use_adapter
        self.use_specaug = use_specaug
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Enable output_hidden_states for access to all layer outputs
        self.model.config.output_hidden_states = True
        
        # Add SpecAugment for audio augmentation during training
        if use_specaug:
            self.spec_augment = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )
        
        # Adapter layer
        if use_adapter:
            self.adapter = AdapterLayer(self.model.config.hidden_size, adapter_dim, dropout)
        
        # Projection to desired output dimension if different from model dimension
        if self.model.config.hidden_size != output_dim:
            self.projector = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projector = nn.Identity()

    def forward(self, input_values, training=False):
        """
        Forward pass through the audio encoder.
        Args:
            input_values: Audio features (batch_size, seq_len, feat_dim)
            training: Whether in training mode (for SpecAugment)
        """
        # Apply SpecAugment during training if enabled
        if training and self.use_specaug:
            # Reshape if needed for SpecAugment
            batch_size, seq_len, feat_dim = input_values.shape
            reshaped = input_values.view(-1, feat_dim, seq_len)
            augmented = self.spec_augment(reshaped)
            input_values = augmented.view(batch_size, seq_len, feat_dim)
        
        outputs = self.model(input_values=input_values)
        
        # Get the hidden state
        hidden_state = outputs.last_hidden_state
        
        # Apply adapter if enabled
        if self.use_adapter:
            hidden_state = self.adapter(hidden_state)
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            pooled_output = hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            pooled_output = torch.mean(hidden_state, dim=1)
        elif self.pooling_strategy == "max":
            pooled_output = torch.max(hidden_state, dim=1)[0]
        else:
            pooled_output = hidden_state[:, 0]  # Default to CLS
        
        # Project to desired output dimension
        return self.projector(pooled_output)


class ImageEncoder(nn.Module):
    """
    Enhanced image encoder based on ViT with additional augmentation
    and residual adapter layers.
    """
    def __init__(self, 
                 model_name="google/vit-base-patch16-224", 
                 freeze_base=True, 
                 output_dim=768,
                 pooling_strategy="cls",
                 use_adapter=True,
                 adapter_dim=128,
                 dropout=0.1):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
        self.output_dim = output_dim
        self.pooling_strategy = pooling_strategy
        self.use_adapter = use_adapter
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Enable output_hidden_states for access to all layer outputs
        self.model.config.output_hidden_states = True
        
        # Adapter layer
        if use_adapter:
            self.adapter = AdapterLayer(self.model.config.hidden_size, adapter_dim, dropout)
        
        # Projection to desired output dimension if different from model dimension
        if self.model.config.hidden_size != output_dim:
            self.projector = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projector = nn.Identity()

    def forward(self, pixel_values):
        """
        Forward pass through the image encoder.
        """
        outputs = self.model(pixel_values=pixel_values)
        
        # Get the hidden state
        hidden_state = outputs.last_hidden_state
        
        # Apply adapter if enabled
        if self.use_adapter:
            hidden_state = self.adapter(hidden_state)
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            pooled_output = hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            # Skip the cls token in mean pooling
            pooled_output = torch.mean(hidden_state[:, 1:], dim=1)
        elif self.pooling_strategy == "max":
            # Skip the cls token in max pooling
            pooled_output = torch.max(hidden_state[:, 1:], dim=1)[0]
        else:
            pooled_output = hidden_state[:, 0]  # Default to CLS
        
        # Project to desired output dimension
        return self.projector(pooled_output)


class AdapterLayer(nn.Module):
    """
    Adapter layer that can be inserted into transformer models.
    Uses a bottleneck architecture to reduce parameters.
    """
    def __init__(self, input_dim, adapter_dim, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_project(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return residual + hidden_states


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model that combines text, audio, and image encoders
    with various fusion strategies and alignment techniques.
    """
    def __init__(self,
                text_encoder_params=None,
                audio_encoder_params=None,
                image_encoder_params=None,
                embedding_dim=768,
                fusion_method="attention",
                use_contrastive=True,
                contrastive_temperature=0.07,
                cross_modal_layers=2,
                modality_dropout=0.1):
        super().__init__()
        
        # Default encoder parameters if not provided
        if text_encoder_params is None:
            text_encoder_params = {"model_name": "roberta-base", "freeze_base": True}
        if audio_encoder_params is None:
            audio_encoder_params = {"model_name": "MIT/ast-finetuned-audioset-10-10-0.4593", "freeze_base": True}
        if image_encoder_params is None:
            image_encoder_params = {"model_name": "google/vit-base-patch16-224", "freeze_base": True}
        
        # Create encoders with common embedding dimension
        self.text_encoder = TextEncoder(**text_encoder_params, output_dim=embedding_dim)
        self.audio_encoder = AudioEncoder(**audio_encoder_params, output_dim=embedding_dim)
        self.image_encoder = ImageEncoder(**image_encoder_params, output_dim=embedding_dim)
        
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
        self.use_contrastive = use_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.modality_dropout = modality_dropout
        
        # Modality-specific dropout for regularization and robustness
        self.modality_dropout_layer = nn.Dropout(modality_dropout)
        
        # Create fusion module based on the selected method
        if fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, embedding_dim)
            )
        elif fusion_method == "attention":
            # Cross-modal attention fusion
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=8, 
                dim_feedforward=embedding_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            )
            self.fusion_layer = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=cross_modal_layers
            )
            
            # Positional encodings for different modalities
            self.modality_embeddings = nn.Parameter(torch.randn(3, embedding_dim))
        elif fusion_method == "gated":
            # Gated multimodal fusion
            self.modal_gates = nn.Sequential(
                nn.Linear(embedding_dim * 3, 3),
                nn.Softmax(dim=1)
            )
        else:
            # Default to simple averaging
            self.fusion_layer = None
    
    def forward(self, text_inputs=None, audio_inputs=None, image_inputs=None, training=False):
        """
        Forward pass through the multimodal model.
        Each input modality is optional to support flexible combinations.
        """
        # Initialize embeddings for all modalities
        text_embedding = audio_embedding = image_embedding = None
        
        # Get embeddings for provided modalities
        if text_inputs is not None:
            text_embedding = self.text_encoder(**text_inputs)
        
        if audio_inputs is not None:
            audio_embedding = self.audio_encoder(audio_inputs, training=training)
        
        if image_inputs is not None:
            image_embedding = self.image_encoder(image_inputs)
        
        # Count available modalities
        available_embeddings = [emb for emb in [text_embedding, audio_embedding, image_embedding] if emb is not None]
        num_modalities = len(available_embeddings)
        
        if num_modalities == 0:
            raise ValueError("At least one modality input must be provided")
        
        # For single modality, return that embedding directly
        if num_modalities == 1:
            return {
                "fusion_embedding": available_embeddings[0],
                "text_embedding": text_embedding,
                "audio_embedding": audio_embedding,
                "image_embedding": image_embedding,
                "contrastive_loss": None
            }
        
        # Create masked versions for robust multimodal fusion
        if training and self.modality_dropout > 0:
            available_embeddings = [self.modality_dropout_layer(emb) for emb in available_embeddings]
        
        # Multimodal fusion based on the selected method
        if self.fusion_method == "concat":
            # For concatenation, fill missing modalities with zeros
            modal_tensors = []
            if text_embedding is not None:
                modal_tensors.append(text_embedding)
            else:
                modal_tensors.append(torch.zeros_like(available_embeddings[0]))
                
            if audio_embedding is not None:
                modal_tensors.append(audio_embedding)
            else:
                modal_tensors.append(torch.zeros_like(available_embeddings[0]))
                
            if image_embedding is not None:
                modal_tensors.append(image_embedding)
            else:
                modal_tensors.append(torch.zeros_like(available_embeddings[0]))
            
            concat_embedding = torch.cat(modal_tensors, dim=1)
            fusion_embedding = self.fusion_layer(concat_embedding)
            
        elif self.fusion_method == "attention":
            # For attention, only use available modalities
            modal_tensors = []
            modality_indices = []
            
            if text_embedding is not None:
                modal_tensors.append(text_embedding)
                modality_indices.append(0)
                
            if audio_embedding is not None:
                modal_tensors.append(audio_embedding)
                modality_indices.append(1)
                
            if image_embedding is not None:
                modal_tensors.append(image_embedding)
                modality_indices.append(2)
            
            # Add modality embeddings
            for i, (tensor, mod_idx) in enumerate(zip(modal_tensors, modality_indices)):
                modal_tensors[i] = tensor + self.modality_embeddings[mod_idx]
            
            # Stack modalities along sequence dimension for transformer
            stacked_embeddings = torch.stack(modal_tensors, dim=1)  # [batch_size, num_modalities, embedding_dim]
            
            # Pass through transformer encoder for cross-modal attention
            fusion_output = self.fusion_layer(stacked_embeddings)  # [batch_size, num_modalities, embedding_dim]
            
            # Pool across modalities (mean pooling)
            fusion_embedding = torch.mean(fusion_output, dim=1)  # [batch_size, embedding_dim]
            
        elif self.fusion_method == "gated":
            # For gated fusion, fill missing modalities with zeros
            modal_tensors = []
            if text_embedding is not None:
                modal_tensors.append(text_embedding)
            else:
                modal_tensors.append(torch.zeros_like(available_embeddings[0]))
                
            if audio_embedding is not None:
                modal_tensors.append(audio_embedding)
            else:
                modal_tensors.append(torch.zeros_like(available_embeddings[0]))
                
            if image_embedding is not None:
                modal_tensors.append(image_embedding)
            else:
                modal_tensors.append(torch.zeros_like(available_embeddings[0]))
            
            # Concatenate for gate computation
            concat_embedding = torch.cat(modal_tensors, dim=1)
            
            # Compute gates
            gates = self.modal_gates(concat_embedding)  # [batch_size, 3]
            
            # Apply gates to modalities
            fusion_embedding = (
                gates[:, 0:1] * modal_tensors[0] +
                gates[:, 1:2] * modal_tensors[1] +
                gates[:, 2:3] * modal_tensors[2]
            )
        else:
            # Simple average fusion
            fusion_embedding = torch.stack(available_embeddings).mean(dim=0)
        
        # Compute contrastive loss if enabled and in training mode
        contrastive_loss = None
        if training and self.use_contrastive and num_modalities > 1:
            contrastive_loss = self._compute_contrastive_loss(available_embeddings)
        
        return {
            "fusion_embedding": fusion_embedding,
            "text_embedding": text_embedding,
            "audio_embedding": audio_embedding,
            "image_embedding": image_embedding,
            "contrastive_loss": contrastive_loss
        }
    
    def _compute_contrastive_loss(self, embeddings):
        """
        Compute contrastive loss between modalities to improve alignment.
        """
        batch_size = embeddings[0].shape[0]
        num_modalities = len(embeddings)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = [F.normalize(emb, p=2, dim=1) for emb in embeddings]
        
        # Compute pairwise contrastive losses between modalities
        total_loss = 0
        for i in range(num_modalities):
            for j in range(i+1, num_modalities):
                # Compute similarity matrix
                sim_matrix = torch.matmul(
                    normalized_embeddings[i], normalized_embeddings[j].transpose(0, 1)
                ) / self.contrastive_temperature
                
                # Targets are the diagonal elements (positive pairs)
                targets = torch.arange(batch_size, device=sim_matrix.device)
                
                # Compute loss in both directions (i→j and j→i)
                loss_i_to_j = F.cross_entropy(sim_matrix, targets)
                loss_j_to_i = F.cross_entropy(sim_matrix.transpose(0, 1), targets)
                
                total_loss += (loss_i_to_j + loss_j_to_i) / 2
        
        # Average over all modality pairs
        num_pairs = (num_modalities * (num_modalities - 1)) // 2
        return total_loss / num_pairs


# Example of creating LoRA-enabled multimodal model
def create_lora_multimodal_model(
    text_model="roberta-base",
    audio_model="MIT/ast-finetuned-audioset-10-10-0.4593",
    image_model="google/vit-base-patch16-224",
    embedding_dim=768,
    fusion_method="attention",
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1):
    
    # Create the base multimodal model
    model = MultimodalFusionModel(
        text_encoder_params={
            "model_name": text_model,
            "freeze_base": True,
            "output_dim": embedding_dim,
            "pooling_strategy": "mean",
            "use_adapter": True,
            "adapter_dim": 128
        },
        audio_encoder_params={
            "model_name": audio_model,
            "freeze_base": True,
            "output_dim": embedding_dim,
            "use_adapter": True,
            "adapter_dim": 128
        },
        image_encoder_params={
            "model_name": image_model,
            "freeze_base": True,
            "output_dim": embedding_dim,
            "use_adapter": True,
            "adapter_dim": 128
        },
        embedding_dim=embedding_dim,
        fusion_method=fusion_method,
        use_contrastive=True
    )
    
    # Create PEFT config for the fusion model
    peft_config = {
        'rank': lora_rank,
        'alpha': lora_alpha,
        'dropout': lora_dropout,
        'target_modules': [],
        'layer_patterns': {
            # Target only specific parts of the model
            'layers': [-1, -2],  # Last layers of fusion
            'module_types': ['query', 'value']  # Only query and value projections
        }
    }
    
    # Import the PEFTManager
    from peft_modules import PEFTManager
    
    # Apply LoRA to the model
    peft_model = PEFTManager(model, peft_type="lora", peft_config=peft_config)
    
    return peft_model