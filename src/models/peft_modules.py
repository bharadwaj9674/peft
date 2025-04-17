import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple
from functools import reduce


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This module wraps an existing Linear module and adds low-rank adaptation.
    The original parameters are frozen, and only the LoRA parameters are trained.
    """
    def __init__(self, linear_module, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = linear_module  # Original linear module
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        
        # LoRA parameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA A and B matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze the original weights
        for param in self.linear.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Save original shape for later reshaping if needed
        orig_shape = x.shape
        
        # Handle different input shapes
        if len(orig_shape) > 2:
            # For inputs with more than 2 dimensions (e.g., attention layers)
            x_2d = x.reshape(-1, orig_shape[-1])
            
            # Original linear output
            original_output = self.linear(x)
            
            # LoRA path
            lora_x = self.lora_dropout(x_2d)
            lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T
            lora_output = lora_output.view(*orig_shape[:-1], -1)
        else:
            # Standard case for 2D tensors
            original_output = self.linear(x)
            lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        # Combine outputs
        return original_output + (lora_output * self.scaling)


class PEFTManager(nn.Module):
    """
    Manager class to apply various PEFT methods to a pre-trained model.
    
    This class scans a model for target modules and replaces them with 
    parameter-efficient versions that include LoRA, Prefix Tuning, or Adapters.
    """
    def __init__(self, base_model, peft_type="lora", peft_config=None):
        super().__init__()
        self.base_model = base_model
        self.peft_type = peft_type
        self.peft_modules = nn.ModuleDict()
        
        # Copy important attributes from the base model
        if hasattr(base_model, 'output_dim'):
            self.output_dim = base_model.output_dim
        
        # Apply PEFT to the base model
        if peft_type == "lora":
            self._apply_lora(peft_config)
        else:
            raise ValueError(f"Unknown PEFT type: {peft_type}")
    
    def _apply_lora(self, config):
        """Apply LoRA to specific linear layers in the base model."""
        if config is None:
            config = {
                'rank': 8,
                'alpha': 16,
                'dropout': 0.1,
                # More selective targeting - by default only apply to value projections
                # in the last few layers and output projection
                'target_modules': [],
                'layer_patterns': {
                    # Target only specific layer numbers (e.g., last few layers)
                    'layers': [-1, -2, -3, -4],  # Negative indices for last layers
                    # Target specific module types within those layers
                    'module_types': ['value', 'output']  # Only value and output projections
                }
            }
        
        # Get all module names first
        all_modules = dict(self.base_model.named_modules())
        
        # Find which modules to target based on the configuration
        target_modules = set()
        
        # Add explicit target modules if provided
        if 'target_modules' in config and config['target_modules']:
            for name in all_modules:
                if any(target in name for target in config['target_modules']):
                    target_modules.add(name)
        
        # Add modules based on layer patterns
        if 'layer_patterns' in config:
            patterns = config['layer_patterns']
            
            # Find all layer modules
            layer_modules = []
            for name in all_modules:
                if 'layer' in name.lower() or 'block' in name.lower():
                    try:
                        # Try to extract layer number
                        parts = name.split('.')
                        for part in parts:
                            if 'layer' in part.lower() or 'block' in part.lower():
                                # Extract number from layer/block name
                                num_part = ''.join(c for c in part if c.isdigit())
                                if num_part:
                                    layer_modules.append((name, int(num_part)))
                                    break
                    except:
                        continue
            
            # Sort layers by index
            layer_modules.sort(key=lambda x: x[1])
            
            # Select target layers based on indices
            target_layers = []
            for layer_idx in patterns.get('layers', []):
                if layer_idx < 0:
                    # Negative indexing (from the end)
                    idx = max(0, len(layer_modules) + layer_idx)
                    if idx < len(layer_modules):
                        target_layers.append(layer_modules[idx][0])
                elif layer_idx < len(layer_modules):
                    target_layers.append(layer_modules[layer_idx][0])
            
            # Find modules within target layers that match module types
            module_types = patterns.get('module_types', [])
            for layer_name in target_layers:
                for name in all_modules:
                    if name.startswith(layer_name) and any(module_type in name.lower() for module_type in module_types):
                        if isinstance(all_modules[name], nn.Linear):
                            target_modules.add(name)
        
        print(f"Applying LoRA to {len(target_modules)} modules: {target_modules}")
        
        # Replace target modules with LoRA versions
        for name in target_modules:
            module = all_modules[name]
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parent_name, child_name = self._get_parent_and_child_name(name)
                parent_module = self._get_module_by_name(self.base_model, parent_name)
                
                # Create wrapped LoRA module
                lora_module = LoRALinear(
                    module,
                    rank=config.get('rank', 8),
                    alpha=config.get('alpha', 16),
                    dropout=config.get('dropout', 0.1)
                )
                
                # Add to tracked PEFT modules
                self.peft_modules[name.replace(".", "_")] = lora_module
                
                # Replace the original module with the wrapped one
                setattr(parent_module, child_name, lora_module)
    
    def _get_parent_and_child_name(self, name):
        """Split a module name into parent and child parts."""
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            return '', parts[0]  # Module is at the top level
        return parts[0], parts[1]
    
    def _get_module_by_name(self, model, name):
        """Get a module by its name."""
        if name == '':
            return model
        return reduce(getattr, name.split('.'), model)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that delegates to the modified base model.
        The PEFT modules are integrated directly in the model's forward path.
        """
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (the PEFT parameters)."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_trainable_parameter_count(self):
        """Get the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameter_count(self):
        """Get the total parameter count of the model."""
        return sum(p.numel() for p in self.parameters())