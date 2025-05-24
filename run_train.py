"""
Main training script for cross-modal retrieval with PEFT.
"""

import os
import torch
import argparse
import yaml
from omegaconf import OmegaConf
import logging
import sys

from src.models.encoders import TextEncoder, AudioEncoder, ImageEncoder
from src.models.peft_modules import PEFTManager
from src.models.cman import CMAN
from src.data import get_dataloader
from src.trainer import Trainer
from src.utils import print_model_summary


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = OmegaConf.load(f)
    return config


def build_model(config):
    """Build model from configuration."""
    # Create text encoder
    text_encoder = TextEncoder(
        model_name=config.model.text.model_name,
        freeze_base=config.model.text.freeze_base,
        output_dim=config.model.text.output_dim
    )
    
    # Create audio encoder
    audio_encoder = AudioEncoder(
        model_name=config.model.audio.model_name,
        freeze_base=config.model.audio.freeze_base,
        output_dim=config.model.audio.output_dim
    )
    
    # Create image encoder
    image_encoder = ImageEncoder(
        model_name=config.model.image.model_name,
        freeze_base=config.model.image.freeze_base,
        output_dim=config.model.image.output_dim
    )
    
    # Apply PEFT to encoders if specified
    if config.peft.enabled:
        text_peft = PEFTManager(
            text_encoder,
            peft_type=config.peft.method,
            peft_config=config.peft.text
        )
        
        audio_peft = PEFTManager(
            audio_encoder,
            peft_type=config.peft.method,
            peft_config=config.peft.audio
        )
        
        image_peft = PEFTManager(
            image_encoder,
            peft_type=config.peft.method,
            peft_config=config.peft.image
        )
        
        # Replace original encoders with PEFT versions
        text_encoder = text_peft
        audio_encoder = audio_peft
        image_encoder = image_peft
    
    # Create cross-modal model
    model = CMAN(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        image_encoder=image_encoder,
        num_layers=3
        # embedding_dim=config.model.embedding_dim,
        # dropout=config.model.dropout
    )
    
    return model


def build_optimizer(model, config):
    """Build optimizer from configuration."""
    # If using PEFT, only optimize trainable parameters
    if config.peft.enabled:
        trainable_params = []
        for module in model.children():
            if hasattr(module, 'get_trainable_parameters'):
                trainable_params.extend(module.get_trainable_parameters())
            else:
                trainable_params.extend([p for p in module.parameters() if p.requires_grad])
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        # Otherwise, optimize all parameters that require gradients
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    
    return optimizer


def print_model_structure(model, indent=0, max_depth=None):
    """
    Print the model structure in a human-readable hierarchical format.
    
    Args:
        model: PyTorch model
        indent: Starting indentation level
        max_depth: Maximum depth to print (None for unlimited)
    """
    if max_depth is not None and indent > max_depth:
        return
        
    # Get model class name
    model_type = model.__class__.__name__
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print with proper indentation
    prefix = '  ' * indent
    print(f"{prefix}└─ {model_type} ({num_params:,} params, {trainable_params:,} trainable)")
    
    # Print children modules
    for name, child in model.named_children():
        print(f"{prefix}   └─ {name}:")
        print_model_structure(child, indent + 2, max_depth)


def main(args):
    """Main training function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.peft_method:
        config.peft.method = args.peft_method
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config.training.output_dir, "config.yaml"), 'w') as f:
        OmegaConf.save(config=config, f=f)
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model.to(device)
    
    print_model_structure(model, max_depth=5) 

    # Add this to verify trainable parameters
    logger.info("Checking trainable parameters:")
    lora_params_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable: {name}")
            lora_params_count += param.numel()
    logger.info(f"Total trainable parameters: {lora_params_count}")
    
    # Print model summary
    print_model_summary(model, title="Cross-Modal Retrieval Model Summary")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = get_dataloader(
        data_root=config.data.root,
        split="train",
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_text_length=config.data.max_text_length,
        audio_length=config.data.audio_length,
        audio_sample_rate=config.data.audio_sample_rate,
        image_size=config.data.image_size,
        subset_size=config.data.subset_size if hasattr(config.data, "subset_size") else None,
        dataset_type="flickr8k"
    )
    
    val_dataloader = get_dataloader(
        data_root=config.data.root,
        split="val",
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_text_length=config.data.max_text_length,
        audio_length=config.data.audio_length,
        audio_sample_rate=config.data.audio_sample_rate,
        #image_size=config.data.image_size,
        subset_size=config.data.subset_size if hasattr(config.data, "subset_size") else None,
        dataset_type="flickr8k"
    )
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs * len(train_dataloader)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.training.num_epochs,
        device=device,
        output_dir=config.training.output_dir,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
        eval_interval=config.training.eval_interval,
        mixed_precision=config.training.mixed_precision if hasattr(config.training, 'mixed_precision') else False
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cross-modal retrieval model with PEFT")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--peft_method", type=str, choices=["lora", "prefix_tuning", "adapter"], help="PEFT method")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for resuming training")
    
    args = parser.parse_args()
    main(args)