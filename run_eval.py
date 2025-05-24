"""
Evaluation script for cross-modal retrieval models.
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
from src.evaluator import Evaluator


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = OmegaConf.load(f)
    return config


def load_model(checkpoint_path, config):
    """Load model from checkpoint."""
    # Create encoders
    text_encoder = TextEncoder(
        model_name=config.model.text.model_name,
        freeze_base=config.model.text.freeze_base,
        output_dim=config.model.text.output_dim
    )
    
    audio_encoder = AudioEncoder(
        model_name=config.model.audio.model_name,
        freeze_base=config.model.audio.freeze_base,
        output_dim=config.model.audio.output_dim
    )
    
    image_encoder = ImageEncoder(
        model_name=config.model.image.model_name,
        freeze_base=config.model.image.freeze_base,
        output_dim=config.model.image.output_dim
    )
    
    # Apply PEFT if specified
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
    
    # Create model
    model = CMAN(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        image_encoder=image_encoder,
        num_layers=3,
        # embedding_dim=config.model.embedding_dim,
        # dropout=config.model.dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def main(args):
    """Main evaluation function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config)
    model.to(device)
    
    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = get_dataloader(
        data_root=config.data.root,
        split=args.split,
        batch_size=args.batch_size or config.data.batch_size,
        num_workers=config.data.num_workers,
        max_text_length=config.data.max_text_length,
        audio_length=config.data.audio_length,
        audio_sample_rate=config.data.audio_sample_rate,
        image_size=config.data.image_size,
        dataset_type="flickr8k"
    )
    
    # Output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "evaluation")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics, embeddings = evaluator.evaluate()
    
    # Print metrics
    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.2f}")
    
    logger.info(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--output_dir", type=str, help="Output directory for evaluation results")
    
    args = parser.parse_args()
    main(args)