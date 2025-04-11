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
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoders import TextEncoder, AudioEncoder, ImageEncoder
from src.models.peft_modules import PEFTManager
from src.models.fusion import CrossModalModel
from src.data.datasets import get_dataloader
from src.data.preprocessing import prepare_batch_for_model
from src.evaluation.metrics import compute_retrieval_metrics, compute_multimodal_retrieval_metrics


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
    model = CrossModalModel(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        image_encoder=image_encoder,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def evaluate_model(model, dataloader, device, output_dir=None):
    """Evaluate model on the given dataloader."""
    model.eval()
    
    # Collect embeddings
    all_embeddings = {
        "text": [],
        "audio": [],
        "image": []
    }
    
    all_ids = []
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get batch IDs
            batch_ids = batch.pop("id") if "id" in batch else None
            
            # Prepare batch for model
            model_inputs = prepare_batch_for_model(batch, device)
            
            # Forward pass
            outputs = model(**model_inputs)
            
            # Collect embeddings
            if "text_embeddings" in outputs:
                all_embeddings["text"].append(outputs["text_embeddings"].cpu())
            if "audio_embeddings" in outputs:
                all_embeddings["audio"].append(outputs["audio_embeddings"].cpu())
            if "image_embeddings" in outputs:
                all_embeddings["image"].append(outputs["image_embeddings"].cpu())
            
            # Collect IDs
            if batch_ids is not None:
                all_ids.extend(batch_ids)
    
    # Concatenate embeddings
    embeddings_dict = {}
    for modality, embeddings_list in all_embeddings.items():
        if embeddings_list:
            embeddings_dict[modality] = torch.cat(embeddings_list, dim=0)
    
    # Compute metrics
    metrics = compute_multimodal_retrieval_metrics(embeddings_dict)
    
    # Print metrics
    print("\nRetrieval Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")
    
    # Save metrics if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualization of metrics
        create_metrics_visualization(metrics, output_dir)
        
        # Save embeddings for later analysis
        embeddings_path = os.path.join(output_dir, "embeddings.pt")
        torch.save({
            "embeddings": embeddings_dict,
            "ids": all_ids
        }, embeddings_path)
    
    return metrics, embeddings_dict


def create_metrics_visualization(metrics, output_dir):
    """Create visualizations of metrics."""
    # Group metrics by type (R@K, mAP)
    grouped_metrics = {}
    
    for metric_name, value in metrics.items():
        # Parse metric name
        parts = metric_name.split('_')
        direction = parts[0]  # e.g., "text2audio"
        metric_type = parts[1]  # e.g., "R@1", "mAP"
        
        # Group by metric type
        if metric_type not in grouped_metrics:
            grouped_metrics[metric_type] = {}
        
        grouped_metrics[metric_type][direction] = value
    
    # Create bar chart for each metric type
    for metric_type, values in grouped_metrics.items():
        plt.figure(figsize=(10, 6))
        
        # Sort directions by value
        sorted_directions = sorted(values.keys(), key=lambda k: values[k], reverse=True)
        sorted_values = [values[d] for d in sorted_directions]
        
        # Create bar chart
        bars = plt.bar(sorted_directions, sorted_values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Add title and labels
        plt.title(f"Cross-Modal Retrieval: {metric_type}")
        plt.xlabel("Direction")
        plt.ylabel("Value (%)")
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{metric_type.replace('@', '_at_')}.png"))
        plt.close()
    
    # Create comparison of R@K across different K values
    r_at_k_metrics = {}
    
    for metric_name, value in metrics.items():
        if "R@" in metric_name:
            parts = metric_name.split('_')
            direction = parts[0]  # e.g., "text2audio"
            k_value = int(parts[1].replace("R@", ""))
            
            if direction not in r_at_k_metrics:
                r_at_k_metrics[direction] = {}
            
            r_at_k_metrics[direction][k_value] = value
    
    # Create plot for each direction
    for direction, values in r_at_k_metrics.items():
        plt.figure(figsize=(8, 6))
        
        # Sort by K value
        sorted_k = sorted(values.keys())
        sorted_values = [values[k] for k in sorted_k]
        
        # Create line plot
        plt.plot(sorted_k, sorted_values, marker='o', linestyle='-', linewidth=2)
        
        # Add value labels
        for k, value in zip(sorted_k, sorted_values):
            plt.text(k, value + 1, f'{value:.2f}', ha='center')
        
        # Add title and labels
        plt.title(f"Recall@K for {direction}")
        plt.xlabel("K")
        plt.ylabel("Recall@K (%)")
        plt.xticks(sorted_k)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{direction}_recall_at_k.png"))
        plt.close()


def compare_peft_methods(configs, checkpoints, output_dir):
    """Compare different PEFT methods."""
    # Load models and evaluate
    results = {}
    
    for method, (config_path, checkpoint_path) in zip(configs.keys(), zip(configs.values(), checkpoints.values())):
        # Load config
        config = load_config(config_path)
        
        # Load model
        model = load_model(checkpoint_path, config)
        
        # Create dataloader
        val_dataloader = get_dataloader(
            data_root=config.data.root,
            split="val",
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            max_text_length=config.data.max_text_length,
            audio_length=config.data.audio_length,
            audio_sample_rate=config.data.audio_sample_rate,
            image_size=config.data.image_size
        )
        
        # Evaluate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        metrics, _ = evaluate_model(model, val_dataloader, device)
        
        # Store results
        results[method] = metrics
    
    # Create comparison table and visualizations
    create_peft_comparison(results, output_dir)


def create_peft_comparison(results, output_dir):
    """Create comparison of PEFT methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    results_path = os.path.join(output_dir, "peft_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison table for key metrics
    key_metrics = ["text2audio_R@1", "audio2text_R@1", "text2image_R@1", "image2text_R@1", "text2audio_mAP", "text2image_mAP"]
    
    # Extract methods and metrics
    methods = list(results.keys())
    
    # Create a DataFrame for comparison
    comparison_data = []
    
    for method in methods:
        row = {"Method": method}
        for metric in key_metrics:
            if metric in results[method]:
                row[metric] = results[method][metric]
            else:
                row[metric] = None
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    df.to_csv(os.path.join(output_dir, "peft_comparison.csv"), index=False)
    
    # Create comparison plots for each key metric
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        
        # Get values for this metric
        values = [results[method].get(metric, 0) for method in methods]
        
        # Create bar chart
        bars = plt.bar(methods, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Add title and labels
        plt.title(f"Comparison of PEFT Methods: {metric}")
        plt.xlabel("PEFT Method")
        plt.ylabel("Value (%)")
        plt.ylim(0, max(values) * 1.1)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
        plt.close()


def main(args):
    """Main evaluation function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.compare:
        # Compare PEFT methods
        logger.info("Comparing PEFT methods...")
        
        # Parse configurations and checkpoints
        configs = {}
        checkpoints = {}
        
        for pair in args.compare.split(','):
            method, config_path, checkpoint_path = pair.split(':')
            configs[method] = config_path
            checkpoints[method] = checkpoint_path
        
        # Output directory
        output_dir = args.output_dir or "./outputs/comparison"
        
        # Compare methods
        compare_peft_methods(configs, checkpoints, output_dir)
        
        logger.info(f"Comparison results saved to {output_dir}")
    else:
        # Evaluate a single model
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
            image_size=config.data.image_size
        )
        
        # Output directory
        output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "evaluation")
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics, embeddings = evaluate_model(model, dataloader, device, output_dir)
        
        logger.info(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--output_dir", type=str, help="Output directory for evaluation results")
    parser.add_argument("--compare", type=str, help="Compare PEFT methods (format: method1:config1:ckpt1,method2:config2:ckpt2)")
    
    args = parser.parse_args()
    main(args)