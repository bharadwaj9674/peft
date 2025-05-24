"""
Evaluation utilities for cross-modal retrieval models.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm

from .utils import prepare_batch_for_model


def compute_similarity_matrix(query_embeddings, gallery_embeddings):
    """
    Compute cosine similarity matrix between query and gallery embeddings.
    
    Args:
        query_embeddings: Query embeddings (num_queries, embedding_dim)
        gallery_embeddings: Gallery embeddings (num_gallery, embedding_dim)
        
    Returns:
        Similarity matrix of shape (num_queries, num_gallery)
    """
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity_matrix = torch.matmul(query_embeddings, gallery_embeddings.T)
    
    return similarity_matrix


def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute Recall@K metrics.
    
    Args:
        similarity_matrix: Similarity matrix of shape (num_queries, num_gallery)
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary of Recall@K values
    """
    num_queries = similarity_matrix.shape[0]
    
    # Get the indices of the sorted similarities (descending order)
    # For each query, we get the indices of the most similar gallery items
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Compute position of correct match (diagonal)
    # We assume that for each query i, the gallery item i is the correct match
    correct_indices = torch.arange(num_queries, device=similarity_matrix.device)
    positions = torch.zeros(num_queries, dtype=torch.long)
    
    for i in range(num_queries):
        # Find the position of the correct match in the sorted list
        positions[i] = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item()
    
    # Compute Recall@K
    recall_at_k = {}
    for k in k_values:
        recall = (positions < k).float().mean().item()
        recall_at_k[f"R@{k}"] = recall * 100  # Convert to percentage
    
    return recall_at_k


def compute_mean_average_precision(similarity_matrix):
    """
    Compute Mean Average Precision (mAP) for retrieval.
    
    Args:
        similarity_matrix: Similarity matrix of shape (num_queries, num_gallery)
        
    Returns:
        mAP value
    """
    num_queries = similarity_matrix.shape[0]
    
    # Create ground truth matrix
    # We assume that for each query i, the gallery item i is the correct match
    ground_truth = torch.eye(num_queries, device=similarity_matrix.device)
    
    # Compute AP for each query
    ap_values = []
    
    for i in range(num_queries):
        # Get similarities for this query
        similarities = similarity_matrix[i].cpu().numpy()
        
        # Get ground truth for this query
        gt = ground_truth[i].cpu().numpy()
        
        # Sort similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_gt = gt[sorted_indices]
        
        # Compute Average Precision
        ap = average_precision_score(sorted_gt, similarities[sorted_indices])
        ap_values.append(ap)
    
    # Compute Mean Average Precision
    map_value = np.mean(ap_values) * 100  # Convert to percentage
    
    return map_value


def compute_retrieval_metrics(query_embeddings, gallery_embeddings, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics for cross-modal retrieval.
    
    Args:
        query_embeddings: Query embeddings (num_queries, embedding_dim)
        gallery_embeddings: Gallery embeddings (num_gallery, embedding_dim)
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary of retrieval metrics
    """
    # Ensure embeddings are on CPU for metrics computation
    query_embeddings = query_embeddings.cpu()
    gallery_embeddings = gallery_embeddings.cpu()
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(query_embeddings, gallery_embeddings)
    
    # Compute Recall@K
    recall_metrics = compute_recall_at_k(similarity_matrix, k_values=k_values)
    
    # Compute Mean Average Precision
    map_value = compute_mean_average_precision(similarity_matrix)
    
    # Combine metrics
    metrics = recall_metrics
    metrics["mAP"] = map_value
    
    return metrics


def compute_multimodal_retrieval_metrics(embeddings_dict, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics for all pairs of modalities.
    
    Args:
        embeddings_dict: Dictionary of embeddings for each modality
                         {modality_name: embeddings}
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary of retrieval metrics for each pair of modalities
    """
    # Extract modalities and embeddings
    modalities = list(embeddings_dict.keys())
    
    # Initialize metrics dictionary
    all_metrics = {}
    
    # Compute metrics for each pair of modalities
    for i in range(len(modalities)):
        for j in range(len(modalities)):
            if i != j:  # Skip same modality
                mod_from = modalities[i]
                mod_to = modalities[j]
                
                # Get embeddings
                query_embeddings = embeddings_dict[mod_from]
                gallery_embeddings = embeddings_dict[mod_to]
                
                # Compute metrics
                metrics = compute_retrieval_metrics(
                    query_embeddings, 
                    gallery_embeddings, 
                    k_values=k_values
                )
                
                # Add to metrics dictionary
                for metric_name, value in metrics.items():
                    all_metrics[f"{mod_from}2{mod_to}_{metric_name}"] = value
    
    return all_metrics


class Evaluator:
    """Evaluator for cross-modal retrieval models."""

    def __init__(self, model, dataloader, device="cuda", output_dir=None):
        """
        Initialize the evaluator.
        
        Args:
            model: CMAN model to evaluate
            dataloader: Dataloader for evaluation data
            device: Device to use
            output_dir: Directory to save results
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
    
    def evaluate(self, k_values=[1, 5, 10]):
        """
        Evaluate the model on the dataloader.
        
        Args:
            k_values: List of K values for Recall@K
            
        Returns:
            Dictionary of metrics, Dictionary of embeddings
        """
        self.model.eval()
        
        # Collect embeddings
        all_embeddings = {
            "text": [],
            "audio": [],
            "image": []
        }
        
        all_ids = []
        
        # Process batches
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):

                # Debug prints
                print("Batch keys:", batch.keys())
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key} shape: {value.shape}")

                # Get batch IDs
                batch_ids = batch.pop("id") if "id" in batch else None
                
                # Prepare batch for model
                model_inputs = prepare_batch_for_model(batch, self.device)

                # Debug model inputs
                print("Model inputs keys:", model_inputs.keys())
                for key, value in model_inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key} shape: {value.shape}")
                
                # Forward pass
                outputs = self.model(**model_inputs)
                
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
        metrics = compute_multimodal_retrieval_metrics(embeddings_dict, k_values=k_values)
        
        # Save results if output directory is specified
        if self.output_dir:
            # Save metrics as JSON
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Create visualization of metrics
            self._create_metrics_visualization(metrics)
            
            # Save embeddings for later analysis
            embeddings_path = os.path.join(self.output_dir, "embeddings.pt")
            torch.save({
                "embeddings": embeddings_dict,
                "ids": all_ids
            }, embeddings_path)
        
        return metrics, embeddings_dict
    
    def _create_metrics_visualization(self, metrics):
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
            plt.savefig(os.path.join(self.output_dir, f"{metric_type.replace('@', '_at_')}.png"))
            plt.close()