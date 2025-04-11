"""
Evaluation metrics for cross-modal retrieval.
"""

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


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