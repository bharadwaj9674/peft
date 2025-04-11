"""
Utilities for counting and analyzing model parameters.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tabulate import tabulate


def count_parameters(model, trainable_only=False):
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_parameter_groups(model):
    """
    Group model parameters by module.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of parameter groups
    """
    parameter_groups = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = list(module.parameters())
            if len(params) > 0:
                num_params = sum(p.numel() for p in params)
                trainable_params = sum(p.numel() for p in params if p.requires_grad)
                
                parameter_groups[name] = {
                    "total_params": num_params,
                    "trainable_params": trainable_params,
                    "percentage_trainable": 100 * trainable_params / max(1, num_params)
                }
    
    return parameter_groups


def format_parameter_groups(parameter_groups, sort_by="total_params", ascending=False):
    """
    Format parameter groups into a readable table.
    
    Args:
        parameter_groups: Dictionary of parameter groups
        sort_by: Column to sort by
        ascending: Whether to sort in ascending order
        
    Returns:
        Formatted table as string
    """
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(parameter_groups, orient="index")
    
    # Add human-readable sizes
    df["total_params_readable"] = df["total_params"].apply(format_num_params)
    df["trainable_params_readable"] = df["trainable_params"].apply(format_num_params)
    
    # Sort
    df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Format table
    table = df[["total_params_readable", "trainable_params_readable", "percentage_trainable"]]
    table.columns = ["Total Params", "Trainable Params", "% Trainable"]
    
    # Convert to string
    formatted_table = tabulate(table, headers="keys", tablefmt="grid")
    
    return formatted_table


def format_num_params(num_params):
    """
    Format number of parameters to be human-readable.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"


def print_model_summary(model, title=None):
    """
    Print a summary of model parameters.
    
    Args:
        model: PyTorch model
        title: Optional title for the summary
        
    Returns:
        None (prints to console)
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    # Count parameters
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Print summary
    print(f"Total parameters: {format_num_params(total_params)}")
    print(f"Trainable parameters: {format_num_params(trainable_params)}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    # Get parameter groups
    parameter_groups = get_parameter_groups(model)
    
    # Print parameter groups
    print("\nParameter distribution by module:")
    print(format_parameter_groups(parameter_groups))


def compare_peft_methods(base_model, peft_models, peft_names):
    """
    Compare parameter counts between base model and PEFT variants.
    
    Args:
        base_model: Base model without PEFT
        peft_models: List of models with different PEFT methods
        peft_names: List of names for each PEFT method
        
    Returns:
        Comparison DataFrame
    """
    # Count parameters for base model
    base_total = count_parameters(base_model)
    base_trainable = count_parameters(base_model, trainable_only=True)
    
    # Initialize comparison data
    comparison = {
        "Method": ["Base Model"] + peft_names,
        "Total Parameters": [base_total],
        "Trainable Parameters": [base_trainable],
        "Parameter Efficiency": [100.0]  # Base model is 100% of itself
    }
    
    # Count parameters for each PEFT model
    for model in peft_models:
        total = count_parameters(model)
        trainable = count_parameters(model, trainable_only=True)
        efficiency = 100 * trainable / base_trainable
        
        comparison["Total Parameters"].append(total)
        comparison["Trainable Parameters"].append(trainable)
        comparison["Parameter Efficiency"].append(efficiency)
    
    # Create DataFrame
    df = pd.DataFrame(comparison)
    
    # Add human-readable sizes
    df["Total Parameters (readable)"] = df["Total Parameters"].apply(format_num_params)
    df["Trainable Parameters (readable)"] = df["Trainable Parameters"].apply(format_num_params)
    
    # Format efficiency as percentage
    df["Parameter Efficiency"] = df["Parameter Efficiency"].apply(lambda x: f"{x:.2f}%")
    
    return df