"""
Utilities for cross-modal retrieval, including preprocessing and parameter counting.
"""

import torch
import numpy as np
import pandas as pd
import librosa
from tabulate import tabulate
from PIL import Image


def process_audio(waveform, sample_rate, target_length=10, processor=None):
    """
    Process audio waveform for input to the model.
    
    Args:
        waveform: Audio waveform numpy array
        sample_rate: Audio sample rate
        target_length: Target audio length in seconds
        processor: Audio feature processor
        
    Returns:
        Processed audio features
    """
    print(f"Processing audio - Input shape: {waveform.shape}, sample_rate: {sample_rate}")

    # Resample if needed
    if sample_rate != 16000:
        print(f" Resampling from {sample_rate}Hz to 16000Hz")
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Ensure audio is of target length
    target_samples = int(target_length * sample_rate)
    print(f" Target samples: {target_samples}, Current samples: {len(waveform)}")
    
    if len(waveform) > target_samples:
        # Randomly crop to target length
        start = np.random.randint(0, len(waveform) - target_samples)
        waveform = waveform[start:start + target_samples]
        print(f" Cropped to {waveform.shape}")
    else:
        # Pad with zeros if shorter
        padding = target_samples - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
        print(f" Padded to {waveform.shape}")

    # Convert to tensor
    waveform_tensor = torch.from_numpy(waveform).float()

    # Apply feature extraction if processor is provided
    if processor is not None:
        try:
            print(f"  Using processor: {type(processor).__name__}")
            
            # AST processor expects numpy array or list, not tensor
            # Convert back to numpy for processing
            waveform_np = waveform_tensor.numpy() if isinstance(waveform_tensor, torch.Tensor) else waveform_tensor
            
            features = processor(waveform_np, sampling_rate=sample_rate, return_tensors="pt")
            print(f"  Processor output: {features.keys()}")
            
            if "input_values" in features:
                print(f"  Output shape: {features['input_values'].shape}")
                # AST should output [1, 1024, 128] for spectrograms
                return features
            else:
                raise ValueError("Processor did not return 'input_values'")
                
        except Exception as e:
            print(f"  ⚠️ PROCESSOR ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # IMPORTANT: Don't fall back to raw waveform!
            # Return zero spectrogram features instead
            print(f"  Returning zero spectrogram features")
            return {"input_values": torch.zeros(1, 1024, 128)}
    
    # If no processor, we cannot return raw waveform for AST model
    print(f"  ERROR: No processor provided for audio processing")
    print(f"  Returning zero spectrogram features")
    return {"input_values": torch.zeros(1, 1024, 128)}
    
    # # Apply feature extraction if processor is provided
    # if processor is not None:
    #     try:
    #         print(f"  Using processor: {type(processor).__name__}")
    #         features = processor(waveform_tensor, sampling_rate=sample_rate, return_tensors="pt")
    #         print(f"  Processor output: {features.keys()}")
    #         print(f"  Output shape: {features['input_values'].shape}")
    #         return features
    #     except Exception as e:
    #         print(f"  ⚠️ PROCESSOR ERROR: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         # Here's where the problem might be - falling back to raw waveform
    #         print(f"  Falling back to raw waveform: {waveform_tensor.shape}")
    #         return {"input_values": waveform_tensor.unsqueeze(0)}
    
    # # Return raw waveform if no processor
    # print(f"  No processor provided, returning raw waveform")
    # return {"input_values": waveform_tensor.unsqueeze(0)}


def process_image(image, processor=None, image_size=224):
    """
    Process image for input to the model.
    
    Args:
        image: PIL Image
        processor: Image feature processor
        image_size: Image size for resizing
        
    Returns:
        Processed image features
    """
    
    # Apply feature extraction if processor is provided
    if processor is not None:    
        features = processor(images=image, return_tensors="pt")
        return features
    
    # Basic preprocessing if no processor
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size))
    
    # Convert to tensor and normalize
    img_array = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    
    # Apply normalization (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return {"pixel_values": img_tensor.unsqueeze(0)}


def prepare_batch_for_model(batch, device):
    """
    Prepare a batch of data for input to the model.
    
    Args:
        batch: Dictionary of tensors from the dataloader
        device: Device to move tensors to
        
    Returns:
        Dictionary of tensors ready for model input
    """
    model_inputs = {}
    
    # Copy relevant tensors to device
    if "input_ids" in batch:
        model_inputs["input_ids"] = batch["input_ids"].to(device)
        if "attention_mask" in batch:
            model_inputs["attention_mask"] = batch["attention_mask"].to(device)
    
    if "audio_values" in batch:
        input_values = batch["audio_values"].to(device)
        print("Audio values shape before model:", input_values.shape)
        
        # Fix 1: Try reshaping if needed (if it's 2D and needs to be 4D)
        if len(input_values.shape) == 3:

            model_inputs["input_values"] = input_values
        elif len(input_values.shape) == 2 and input_values[-1] in [48000, 8000, 160000]:

            raise ValueError(
                f"Received raw waveform data of shape {input_values.shape} instead of processed spectrogram. "
                f"Audio processing may have failed in the dataset."
            )
        else:
            print(f"WARNING: Unexpected audio shape: {input_values.shape}")
            model_inputs["input_values"] = input_values

        #     print("Reshaped audio values:", input_values.shape)
            
        # model_inputs["input_values"] = input_values

    
    if "pixel_values" in batch:
        model_inputs["pixel_values"] = batch["pixel_values"].to(device)
    
    return model_inputs


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