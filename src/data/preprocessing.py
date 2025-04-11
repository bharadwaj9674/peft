"""
Preprocessing utilities for audio, image, and text data.
"""

import torch
import numpy as np
import librosa


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
    # Resample if needed
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Ensure audio is of target length
    target_samples = int(target_length * sample_rate)
    
    if len(waveform) > target_samples:
        # Randomly crop to target length
        start = np.random.randint(0, len(waveform) - target_samples)
        waveform = waveform[start:start + target_samples]
    else:
        # Pad with zeros if shorter
        padding = target_samples - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
    
    # Convert to tensor
    waveform_tensor = torch.from_numpy(waveform).float()
    
    # Apply feature extraction if processor is provided
    if processor is not None:
        features = processor(waveform_tensor, sampling_rate=sample_rate, return_tensors="pt")
        return features


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


def prepare_batch_for_model(batch, device):
    """
    Prepare a batch of data for input to the model.
    
    Args:
        batch: Dictionary of tensors from the dataloader
        device: Device to move tensors to
        
    Returns:
        Dictionary of tensors ready for model input
    """
    model_inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "input_values": batch["audio_values"].to(device),
        "pixel_values": batch["pixel_values"].to(device)
    }
    
    return model_inputs