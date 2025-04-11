"""
Dataset classes for cross-modal retrieval.
"""

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import librosa
from transformers import RobertaTokenizer, AutoImageProcessor, AutoProcessor
from .preprocessing import process_audio, process_image


class CrossModalDataset(Dataset):
    """
    Dataset for cross-modal retrieval using Clotho/HowTo100M subset.
    
    Each item contains text, audio, and image data.
    """
    def __init__(
        self,
        data_root,
        split="train",
        max_text_length=77,
        audio_length=10,
        audio_sample_rate=16000,
        image_size=224,
        subset_size=None
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.max_text_length = max_text_length
        self.audio_length = audio_length
        self.audio_sample_rate = audio_sample_rate
        self.image_size = image_size
        
        # Load data annotations
        self.annotations = self._load_annotations(subset_size)
        
        # Initialize tokenizer and feature extractors
        self.tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
        self.audio_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch32-224-in21k", use_fast=True)
    
    def _load_annotations(self, subset_size=None):
        """
        Load dataset annotations from a JSON file.
        
        Returns:
            List of data instances, each with paths to text, audio, and image files
        """
        # Assume annotations are stored in a JSON file
        annotations_file = os.path.join(self.data_root, f"{self.split}_annotations.json")

        if os.path.exists(annotations_file):
            print("Loaded annotations file from: ", annotations_file)
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
        # Take a subset if specified
        if subset_size is not None:
            annotations = annotations[:min(subset_size, len(annotations))]
            
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary with tokenized text, processed audio, and image features
        """
        item = self.annotations[idx]
        
        # Process text
        text = item["text"]
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract text inputs
        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)
        
        # Process audio
        audio_path = os.path.join(self.data_root, "audio", item["audio_path"])
        
        try:
            # Load and process actual audio if file exists
            if os.path.exists(audio_path):
                print("Loading audio file from: ", audio_path)
                audio_waveform, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
                audio_features = process_audio(
                    audio_waveform, 
                    sr, 
                    target_length=self.audio_length, 
                    processor=self.audio_processor
                )
                
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
        
        # Process image
        image_path = os.path.join(self.data_root, "images", item["image_path"])

        try:
            # Load and process actual image if file exists
            if os.path.exists(image_path):
                print("Loading image from: ", image_path)
                image = Image.open(image_path).convert("RGB")
                image_features = process_image(image, self.image_processor)

        except Exception as e:
            print(f"Error processing image file {image_path}: {e}")
        
        # Prepare output dictionary
        output = {
            "id": item["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_values": audio_features["input_values"].squeeze(0),
            "pixel_values": image_features["pixel_values"].squeeze(0)
        }   
        
        return output


def get_dataloader(
    data_root,
    split="train",
    batch_size=32,
    num_workers=4,
    max_text_length=77,
    audio_length=10,
    audio_sample_rate=16000,
    image_size=224,
    subset_size=None
):
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split (train, val, test)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_text_length: Maximum text length
        audio_length: Audio length in seconds
        audio_sample_rate: Audio sample rate
        image_size: Image size for resizing
        subset_size: Limit the dataset to this number of samples
        
    Returns:
        DataLoader for the specified dataset
    """
    dataset = CrossModalDataset(
        data_root=data_root,
        split=split,
        max_text_length=max_text_length,
        audio_length=audio_length,
        audio_sample_rate=audio_sample_rate,
        image_size=image_size,
        subset_size=subset_size
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader