"""
Flickr8k Audio Dataset Adapter to use with your existing training code.

This module loads the prepared dataset and makes it compatible with your
existing training pipeline.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import librosa
import logging
from pathlib import Path
from transformers import RobertaTokenizer, AutoFeatureExtractor
from .preprocessing import process_audio, process_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Flickr8kDataset(Dataset):
    """
    Dataset for cross-modal retrieval using Flickr8k Audio.
    """
    def __init__(
        self,
        data_root,
        split="train",
        max_text_length=77,
        audio_length=5,
        audio_sample_rate=16000,
        image_size=224
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.max_text_length = max_text_length
        self.audio_length = audio_length
        self.audio_sample_rate = audio_sample_rate
        self.image_size = image_size
        
        # Load metadata
        metadata_file = self.data_root / "metadata" / f"{split}_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}. "
                "Please run prepare_my_data.py first."
            )
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize tokenizer and feature extractors
        logger.info("Initializing tokenizer and processors...")
        self.tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
        self.audio_processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch32-224-in21k")
        
        logger.info(f"Dataset loaded with {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary with tokenized text, processed audio, and image features
        """
        item = self.metadata[idx]
        
        # Process text
        text = item["caption"]
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
        audio_features = {"input_values": torch.zeros(1, self.audio_sample_rate * self.audio_length)}
        audio_path = self.data_root / "audio" / self.split / item["audio_filename"]
        
        try:
            if os.path.exists(audio_path):
                # Load audio file
                audio_waveform, sr = librosa.load(str(audio_path), sr=self.audio_sample_rate)
                
                
                # Process audio with AST feature extractor
                audio_features = process_audio(
                    audio_waveform, 
                    sr,
                    target_length=self.audio_length,
                    processor=self.audio_processor
                )
        except Exception as e:
            logger.warning(f"Error processing audio file {audio_path}: {e}")
        
        # Process image
        image_features = {"pixel_values": torch.zeros(1, 3, self.image_size, self.image_size)}
        image_path = self.data_root / "images" / self.split / item["image_filename"]
        
        try:
            if os.path.exists(image_path):
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                image_features = process_image(image, self.image_processor)
        except Exception as e:
            logger.warning(f"Error processing image file {image_path}: {e}")
        
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
    audio_length=5,
    audio_sample_rate=16000,
    image_size=224,
    subset_size=None
):
    """
    Get dataloader for the Flickr8k dataset for the specified split.
    
    This function is compatible with your existing training code.
    
    Args:
        data_root: Root directory of the prepared dataset
        split: Dataset split (train, val, test)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_text_length: Maximum text length
        audio_length: Audio length in seconds
        audio_sample_rate: Audio sample rate
        image_size: Image size for resizing
        subset_size: Limit the dataset to this number of samples
        
    Returns:
        DataLoader for the specified dataset split
    """
    dataset = Flickr8kDataset(
        data_root=data_root,
        split=split,
        max_text_length=max_text_length,
        audio_length=audio_length,
        audio_sample_rate=audio_sample_rate,
        image_size=image_size
    )
    
    # Take a subset if specified
    if subset_size is not None and subset_size < len(dataset):
        from torch.utils.data import Subset
        import random
        
        # Set seed for reproducibility
        random.seed(42)
        
        # Randomly select subset_size indices
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = Subset(dataset, indices)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader