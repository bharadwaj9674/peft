"""
Script to load Flickr8k dataset using only the first audio version and first caption.

This script simplifies the dataset loading by:
1. Always selecting the _0.wav audio files
2. Taking only the first caption for each image
3. Creating a one-to-one mapping between images, audio, and captions
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import librosa
import re
from pathlib import Path
from transformers import RobertaTokenizer, AutoProcessor, AutoImageProcessor


def process_audio(audio_waveform, sample_rate, target_length=5, processor=None):
    """Process audio waveform to fixed length and extract features."""
    # Calculate target number of samples
    target_samples = int(target_length * sample_rate)
    
    # Pad or truncate the audio to the target length
    if len(audio_waveform) > target_samples:
        # Truncate
        audio_waveform = audio_waveform[:target_samples]
    elif len(audio_waveform) < target_samples:
        # Pad with zeros
        padding = np.zeros(target_samples - len(audio_waveform))
        audio_waveform = np.concatenate([audio_waveform, padding])
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_waveform).float()
    
    # Extract features with processor
    if processor is not None:
        with torch.no_grad():
            features = processor(
                audio_tensor, 
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
        return features
    else:
        # Return raw waveform if no processor
        return {"input_values": audio_tensor.unsqueeze(0)}


def process_image(image, processor=None):
    """Process image and extract features."""
    if processor is not None:
        with torch.no_grad():
            features = processor(images=image, return_tensors="pt")
        return features
    else:
        # Basic preprocessing if no processor
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image).unsqueeze(0)
        return {"pixel_values": tensor}


class SimpleFlickr8kDataset(Dataset):
    """
    Simplified dataset for Flickr8k that always uses the first audio version (_0.wav)
    and the first caption for each image.
    """
    def __init__(
        self,
        data_root,
        split="train",
        max_text_length=77,
        audio_length=5,
        audio_sample_rate=16000,
        image_size=224,
        val_split_ratio=0.1,
        test_split_ratio=0.1,
        random_seed=42,
        verbose=True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.max_text_length = max_text_length
        self.audio_length = audio_length
        self.audio_sample_rate = audio_sample_rate
        self.image_size = image_size
        self.verbose = verbose
        
        if verbose:
            print(f"Initializing Flickr8k dataset from {self.data_root}")
            print(f"Always using first audio version (_0.wav) and first caption")
        
        # Initialize processors
        self.tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
        try:
            self.audio_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            if verbose:
                print("Loaded AST audio processor")
        except Exception as e:
            if verbose:
                print(f"Could not load audio processor: {e}. Using raw waveforms.")
            self.audio_processor = None
            
        try:
            self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch32-224-in21k")
            if verbose:
                print("Loaded ViT image processor")
        except Exception as e:
            if verbose:
                print(f"Could not load image processor: {e}. Using basic preprocessing.")
            self.image_processor = None
        
        # Build dataset items
        self.items = self._build_dataset(val_split_ratio, test_split_ratio, random_seed)
        if verbose:
            print(f"Loaded {len(self.items)} samples for {split} split")
    
    def _find_directory(self, possible_names, file_pattern="*"):
        """Find a directory with the given possible names or containing matching files."""
        # First try standard directory names
        for name in possible_names:
            path = self.data_root / name
            if path.exists() and path.is_dir():
                if self.verbose:
                    print(f"Found directory: {path}")
                return path
                
        # Then look for directories with matching files
        for path in self.data_root.iterdir():
            if path.is_dir() and list(path.glob(file_pattern)):
                if self.verbose:
                    print(f"Found directory with {file_pattern} files: {path}")
                return path
                
        # Fallback to data root
        if self.verbose:
            print(f"No directory found for {file_pattern}, using data root: {self.data_root}")
        return self.data_root
        
    def _find_audio_dir(self):
        """Find the audio directory."""
        return self._find_directory(["audio", "Audio", "AUDIO"], "*.wav")
        
    def _find_image_dir(self):
        """Find the image directory."""
        return self._find_directory(["Images", "images", "IMAGES"], "*.jpg")
    
    def _parse_captions(self):
        """Parse captions from various possible formats."""
        captions = {}
        
        # Try known caption files
        caption_files = [
            self.data_root / "captions.txt",
            self.data_root / "wav2capt.txt"
        ]
        
        # Also look for any text files
        caption_files.extend(self.data_root.glob("*.txt"))
        
        for path in caption_files:
            if not path.exists():
                continue
                
            if self.verbose:
                print(f"Trying to parse captions from {path}")
                
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Try to match format: image.jpg,caption
                        if '.jpg,' in line:
                            parts = line.split('.jpg,', 1)
                            if len(parts) == 2:
                                filename = parts[0] + '.jpg'
                                caption = parts[1].strip()
                                
                                # Extract ID from filename (digits at the beginning)
                                id_match = re.search(r'^(\d+)', filename)
                                if id_match:
                                    image_id = id_match.group(1)
                                    if image_id not in captions:
                                        captions[image_id] = caption
                                        
                        # Try to match wav_file image_file #tag format
                        elif ' ' in line and '#' in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                image_file = parts[1]
                                
                                # Extract ID from image filename
                                id_match = re.search(r'^(\d+)', image_file)
                                if id_match:
                                    image_id = id_match.group(1)
                                    if image_id not in captions:
                                        # Use parts after the image filename as caption
                                        caption = " ".join(parts[2:])
                                        captions[image_id] = caption
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing {path}: {e}")
        
        if self.verbose:
            print(f"Loaded {len(captions)} captions")
            
        return captions
    
    def _build_dataset(self, val_split_ratio, test_split_ratio, random_seed):
        """Build dataset with first audio version and first caption per image."""
        # Find directories
        audio_dir = self._find_audio_dir()
        image_dir = self._find_image_dir()
        
        # Load captions - only keeping the first one per ID
        captions = self._parse_captions()
        
        # Find all images and organize by ID
        image_files = {}
        for path in image_dir.glob("*.jpg"):
            # Extract ID from filename
            id_match = re.search(r'^(\d+)', path.name)
            if id_match:
                image_id = id_match.group(1)
                image_files[image_id] = path
                
        if self.verbose:
            print(f"Found {len(image_files)} images")
        
        # Find all "_0.wav" audio files (first version) and organize by ID
        audio_files = {}
        for path in audio_dir.glob("*_0.wav"):
            # Extract ID from filename
            id_match = re.search(r'^(\d+)', path.name)
            if id_match:
                audio_id = id_match.group(1)
                audio_files[audio_id] = path
                
        if self.verbose:
            print(f"Found {len(audio_files)} audio files with _0.wav suffix")
        
        # Find IDs that have image, audio, and caption
        image_ids = set(image_files.keys())
        audio_ids = set(audio_files.keys())
        caption_ids = set(captions.keys())
        
        common_ids = list(image_ids.intersection(audio_ids).intersection(caption_ids))
        
        if self.verbose:
            print(f"Found {len(common_ids)} IDs with all three: image, _0.wav audio, and caption")
            
        # If not enough samples, also consider IDs with just image and caption
        if len(common_ids) < 10:
            if self.verbose:
                print("Not enough samples with all three, including IDs with just image and caption")
            image_caption_ids = list(image_ids.intersection(caption_ids))
            common_ids = sorted(list(set(common_ids + image_caption_ids)))
            if self.verbose:
                print(f"Now have {len(common_ids)} IDs (including some without audio)")
        
        # Split into train/val/test
        np.random.seed(random_seed)
        np.random.shuffle(common_ids)
        
        total = len(common_ids)
        test_size = max(1, int(total * test_split_ratio))
        val_size = max(1, int(total * val_split_ratio))
        train_size = total - test_size - val_size
        
        if self.verbose:
            print(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
        
        if self.split == "train":
            split_ids = common_ids[:train_size]
        elif self.split == "val":
            split_ids = common_ids[train_size:train_size+val_size]
        else:  # test
            split_ids = common_ids[train_size+val_size:]
        
        # Create items for each ID in this split
        items = []
        for data_id in split_ids:
            # Get image
            image_path = image_files.get(data_id)
            if image_path is None:
                continue
                
            # Get audio (may be None for some IDs)
            audio_path = audio_files.get(data_id)
            
            # Get caption
            caption = captions.get(data_id)
            if caption is None:
                continue
                
            # Add to items
            items.append({
                "id": data_id,
                "text": caption,
                "audio_path": audio_path.relative_to(self.data_root) if audio_path else None,
                "image_path": image_path.relative_to(self.data_root)
            })
        
        if self.verbose and items:
            print(f"Created {len(items)} items for {self.split} split")
            # Show a sample
            sample = items[0] if items else None
            if sample:
                print("\nSample item:")
                print(f"ID: {sample['id']}")
                print(f"Caption: {sample['text']}")
                print(f"Audio: {sample['audio_path']}")
                print(f"Image: {sample['image_path']}")
                
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        """Get processed item with text, audio and image features."""
        item = self.items[idx]
        
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
        audio_features = {"input_values": torch.zeros(1, self.audio_sample_rate * self.audio_length)}
        if item["audio_path"]:
            audio_path = self.data_root / item["audio_path"]
            
            try:
                if os.path.exists(audio_path):
                    # Load audio file
                    audio_waveform, sr = librosa.load(str(audio_path), sr=self.audio_sample_rate)
                    
                    # Process audio
                    audio_features = process_audio(
                        audio_waveform, 
                        sr,
                        target_length=self.audio_length,
                        processor=self.audio_processor
                    )
            except Exception as e:
                if self.verbose:
                    print(f"Error processing audio file {audio_path}: {e}")
        
        # Process image
        image_features = {"pixel_values": torch.zeros(1, 3, self.image_size, self.image_size)}
        image_path = self.data_root / item["image_path"]
        
        try:
            if os.path.exists(image_path):
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                image_features = process_image(image, self.image_processor)
        except Exception as e:
            if self.verbose:
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
    audio_length=5,
    audio_sample_rate=16000,
    image_size=224,
    val_split_ratio=0.1,
    test_split_ratio=0.1,
    random_seed=42,
    subset_size=None,
    verbose=True
):
    """
    Get dataloader for the Flickr8k dataset using the first audio version and caption.
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split (train, val, test)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_text_length: Maximum text length
        audio_length: Audio length in seconds
        audio_sample_rate: Audio sample rate
        image_size: Image size for resizing
        val_split_ratio: Portion of data to use for validation
        test_split_ratio: Portion of data to use for testing
        random_seed: Random seed for reproducibility
        subset_size: Limit the dataset to this number of samples
        verbose: Print detailed information
        
    Returns:
        DataLoader for the specified dataset split
    """
    # Create dataset
    dataset = SimpleFlickr8kDataset(
        data_root=data_root,
        split=split,
        max_text_length=max_text_length,
        audio_length=audio_length,
        audio_sample_rate=audio_sample_rate,
        image_size=image_size,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
        random_seed=random_seed,
        verbose=verbose
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset for {split} split is empty! Check your data structure.")
    
    # Take a subset if specified
    if subset_size is not None and subset_size < len(dataset):
        from torch.utils.data import Subset
        import random
        
        # Set seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select subset_size indices
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = Subset(dataset, indices)
        if verbose:
            print(f"Using subset of {subset_size} samples from {split} split")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    
    if verbose:
        print(f"Created dataloader with {len(dataset)} samples, batch size {batch_size}")
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Flickr8k dataset with first audio and caption")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Flickr8k dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    # Create dataloaders
    print("Creating train dataloader...")
    train_loader = get_dataloader(
        data_root=args.data_root,
        split="train", 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\nCreating validation dataloader...")
    val_loader = get_dataloader(
        data_root=args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\nCreating test dataloader...")
    test_loader = get_dataloader(
        data_root=args.data_root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Test a batch
    print("\nTesting a batch from train loader...")
    for batch in train_loader:
        print(f"Batch size: {len(batch['id'])}")
        print(f"Audio shape: {batch['audio_values'].shape}")
        print(f"Image shape: {batch['pixel_values'].shape}")
        print(f"Text shape: {batch['input_ids'].shape}")
        break
        
    print("\nDone!")