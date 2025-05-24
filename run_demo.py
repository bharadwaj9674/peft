"""
Demo script for cross-modal retrieval with Flickr8k dataset integration.
"""

import os
import torch
import argparse
import yaml
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import librosa
import librosa.display
import json
from pathlib import Path
import tempfile
from tqdm import tqdm
from transformers import RobertaTokenizer, AutoFeatureExtractor

from src.models.encoders import TextEncoder, AudioEncoder, ImageEncoder
from src.models.peft_modules import PEFTManager
from src.models.cman import CMAN
from src.utils import process_audio, process_image


# Wrapper functions to adapt the interface
def encode_text_wrapper(model, input_ids, attention_mask):
    """Wrapper to encode text using CMAN's forward method."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs["text_embeddings"]

def encode_audio_wrapper(model, input_values):
    """Wrapper to encode audio using CMAN's forward method."""
    outputs = model(input_values=input_values)
    return outputs["audio_embeddings"]

def encode_image_wrapper(model, pixel_values):
    """Wrapper to encode image using CMAN's forward method."""
    outputs = model(pixel_values=pixel_values)
    return outputs["image_embeddings"]


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
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


class GalleryManager:
    """Manages the gallery items for the demo."""
    
    def __init__(self, data_root, split="test", limit=500):
        """
        Initialize the gallery manager.
        
        Args:
            data_root: Root directory of the dataset
            split: Dataset split to use for the gallery (test is recommended)
            limit: Maximum number of gallery items to load
        """
        self.data_root = Path(data_root)
        self.split = split
        self.limit = limit
        
        # Load metadata
        metadata_file = self.data_root / "metadata" / f"{split}_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
        # Limit the number of gallery items
        if limit and limit < len(self.metadata):
            self.metadata = self.metadata[:limit]
        
        # Initialize tokenizer and feature extractors
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.audio_processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        
        # Initialize embeddings cache
        self.text_embeddings = None
        self.audio_embeddings = None
        self.image_embeddings = None
        
        print(f"Gallery initialized with {len(self.metadata)} items from {split} split")
    
    def compute_gallery_embeddings(self, model, device):
        """
        Compute embeddings for all gallery items.
        
        Args:
            model: Cross-modal model
            device: Device to use for computation
        """
        print("Computing gallery embeddings...")
        
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Initialize lists to store embeddings
        text_embeddings = []
        audio_embeddings = []
        image_embeddings = []
        
        # Process gallery items in batches
        batch_size = 16  # Reduced batch size to avoid memory issues
        num_batches = (len(self.metadata) + batch_size - 1) // batch_size
        
        model.eval()
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing gallery embeddings"):
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.metadata))
                batch_metadata = self.metadata[start_idx:end_idx]
                
                # Process text
                text_batch = []
                for item in batch_metadata:
                    text_encoding = self.tokenizer(
                        item["caption"],
                        max_length=77,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    text_batch.append({
                        "input_ids": text_encoding["input_ids"].squeeze(0),
                        "attention_mask": text_encoding["attention_mask"].squeeze(0)
                    })
                
                if text_batch:
                    # Stack all tensors into a batch
                    input_ids = torch.stack([item["input_ids"] for item in text_batch])
                    attention_mask = torch.stack([item["attention_mask"] for item in text_batch])
                    
                    # Move to the same device as the model
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    # Get text embeddings using the wrapper
                    batch_text_embeddings = encode_text_wrapper(model, input_ids, attention_mask)
                    
                    # Move back to CPU to save memory
                    text_embeddings.append(batch_text_embeddings.cpu())
                
                # Process audio files
                valid_audio_inputs = []
                valid_audio_indices = []
                
                for i, item in enumerate(batch_metadata):
                    audio_path = self.data_root / "audio" / self.split / item["audio_filename"]
                    if audio_path.exists():
                        try:
                            waveform, sr = librosa.load(str(audio_path), sr=16000)
                            audio_features = process_audio(waveform, sr, processor=self.audio_processor)
                            valid_audio_inputs.append(audio_features["input_values"].squeeze(0))
                            valid_audio_indices.append(i)
                        except Exception as e:
                            print(f"Error processing audio file {audio_path}: {e}")
                
                if valid_audio_inputs:
                    # Stack into batch
                    audio_values = torch.stack(valid_audio_inputs).to(device)
                    
                    # Get audio embeddings using the wrapper
                    batch_audio_embeddings = encode_audio_wrapper(model, audio_values)
                    batch_audio_embeddings = batch_audio_embeddings.cpu()
                    
                    # Map back to original indices
                    for i, orig_idx in enumerate(valid_audio_indices):
                        audio_embeddings.append((start_idx + orig_idx, batch_audio_embeddings[i]))
                
                # Fill in missing audio embeddings with None
                for i in range(start_idx, end_idx):
                    if not any(idx == i for idx, _ in audio_embeddings):
                        audio_embeddings.append((i, None))
                
                # Process image files
                valid_image_inputs = []
                valid_image_indices = []
                
                for i, item in enumerate(batch_metadata):
                    image_path = self.data_root / "images" / self.split / item["image_filename"]
                    if image_path.exists():
                        try:
                            image = Image.open(image_path).convert("RGB")
                            image_features = process_image(image, self.image_processor)
                            valid_image_inputs.append(image_features["pixel_values"].squeeze(0))
                            valid_image_indices.append(i)
                        except Exception as e:
                            print(f"Error processing image file {image_path}: {e}")
                
                if valid_image_inputs:
                    # Stack into batch
                    pixel_values = torch.stack(valid_image_inputs).to(device)
                    
                    # Get image embeddings using the wrapper
                    batch_image_embeddings = encode_image_wrapper(model, pixel_values)
                    batch_image_embeddings = batch_image_embeddings.cpu()
                    
                    # Map back to original indices
                    for i, orig_idx in enumerate(valid_image_indices):
                        image_embeddings.append((start_idx + orig_idx, batch_image_embeddings[i]))
                
                # Fill in missing image embeddings with None
                for i in range(start_idx, end_idx):
                    if not any(idx == i for idx, _ in image_embeddings):
                        image_embeddings.append((i, None))
                
                # Clear memory
                torch.cuda.empty_cache()
        
        # Concatenate embeddings and process for storage
        if text_embeddings:
            self.text_embeddings = torch.cat(text_embeddings, dim=0)
        
        # Sort audio embeddings by index and extract just the embeddings
        audio_embeddings.sort(key=lambda x: x[0])
        self.audio_embeddings = [item[1] for item in audio_embeddings]
        
        # Sort image embeddings by index and extract just the embeddings
        image_embeddings.sort(key=lambda x: x[0])
        self.image_embeddings = [item[1] for item in image_embeddings]
        
        print("Gallery embeddings computed successfully.")
    
    def search_text(self, query_embedding, top_k=5):
        """
        Search for text matches to the query.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top results to return
            
        Returns:
            List of {index, similarity, caption} dictionaries
        """
        if self.text_embeddings is None:
            return []
        
        query_embedding = query_embedding.cpu()

        # Compute similarities
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        gallery_embeddings = torch.nn.functional.normalize(self.text_embeddings, p=2, dim=1)
        similarities = torch.matmul(query_embedding, gallery_embeddings.T).squeeze(0)
        
        # Get top-k matches
        top_values, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        # Prepare results
        results = []
        for i, (index, similarity) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
            results.append({
                "index": index,
                "similarity": similarity,
                "caption": self.metadata[index]["caption"]
            })
        
        return results
    
    def search_audio(self, query_embedding, top_k=5):
        """
        Search for audio matches to the query.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top results to return
            
        Returns:
            List of {index, similarity, path} dictionaries
        """
        if not self.audio_embeddings:
            return []
        
        query_embedding = query_embedding.detach().cpu()
        
        # Compute similarities for non-None embeddings
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        similarities = []
        
        for i, embedding in enumerate(self.audio_embeddings):
            if embedding is not None:
                embedding = embedding.detach().cpu()
                embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1)
                similarity = torch.matmul(query_embedding, embedding.T).item()
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k matches
        top_matches = similarities[:min(top_k, len(similarities))]
        
        # Prepare results
        results = []
        for index, similarity in top_matches:
            item = self.metadata[index]
            audio_path = self.data_root / "audio" / self.split / item["audio_filename"]
            results.append({
                "index": index,
                "similarity": similarity,
                "caption": item["caption"],
                "path": str(audio_path)
            })
        
        return results
    
    def search_image(self, query_embedding, top_k=5):
        """
        Search for image matches to the query.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top results to return
            
        Returns:
            List of {index, similarity, path} dictionaries
        """
        if not self.image_embeddings:
            return []
        
        query_embedding = query_embedding.cpu()

        # Compute similarities for non-None embeddings
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        similarities = []
        
        for i, embedding in enumerate(self.image_embeddings):
            if embedding is not None:
                embedding = embedding.detach().cpu()
                embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1)
                similarity = torch.matmul(query_embedding, embedding.T).item()
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k matches
        top_matches = similarities[:min(top_k, len(similarities))]
        
        # Prepare results
        results = []
        for index, similarity in top_matches:
            item = self.metadata[index]
            image_path = self.data_root / "images" / self.split / item["image_filename"]
            results.append({
                "index": index,
                "similarity": similarity,
                "caption": item["caption"],
                "path": str(image_path)
            })
        
        return results
    
    def get_audio_path(self, index):
        """Get audio file path for the given index."""
        item = self.metadata[index]
        return str(self.data_root / "audio" / self.split / item["audio_filename"])
    
    def get_image_path(self, index):
        """Get image file path for the given index."""
        item = self.metadata[index]
        return str(self.data_root / "images" / self.split / item["image_filename"])


def demo_app(model, config, gallery_manager):
    """Create Streamlit demo app."""
    st.title("Cross-Modal Retrieval Demo")
    st.sidebar.title("Parameter-Efficient Fine-Tuning (PEFT)")
    
    # Display PEFT method info
    if config.peft.enabled:
        st.sidebar.markdown(f"**PEFT Method:** {config.peft.method}")
        
        # Display parameter efficiency
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        efficiency = 100 * trainable_params / total_params
        
        st.sidebar.markdown(f"**Parameter Efficiency:**")
        st.sidebar.markdown(f"- Trainable: {trainable_params:,} ({efficiency:.2f}%)")
        st.sidebar.markdown(f"- Total: {total_params:,}")
    else:
        st.sidebar.markdown("**PEFT not enabled**")
    
    # Initialize tokenizer and feature extractors
    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    audio_processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch32-224-in21k")
    
    # Set up tabs for different query types
    tab1, tab2, tab3 = st.tabs(["Text to Audio/Image", "Audio to Text/Image", "Image to Text/Audio"])
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Tab 1: Text to Audio/Image
    with tab1:
        st.header("Text Query")
        text_query = st.text_area("Enter your text query:", "A dog barking")
        
        if st.button("Search with Text", key="text_search"):
            with torch.no_grad():
                # Tokenize text
                text_encoding = tokenizer(
                    text_query,
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = text_encoding["input_ids"].to(device)
                attention_mask = text_encoding["attention_mask"].to(device)
                
                # Get text embedding using the wrapper
                text_embedding = encode_text_wrapper(model, input_ids, attention_mask)
                
                # Search gallery
                with st.spinner("Searching gallery..."):
                    audio_matches = gallery_manager.search_audio(text_embedding, top_k=3)
                    image_matches = gallery_manager.search_image(text_embedding, top_k=3)
                
                # Display results
                if audio_matches:
                    st.subheader("Audio Matches")
                    cols = st.columns(min(3, len(audio_matches)))
                    
                    for i, match in enumerate(audio_matches[:3]):
                        with cols[i]:
                            # Load audio
                            try:
                                audio_path = match["path"]
                                waveform, sr = librosa.load(audio_path, sr=16000)
                                
                                # Create spectrogram
                                fig, ax = plt.subplots(figsize=(3, 2))
                                S = librosa.feature.melspectrogram(y=waveform, sr=sr)
                                S_dB = librosa.power_to_db(S, ref=np.max)
                                librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                                ax.set_title(f"Similarity: {match['similarity']:.2f}")
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Display audio
                                st.audio(audio_path)
                                
                                # Display caption
                                st.caption(match["caption"])
                            except Exception as e:
                                st.error(f"Error displaying audio: {e}")
                else:
                    st.warning("No audio matches found.")
                
                if image_matches:
                    st.subheader("Image Matches")
                    cols = st.columns(min(3, len(image_matches)))
                    
                    for i, match in enumerate(image_matches[:3]):
                        with cols[i]:
                            # Load and display image
                            try:
                                image_path = match["path"]
                                image = Image.open(image_path).convert("RGB")
                                st.image(image, caption=f"Similarity: {match['similarity']:.2f}")
                                
                                # Display caption
                                st.caption(match["caption"])
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                else:
                    st.warning("No image matches found.")
    
    # Tab 2: Audio to Text/Image
    with tab2:
        st.header("Audio Query")
        audio_file = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg"])
        
        if audio_file is not None:
            st.audio(audio_file)
            
            if st.button("Search with Audio", key="audio_search"):
                with torch.no_grad():
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_file.getvalue())
                        audio_path = tmp_file.name
                    
                    # Load and process audio
                    try:
                        waveform, sr = librosa.load(audio_path, sr=16000)
                        audio_features = process_audio(waveform, sr, processor=audio_processor)
                        
                        # Move to device
                        input_values = audio_features["input_values"].to(device)
                        
                        # Get audio embedding using the wrapper
                        audio_embedding = encode_audio_wrapper(model, input_values)
                        
                        # Search gallery
                        with st.spinner("Searching gallery..."):
                            text_matches = gallery_manager.search_text(audio_embedding, top_k=5)
                            image_matches = gallery_manager.search_image(audio_embedding, top_k=3)
                        
                        # Display text results
                        if text_matches:
                            st.subheader("Text Matches")
                            
                            for i, match in enumerate(text_matches):
                                st.markdown(f"**{i+1}. {match['caption']}** (Similarity: {match['similarity']:.2f})")
                        else:
                            st.warning("No text matches found.")
                        
                        # Display image results
                        if image_matches:
                            st.subheader("Image Matches")
                            cols = st.columns(min(3, len(image_matches)))
                            
                            for i, match in enumerate(image_matches[:3]):
                                with cols[i]:
                                    # Load and display image
                                    try:
                                        image_path = match["path"]
                                        image = Image.open(image_path).convert("RGB")
                                        st.image(image, caption=f"Similarity: {match['similarity']:.2f}")
                                        
                                        # Display caption
                                        st.caption(match["caption"])
                                    except Exception as e:
                                        st.error(f"Error displaying image: {e}")
                        else:
                            st.warning("No image matches found.")
                    
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
    
    # Tab 3: Image to Text/Audio
    with tab3:
        st.header("Image Query")
        image_file = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Search with Image", key="image_search"):
                with torch.no_grad():
                    try:
                        # Process image - ensure proper shape
                        image_features = process_image(image, image_processor)
                        
                        # Debug the shape
                        pixel_values = image_features["pixel_values"]
                        
                        # Ensure proper 4D shape: [batch_size, channels, height, width]
                        # If it's missing batch dimension, add it
                        if len(pixel_values.shape) == 3:
                            pixel_values = pixel_values.unsqueeze(0)
                        
                        # If tensor has unexpected shape, reshape it correctly
                        if len(pixel_values.shape) != 4:
                            st.error(f"Unexpected pixel_values shape: {pixel_values.shape}. Attempting to fix...")
                            # This is a fallback solution if the shape is completely wrong
                            if hasattr(image_processor, "size"):
                                size = image_processor.size.get("height", 224)
                            else:
                                size = 224  # Default for ViT
                            
                            # Resize image and convert to tensor manually
                            resized_img = image.resize((size, size))
                            img_array = np.array(resized_img).transpose(2, 0, 1)  # Convert to [C, H, W]
                            pixel_values = torch.from_numpy(img_array).float().unsqueeze(0)  # Add batch dim
                        
                        # Move to device
                        pixel_values = pixel_values.to(device)
                        
                        # Get image embedding using the wrapper
                        image_embedding = encode_image_wrapper(model, pixel_values)
                        
                        # Search gallery
                        with st.spinner("Searching gallery..."):
                            text_matches = gallery_manager.search_text(image_embedding, top_k=5)
                            audio_matches = gallery_manager.search_audio(image_embedding, top_k=3)
                        
                        # Display text results
                        if text_matches:
                            st.subheader("Text Matches")
                            
                            for i, match in enumerate(text_matches):
                                st.markdown(f"**{i+1}. {match['caption']}** (Similarity: {match['similarity']:.2f})")
                        else:
                            st.warning("No text matches found.")
                        
                        # Display audio results
                        if audio_matches:
                            st.subheader("Audio Matches")
                            cols = st.columns(min(3, len(audio_matches)))
                            
                            for i, match in enumerate(audio_matches[:3]):
                                with cols[i]:
                                    # Load audio
                                    try:
                                        audio_path = match["path"]
                                        waveform, sr = librosa.load(audio_path, sr=16000)
                                        
                                        # Create spectrogram
                                        fig, ax = plt.subplots(figsize=(3, 2))
                                        S = librosa.feature.melspectrogram(y=waveform, sr=sr)
                                        S_dB = librosa.power_to_db(S, ref=np.max)
                                        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                                        ax.set_title(f"Similarity: {match['similarity']:.2f}")
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Display audio
                                        st.audio(audio_path)
                                        
                                        # Display caption
                                        st.caption(match["caption"])
                                    except Exception as e:
                                        st.error(f"Error displaying audio: {e}")
                        else:
                            st.warning("No audio matches found.")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.error(f"Error details: {type(e).__name__}")
                        import traceback
                        st.code(traceback.format_exc())


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-Modal Retrieval Demo")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="../cross_modal_peft/data/flickr8k_audio", help="Path to dataset root")
    parser.add_argument("--gallery_split", type=str, default="test", help="Dataset split to use for gallery")
    parser.add_argument("--gallery_limit", type=int, default=100, help="Maximum number of gallery items")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.checkpoint, config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize gallery manager
    gallery_manager = GalleryManager(
        data_root=args.data_root,
        split=args.gallery_split,
        limit=args.gallery_limit
    )
    
    # Compute gallery embeddings
    gallery_manager.compute_gallery_embeddings(model, device)
    
    # Launch demo app
    demo_app(model, config, gallery_manager)


if __name__ == "__main__":
    main()