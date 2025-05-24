"""
Trainer class for cross-modal retrieval model.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import time
import logging
from torch.cuda.amp import autocast, GradScaler

from .utils import prepare_batch_for_model
from .evaluator import compute_retrieval_metrics


class Trainer:
    """
    Trainer for cross-modal retrieval models.
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer=None,
        scheduler=None,
        num_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="./outputs",
        log_interval=10,
        save_interval=1,
        eval_interval=1,
        mixed_precision=False
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer if not provided
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Initialize scheduler if not provided
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs * len(train_dataloader)
            )
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(os.path.join(output_dir, "logs"))
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Train the model for the specified number of epochs."""
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        
        # Record start time
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training loop
            train_loss = self._train_epoch_with_mixed_precision()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}")
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            
            # Evaluate model if needed
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.num_epochs - 1:
                val_loss, val_metrics = self._evaluate()
                
                # Log validation results
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Val Loss: {val_loss:.4f}")
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                
                for k, v in val_metrics.items():
                    self.logger.info(f"  {k}: {v:.4f}")
                    self.writer.add_scalar(f"Metrics/{k}", v, epoch)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(name="best")
                    self.logger.info(f"New best model saved with val loss: {val_loss:.4f}")
            
            # Save checkpoint if needed
            if (epoch + 1) % self.save_interval == 0 or epoch == self.num_epochs - 1:
                self._save_checkpoint(name=f"epoch_{epoch+1}")
        
        # Record end time and log total training time
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.model
    
    def _train_epoch_with_mixed_precision(self):
        """Train the model for one epoch with optional mixed precision."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)

        # Progress bar for training
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch for model
            model_inputs = prepare_batch_for_model(batch, self.device)
    
            # Forward pass with loss computation using mixed precision if enabled
            model_inputs["return_loss"] = True
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(**model_inputs)
                    loss = outputs["loss"]
                    
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(**model_inputs)
                loss = outputs["loss"]
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log metrics
            if batch_idx % self.log_interval == 0:
                step_loss = loss.item()
                progress_bar.set_postfix({"loss": f"{step_loss:.4f}"})
                self.writer.add_scalar("Loss/step", step_loss, self.global_step)
                self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.global_step)
                
        # Return average loss for the epoch
        return total_loss / num_batches
    
    def _evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        # Collect embeddings for retrieval evaluation
        text_embeddings = []
        audio_embeddings = []
        image_embeddings = []
        ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Get batch IDs
                batch_ids = batch.pop("id") if "id" in batch else None
                
                # Prepare batch for model
                model_inputs = prepare_batch_for_model(batch, self.device)
                
                # Forward pass with loss computation
                model_inputs["return_loss"] = True
                outputs = self.model(**model_inputs)
                
                # Accumulate loss
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                
                # Collect embeddings for retrieval evaluation
                if "text_embeddings" in outputs:
                    text_embeddings.append(outputs["text_embeddings"].cpu())
                if "audio_embeddings" in outputs:
                    audio_embeddings.append(outputs["audio_embeddings"].cpu())
                if "image_embeddings" in outputs:
                    image_embeddings.append(outputs["image_embeddings"].cpu())
                
                # Collect IDs if available
                if batch_ids is not None:
                    ids.extend(batch_ids)
        
        # Compute average loss
        val_loss = total_loss / max(1, num_batches)
        
        # Compute retrieval metrics if we have embeddings
        val_metrics = {}
        if text_embeddings and audio_embeddings:
            text_embs = torch.cat(text_embeddings, dim=0)
            audio_embs = torch.cat(audio_embeddings, dim=0)
            
            # Compute text-to-audio and audio-to-text retrieval metrics
            t2a_metrics = compute_retrieval_metrics(text_embs, audio_embs)
            a2t_metrics = compute_retrieval_metrics(audio_embs, text_embs)
            
            # Combine metrics
            for k, v in t2a_metrics.items():
                val_metrics[f"t2a_{k}"] = v
            for k, v in a2t_metrics.items():
                val_metrics[f"a2t_{k}"] = v
        
        # Add image-related metrics if available
        if text_embeddings and image_embeddings:
            text_embs = torch.cat(text_embeddings, dim=0)
            image_embs = torch.cat(image_embeddings, dim=0)
            
            # Compute text-to-image and image-to-text retrieval metrics
            t2i_metrics = compute_retrieval_metrics(text_embs, image_embs)
            i2t_metrics = compute_retrieval_metrics(image_embs, text_embs)
            
            # Combine metrics
            for k, v in t2i_metrics.items():
                val_metrics[f"t2i_{k}"] = v
            for k, v in i2t_metrics.items():
                val_metrics[f"i2t_{k}"] = v
        
        return val_loss, val_metrics
    
    def _save_checkpoint(self, name="checkpoint"):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch + 1}")
        
        return True