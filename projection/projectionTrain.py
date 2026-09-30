import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import clip
from PIL import Image
import re
import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
import gc
import matplotlib.pyplot as plt

# Set environment variable to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ===================== Projection Module Code =====================

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type == 'identity':
        return IdentityMap()
    
    raise ValueError(f'Unknown projector type: {projector_type}')

# ===================== Configuration Object =====================
class Config:
    def __init__(self, mm_projector_type, mm_hidden_size, hidden_size):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size 

# ===================== 1. Dataset Definition =====================
class PretrainPairsDataset(Dataset):
    """
    Utilises pretrain_pairs.npy, which is a list where each element is (image_path, text_description).
    """
    def __init__(self, pairs_npy_path, tokenizer, prompt_template=None, max_length=256):
        self.data = np.load(pairs_npy_path, allow_pickle=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "Human: Please describe the image.\nAssistant:"
        
        print(f"Dataset loaded with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, text_desc = self.data[idx]
        
        # Only return image path, not processed image
        prompt = self.prompt_template
        return image_path, prompt, text_desc

def collate_fn(batch):
    """Simple collate function that doesn't preprocess images"""
    image_paths = [b[0] for b in batch]
    prompts = [b[1] for b in batch]
    targets = [b[2] for b in batch]
    
    return image_paths, prompts, targets

# Helper function to clear GPU cache
def clear_gpu_cache():
    torch.cuda.empty_cache()
    gc.collect()

# Function to plot and save learning curves
def plot_learning_curves(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curves.png'))
    plt.close()

# ===================== 2. Multimodal Model Definition =====================
class MultimodalModel(nn.Module):
    def __init__(self, llama_model, clip_model, projection_layer, tokenizer):
        super().__init__()
        self.llama_model = llama_model
        self.clip_model = clip_model
        self.projection_layer = projection_layer
        self.tokenizer = tokenizer
        
        # Freeze the LLM and CLIP parameters
        for param in self.llama_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.num_image_tokens = 1  # Represent each image with one "visual token"
        self.device = next(self.projection_layer.parameters()).device
        
        # Load CLIP preprocessing function
        _, self.preprocess = clip.load("ViT-B/16", device='cpu')
    
    def encode_images_batch(self, image_paths, batch_size=2):
        """Process images in small batches to reduce memory usage"""
        all_features = []
        
        # Process list of image paths
        for i in range(0, len(image_paths), batch_size):
            # Get current batch paths
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # Load and preprocess images
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # Use black image as fallback
                    img = Image.new('RGB', (224, 224), color=0)
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
            
            # Stack current batch images and move to GPU
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Process batch with CLIP
                with torch.no_grad():
                    features = self.clip_model.encode_image(batch_tensor)
                all_features.append(features)
                
                # Clean up memory
                del batch_tensor, batch_images
                clear_gpu_cache()
        
        # Merge all batch features
        if all_features:
            return torch.cat(all_features, dim=0)
        
        # Fallback: create empty feature tensor
        return torch.zeros((len(image_paths), self.clip_model.visual.output_dim), 
                           device=self.device)

    def forward(self, clip_feats, input_ids, attention_mask, labels=None):
        """Forward pass with memory optimization"""
        bsz = clip_feats.size(0)
        seq_len = input_ids.size(1)

        # Ensure half precision to save memory
        clip_feats = clip_feats.half()

        # (1) Project CLIP features
        projected = self.projection_layer(clip_feats)
        projected = projected.unsqueeze(1).expand(-1, self.num_image_tokens, -1).contiguous()
        
        # Clean up variables no longer needed
        del clip_feats
        
        # (2) Obtain text embeddings
        with torch.no_grad():
            text_embeds = self.llama_model.model.embed_tokens(input_ids)
        
        # (3) Concatenate visual and text tokens
        cat_embeds = torch.cat([projected, text_embeds], dim=1)
        new_attention_mask = torch.cat([
            torch.ones(bsz, self.num_image_tokens, dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ], dim=1)
        
        # Clean up variables no longer needed
        del projected, text_embeds
        
        # (4) Forward through LLaMA
        outputs = self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            inputs_embeds=cat_embeds
        )
        logits = outputs.logits
        
        # Clean up variables no longer needed
        del cat_embeds, new_attention_mask
        
        # (5) Compute loss
        loss = None
        if labels is not None:
            ignore = torch.full((bsz, self.num_image_tokens), -100, dtype=labels.dtype, device=labels.device)
            expanded_labels = torch.cat([ignore, labels], dim=1)
            
            # Only keep what we need for loss computation to save memory
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = expanded_labels[:, 1:].contiguous()
            
            # Clean up variables no longer needed
            del logits, expanded_labels, ignore
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Clean up variables no longer needed
            del shift_logits, shift_labels
        
        return type("ModelOutput", (), {"loss": loss, "logits": None})  # Don't return logits to save memory

# ===================== Validation Function =====================
def validate(model, val_loader, device, clip_batch_size):
    """Run validation loop and return average loss"""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    
    # Create progress bar for validation
    val_progress = tqdm.tqdm(total=len(val_loader), desc="Validation")
    
    with torch.no_grad():
        for val_image_paths, val_prompts, val_targets in val_loader:
            try:
                # Clear memory
                clear_gpu_cache()
                
                with autocast():
                    # Process images
                    val_clip_feats = model.encode_images_batch(
                        val_image_paths, 
                        batch_size=clip_batch_size
                    )
                    
                    # Prepare text inputs
                    val_batch_input = [pr + " " + tgt for pr, tgt in zip(val_prompts, val_targets)]
                    val_tokenized = model.tokenizer(
                        val_batch_input,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=256  # Use same max_length as training
                    )
                    val_input_ids = val_tokenized["input_ids"].to(device)
                    val_attention_mask = val_tokenized["attention_mask"].to(device)
                    val_labels = val_input_ids.clone()
                    
                    # Forward pass
                    val_outputs = model(val_clip_feats, val_input_ids, val_attention_mask, labels=val_labels)
                
                # Update statistics
                if val_outputs.loss is not None:
                    total_val_loss += val_outputs.loss.item()
                    val_batches += 1
                
                # Clean up variables
                del val_clip_feats, val_input_ids, val_attention_mask, val_labels, val_outputs
                
                # Update progress bar
                val_progress.update(1)
            
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    # Close progress bar
    val_progress.close()
    
    # Calculate average validation loss
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('nan')
    
    return avg_val_loss

# ===================== 3. Main Training Process =====================
def main():
    try:
        # Print available GPU memory info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total_mem - reserved
                print(f"  Total memory: {total_mem:.2f} GB")
                print(f"  Reserved memory: {reserved:.2f} GB")
                print(f"  Allocated memory: {allocated:.2f} GB")
                print(f"  Free memory: {free:.2f} GB")
        
        # Define paths and device
        train_data_path = "./dataset_train/pretrain_pairs_train.npy"
        val_data_path = "./dataset_eval/pretrain_pairs_eval.npy" 
        output_path = "./projection_trained"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Hyperparameters - memory optimized
        batch_size = 4  # Very small batch size
        num_epochs = 10
        lr = 1e-3
        max_length = 256  # Reduced sequence length
        clip_batch_size = 1  # Number of images CLIP processes at once
        grad_accumulation_steps = 8  # Increase gradient accumulation steps
        num_workers = 2  # Reduce worker processes to lower memory usage
        patience = 3  # Number of epochs with increasing validation loss before early stopping

        print(f"Using device: {device}")
        print(f"Batch size: {batch_size}, Grad accumulation: {grad_accumulation_steps}")
        print(f"Effective batch size: {batch_size * grad_accumulation_steps}")
        
        # (A) Load the CLIP model
        print("Loading CLIP model...")
        clip_model, _ = clip.load("ViT-B/16", device=device)
        clip_model.eval()
        
        # (B) Load LLaMA model
        print("Loading LLaMA model...")
        llama_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(
            llama_name, 
            use_auth_token=os.getenv("HF_TOKEN")
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("Loading LLM, this may take a few minutes...")
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_name, 
            use_auth_token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.float16  # Use half precision
        ).to(device)
        
        # Enable gradient checkpointing to save memory
        llama_model.gradient_checkpointing_enable()
        llama_model.eval()

        # (C) Determine embedding dimensions
        embed_dim = llama_model.model.embed_tokens.weight.shape[1]
        clip_output_dim = clip_model.visual.output_dim

        print(f"LLaMA embedding dimension: {embed_dim}")
        print(f"CLIP output dimension: {clip_output_dim}")

        # (D) Create projection configuration
        proj_config = Config(
            mm_projector_type="mlp2x_gelu",
            mm_hidden_size=clip_output_dim,
            hidden_size=embed_dim
        )
        projection_layer = build_vision_projector(proj_config).to(device)
        
        # (E) Build multimodal model
        multi_model = MultimodalModel(
            llama_model=llama_model,
            clip_model=clip_model,
            projection_layer=projection_layer,
            tokenizer=tokenizer
        ).to(device)
        
        # (F) Data loading - training and validation datasets
        print("Setting up training dataset...")
        train_dataset = PretrainPairsDataset(
            pairs_npy_path=train_data_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=num_workers,
            pin_memory=False  # Disable pin_memory to reduce memory usage
        )
        
        print("Setting up validation dataset...")
        val_dataset = PretrainPairsDataset(
            pairs_npy_path=val_data_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # No need to shuffle validation data
            collate_fn=collate_fn, 
            num_workers=num_workers,
            pin_memory=False
        )
        
        # (G) Define optimizer
        optimizer = optim.AdamW(multi_model.projection_layer.parameters(), lr=lr)
        
        # Use gradient scaling for mixed precision training
        scaler = GradScaler()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize tracking variables for early stopping
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        overfit_counter = 0
        early_stop = False
        best_model_path = os.path.join(output_path, "best_model.pth")
        
        # (H) Training loop with validation
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            total_loss = 0.0
            processed_batches = 0
            epoch_start_time = time.time()
            
            # Create progress bar
            progress_bar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            multi_model.train()
            optimizer.zero_grad()  # Zero gradients at the beginning of epoch
            
            for batch_idx, (image_paths, prompts, target_texts) in enumerate(train_loader):
                try:
                    # Clear memory from previous batch
                    clear_gpu_cache()
                    
                    # Gradient accumulation step
                    is_accumulation_step = (batch_idx + 1) % grad_accumulation_steps != 0
                    
                    with autocast():
                        # Process images - in small batches to reduce memory usage
                        clip_feats_batch = multi_model.encode_images_batch(
                            image_paths, 
                            batch_size=clip_batch_size
                        )
                        
                        # Prepare text inputs
                        batch_input = [pr + " " + tgt for pr, tgt in zip(prompts, target_texts)]
                        tokenized = tokenizer(
                            batch_input,
                            return_tensors='pt',
                            padding='max_length',
                            truncation=True,
                            max_length=max_length
                        )
                        input_ids = tokenized["input_ids"].to(device)
                        attention_mask = tokenized["attention_mask"].to(device)
                        labels = input_ids.clone()
                        
                        # Forward pass
                        outputs = multi_model(clip_feats_batch, input_ids, attention_mask, labels=labels)
                        loss = outputs.loss / grad_accumulation_steps  # Scale loss for gradient accumulation
                    
                    # Use gradient scaling for backward pass
                    scaler.scale(loss).backward()
                    
                    # Clean up variables no longer needed
                    del clip_feats_batch, input_ids, attention_mask, labels, outputs
                    clear_gpu_cache()
                    
                    # Only update parameters at the end of accumulation steps
                    if not is_accumulation_step or batch_idx == len(train_loader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    # Update statistics
                    if loss is not None:
                        total_loss += loss.item() * grad_accumulation_steps
                        processed_batches += 1
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": loss.item() * grad_accumulation_steps if loss is not None else "N/A"
                    })
                    
                    # Clean up loss variable
                    del loss
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Continue to next batch
                    continue
            
            # Close progress bar
            progress_bar.close()
            
            # Calculate epoch statistics
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = total_loss / processed_batches if processed_batches > 0 else float('nan')
            train_losses.append(avg_train_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} training stats:")
            print(f"  Train Loss = {avg_train_loss:.4f}")
            print(f"  Train Time = {epoch_time:.2f}s")
            
            # Validation phase
            print("Running validation...")
            val_start_time = time.time()
            avg_val_loss = validate(multi_model, val_loader, device, clip_batch_size)
            val_time = time.time() - val_start_time
            
            print(f"  Validation Loss = {avg_val_loss:.4f}")
            print(f"  Validation Time = {val_time:.2f}s")
            
            val_losses.append(avg_val_loss)
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                overfit_counter = 0
                # Save best model
                torch.save(multi_model.projection_layer.state_dict(), best_model_path)
                print(f"  New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                overfit_counter += 1
                print(f"  Validation loss did not improve. Counter: {overfit_counter}/{patience}")
                
                # Check for early stopping
                if overfit_counter >= patience:
                    print("Early stopping triggered!")
                    early_stop = True
            
            # Save checkpoint for this epoch
            checkpoint_path = os.path.join(output_path, f"projection_layer_epoch{epoch+1}.pth")
            torch.save(multi_model.projection_layer.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            
            # Plot and save learning curves after each epoch
            plot_learning_curves(train_losses, val_losses, output_path)
            print(f"  Updated learning curves plot")
            
            # Clear cache
            clear_gpu_cache()
            
            # Check for early stopping
            if early_stop:
                print("Training stopped early due to overfitting")
                break
        
        # (I) Save final model
        final_path = os.path.join(output_path, f"projection_layer_final.pth")
        torch.save(multi_model.projection_layer.state_dict(), final_path)
        print(f"Training completed. Final model saved to {final_path}")
        
        # Plot final learning curves
        plot_learning_curves(train_losses, val_losses, output_path)
        print(f"Final learning curves saved to {os.path.join(output_path, 'loss_curves.png')}")
        
        # Save loss values as CSV for future reference
        loss_data = np.column_stack((range(1, len(train_losses) + 1), train_losses, val_losses))
        np.savetxt(
            os.path.join(output_path, 'loss_data.csv'),
            loss_data,
            delimiter=',',
            header='epoch,train_loss,val_loss',
            comments=''
        )
    
    except Exception as e:
        print(f"Training error: {e}")
        # Try to save emergency model
        try:
            emergency_path = os.path.join(output_path, "projection_layer_emergency.pth")
            torch.save(multi_model.projection_layer.state_dict(), emergency_path)
            print(f"Saved emergency checkpoint to {emergency_path}")
        except:
            print("Could not save emergency checkpoint")

if __name__ == "__main__":
    main()