import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file as safe_load
import clip
from PIL import Image
import re
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

class ProjectionBasedFeatureExtractor:
    def __init__(self, projection_path, llama_repo="meta-llama/Llama-3.2-1B"):
        """
        Initialize feature extractor with your trained projection layer
        
        Args:
            projection_path: Path to your trained projection weights (best_model.pth)
            llama_repo: LLaMA model repository for getting hidden size
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.clip_model.eval()
        
        # Get LLaMA config for hidden size
        from transformers import AutoConfig
        llama_config = AutoConfig.from_pretrained(llama_repo)
        
        # Build and load projection layer
        self.proj = self.build_projector(
            self.clip_model.visual.output_dim, 
            llama_config.hidden_size, 
            "mlp2x_gelu"
        ).to(self.device)
        
        # Load your trained projection weights
        self.proj.load_state_dict(torch.load(projection_path, map_location=self.device))
        self.proj.eval()
        
        print(f"Loaded projection from: {projection_path}")
        print(f"CLIP output dim: {self.clip_model.visual.output_dim}")
        print(f"LLaMA hidden size: {llama_config.hidden_size}")
    
    def build_projector(self, in_dim: int, hidden_dim: int, style="mlp2x_gelu") -> nn.Module:
        """Build projection layer - same as in your baseline"""
        if style == "linear":
            return nn.Linear(in_dim, hidden_dim)
        m = re.match(r"mlp(\d+)x_gelu", style)
        if m:
            depth = int(m.group(1))
            layers = [nn.Linear(in_dim, hidden_dim)]
            for _ in range(depth - 1):
                layers += [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
            return nn.Sequential(*layers)
        raise ValueError("Unknown projector style")
    
    def nifti_to_rgb(self, path: str):
        """Convert NIFTI to RGB - same as in your baseline"""
        nii = nib.load(path)
        data = nii.get_fdata()
        if data.ndim == 4:
            data = data[..., data.shape[3] // 2]
        slice_ = data[..., data.shape[2] // 2]
        slice_ = (slice_ - slice_.min()) / (np.ptp(slice_) + 1e-6) * 255
        pil = Image.fromarray(slice_.astype(np.uint8)).convert("RGB")
        return pil
    
    def extract_projected_features(self, image_paths):
        """
        Extract features using CLIP and your trained projection
        This matches the encode_imgs method in your MultiModel
        """
        feats = []
        
        for p in image_paths:
            if not os.path.exists(p):
                print(f"Warning: missing image {p}")
                continue
                
            if p.endswith('.nii.gz'):
                pil = self.nifti_to_rgb(p)
            else:
                pil = Image.open(p).convert("RGB")
                
            t = self.preprocess(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Extract CLIP features
                clip_features = self.clip_model.encode_image(t)
                feats.append(clip_features)
        
        # Average features if multiple images - same as your baseline
        if feats:
            avg_feat = torch.mean(torch.stack(feats), dim=0)
            avg_feat = avg_feat.view(1, -1).float()
            
            # Apply projection - this is the key part!
            with torch.no_grad():
                projected_feat = self.proj(avg_feat)
            
            return projected_feat.cpu().numpy().flatten()
        else:
            return None

def convert_dataset_with_projection(
    input_json_path, 
    output_json_path, 
    projection_path,
    llama_repo="meta-llama/Llama-3.2-1B"
):
    """
    Convert multimodal dataset to text dataset with projected features
    
    Args:
        input_json_path: Path to original dataset (train.json)
        output_json_path: Path to save converted dataset
        projection_path: Path to your trained projection weights
        llama_repo: LLaMA model repository
    """
    
    # Initialize feature extractor with your projection
    print("Initializing feature extractor with trained projection...")
    extractor = ProjectionBasedFeatureExtractor(projection_path, llama_repo)
    
    # Load dataset
    print(f"Loading dataset from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each sample
    processed_data = []
    
    for item in tqdm(data, desc="Processing samples"):
        new_item = {
            "id": item["id"],
            "conversations": []
        }
        
        # Extract projected features from images
        if "images" in item and item["images"]:
            # Extract features using your projection
            projected_features = extractor.extract_projected_features(item["images"])
            
            if projected_features is not None:
                # Convert to compact representation
                # Use a subset of features to keep text reasonable
                feature_subset = projected_features[:50]  # First 50 dimensions
                feature_str = ",".join([f"{x:.4f}" for x in feature_subset])
                feature_text = f"[PROJECTED_FEATURES:{feature_str}]"
                
                # Also add human-readable summary
                feature_stats = (
                    f"[Feature Stats: "
                    f"mean={np.mean(projected_features):.4f}, "
                    f"std={np.std(projected_features):.4f}, "
                    f"norm={np.linalg.norm(projected_features):.4f}]"
                )
            else:
                feature_text = "[NO_FEATURES]"
                feature_stats = ""
        else:
            feature_text = "[NO_IMAGES]"
            feature_stats = ""
        
        # Process conversations
        for conv in item["conversations"]:
            new_conv = {"from": conv["from"]}
            
            if conv["from"] == "human":
                # Replace <image> with projected features
                value = conv["value"]
                
                # Add view information from image paths
                view_info = ""
                if "images" in item:
                    for img_path in item["images"]:
                        if "lax_2c" in img_path:
                            view_info += "[2-chamber view] "
                        elif "lax_3c" in img_path:
                            view_info += "[3-chamber view] "
                        elif "lax_4c" in img_path:
                            view_info += "[4-chamber view] "
                
                # Replace <image> with all information
                image_replacement = f"{view_info}{feature_stats} {feature_text}"
                value = value.replace("<image>", image_replacement.strip())
                
                new_conv["value"] = value
            else:
                new_conv["value"] = conv["value"]
            
            new_item["conversations"].append(new_conv)
        
        processed_data.append(new_item)
    
    # Save processed dataset
    print(f"Saving processed dataset to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed {len(processed_data)} samples")
    
    return len(processed_data)

def create_dataset_info():
    """Create dataset_info.json for LLaMA-Factory"""
    dataset_info = {
        "train_medical_projected": {
            "file_name": "train_with_projection.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        },
        "val_medical_projected": {
            "file_name": "val_with_projection.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    }
    
    with open("dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print("Created dataset_info.json")

if __name__ == "__main__":
    # ========== CHANGE THESE PATHS TO YOUR PATHS ==========
    # Path to your trained projection weights
    PROJECTION_PATH = r".\projection\best_model.pth"
    
    # Path to your dataset files
    TRAIN_JSON = "data/processed/train.json"
    VAL_JSON = "data/processed/val.json"  # Optional
    
    # Output paths
    TRAIN_OUTPUT = "train_with_projection.json"
    VAL_OUTPUT = "val_with_projection.json"
    
    # LLaMA model (should match what you use in GUI)
    LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    # =====================================================
    
    # Convert training set
    print("\nConverting training dataset...")
    convert_dataset_with_projection(
        TRAIN_JSON, 
        TRAIN_OUTPUT, 
        PROJECTION_PATH,
        LLAMA_MODEL
    )
    
    # Convert validation set if exists
    if os.path.exists(VAL_JSON):
        print("\nConverting validation dataset...")
        convert_dataset_with_projection(
            VAL_JSON, 
            VAL_OUTPUT, 
            PROJECTION_PATH,
            LLAMA_MODEL
        )
    
    # Create dataset_info.json
    create_dataset_info()
    
    print("\nDataset conversion complete!")
    print(f"Now use '{TRAIN_OUTPUT}' in LLaMA-Factory GUI")
    print("Select dataset name: 'train_medical_projected'")