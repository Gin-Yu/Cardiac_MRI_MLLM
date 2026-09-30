import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import clip
from PIL import Image
import re

# -------------------- Projection Module --------------------
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}

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
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type == "identity":
        return IdentityMap()
    
    raise ValueError(f"Unknown projector type: {projector_type}")

# -------------------- Configuration Object --------------------
class Config:
    def __init__(self, mm_projector_type, mm_hidden_size, hidden_size):
        self.mm_projector_type = mm_projector_type  # Options: 'linear', 'mlp2x_gelu', 'identity', etc.
        self.mm_hidden_size = mm_hidden_size        # e.g. CLIP feature dimension
        self.hidden_size = hidden_size              # e.g. LLaMA embedding dimension

# -------------------- Multimodal Model Definition --------------------
class MultimodalModel(nn.Module):
    def __init__(self, llama_model, clip_model, projection_layer, tokenizer):
        super().__init__()
        self.llama_model = llama_model
        self.clip_model = clip_model
        self.projection_layer = projection_layer
        self.tokenizer = tokenizer
        
        # Freeze LLaMA and CLIP parameters
        for param in self.llama_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.num_image_tokens = 1  # Each image is represented by one "visual token"

    def encode_image(self, image):
        # Use CLIP's default preprocessing (ViT-B/16 in this case)
        preprocess = clip.load("ViT-B/16")[1]
        img_preprocessed = preprocess(image).unsqueeze(0).to(next(self.clip_model.parameters()).device)
        with torch.no_grad():
            clip_feat = self.clip_model.encode_image(img_preprocessed)
        # The output shape is (1, 512) for ViT-B/16 (or (1, 768) for ViT-L/14)
        return clip_feat

    def forward(self, clip_feats, input_ids, attention_mask, labels=None):
        """
        (1) Project visual features via the projection layer.
        (2) Obtain text token embeddings from the frozen LLaMA.
        (3) Concatenate the visual token with text tokens.
        (4) Forward pass through LLaMA and compute the loss if labels are provided.
        """
        bsz = clip_feats.size(0)
        clip_feats = clip_feats.float()
        # Project CLIP features to LLaMA embedding space
        projected = self.projection_layer(clip_feats)  # (B, embed_dim)
        projected = projected.unsqueeze(1).expand(-1, self.num_image_tokens, -1).contiguous()
        
        # Obtain text embeddings from LLaMA (frozen)
        with torch.no_grad():
            text_embeds = self.llama_model.model.embed_tokens(input_ids)  # (B, seq_len, embed_dim)
        
        # Concatenate visual token and text embeddings
        cat_embeds = torch.cat([projected, text_embeds], dim=1)
        new_attention_mask = torch.cat([
            torch.ones(bsz, self.num_image_tokens, dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ], dim=1)
        
        outputs = self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            inputs_embeds=cat_embeds
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            ignore = torch.full((bsz, self.num_image_tokens), -100, dtype=labels.dtype, device=labels.device)
            expanded_labels = torch.cat([ignore, labels], dim=1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = expanded_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return type("ModelOutput", (), {"loss": loss, "logits": logits})

# -------------------- Interactive Testing Function --------------------
def interactive_test(multi_model, tokenizer, device, image_path):
    # Load and encode the cardiac image once
    image = Image.open(image_path).convert("RGB")
    clip_feats = multi_model.encode_image(image)
    multi_model.eval()
    
    print("Enter your clinical question (type 'quit' to exit):")
    while True:
        prompt = input(">> ")
        if prompt.lower().strip() == "quit":
            break
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = multi_model(clip_feats, input_ids, attention_mask)
        # Greedy decoding (for simplicity)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)
        generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        print("Generated Answer:", generated_text)

# -------------------- Main Testing Script --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model (ViT-B/16)
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    # Load LLaMA model (3.2-1B) with your Hugging Face token
    llama_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(llama_name, use_auth_token="YOUR_HF_TOKEN")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llama_model = AutoModelForCausalLM.from_pretrained(llama_name, use_auth_token="YOUR_HF_TOKEN").to(device)
    llama_model.eval()

    # Determine embedding dimensions
    embed_dim = llama_model.model.embed_tokens.weight.shape[1]  # e.g. 2048 for LLaMA-3.2-1B
    clip_output_dim = clip_model.visual.output_dim  # e.g. 512 for ViT-B/16

    # Create projection layer configuration and instantiate the projection
    proj_config = Config(
        mm_projector_type="mlp2x_gelu",  # 2-layer MLP with GELU activations
        mm_hidden_size=clip_output_dim,
        hidden_size=embed_dim
    )
    projection_layer = build_vision_projector(proj_config).to(device)

    # Build the multimodal model
    multi_model = MultimodalModel(
        llama_model=llama_model,
        clip_model=clip_model,
        projection_layer=projection_layer,
        tokenizer=tokenizer
    ).to(device)

    # Load the trained projection layer weights
    projection_path = "projection/best_model.pth"
    multi_model.projection_layer.load_state_dict(torch.load(projection_path, map_location=device))
    print("Projection layer weights loaded.")

    # Set the cardiac image path (update with your actual image file)
    image_path = r".\projection\57\SAMPLE_SUBJECT\lax_2c_rand.png"

    # Run interactive test
    interactive_test(multi_model, tokenizer, device, image_path)

if __name__ == "__main__":
    main()
