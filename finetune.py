#!/usr/bin/env python3
"""
最简化快速微调脚本 - 去除所有可能的性能问题
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import clip
from PIL import Image
from tqdm import tqdm
import argparse
import re
import gc
import time

# LoRA相关导入
from peft import LoraConfig, get_peft_model, TaskType

def get_embedding_layer(model):
    """获取模型的embedding层"""
    for name, module in model.named_modules():
        if 'embed_tokens' in name:
            return module
    raise AttributeError("Could not find embedding layer")

def build_vision_projector(config):
    """创建projection layer"""
    projector_type = config.get("mm_projector_type", "mlp2x_gelu")
    hidden_size = config["hidden_size"] 
    mm_hidden_size = config["mm_hidden_size"]
    
    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)
    
    mlp_match = re.match(r"mlp(\d+)x_gelu", projector_type)
    if mlp_match:
        depth = int(mlp_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, depth):
            modules.extend([nn.GELU(), nn.Linear(hidden_size, hidden_size)])
        return nn.Sequential(*modules)
    
    return nn.Linear(mm_hidden_size, hidden_size)

class MinimalMultimodalModel(nn.Module):
    def __init__(self, llama_model, clip_model, projection_layer, tokenizer):
        super().__init__()
        self.llama_model = llama_model
        self.clip_model = clip_model  
        self.projection_layer = projection_layer
        self.tokenizer = tokenizer
        
        # 最小配置
        self.num_image_tokens_per_image = 1
        self.max_images = 3
        
        # 冻结CLIP和projection
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.projection_layer.parameters():
            param.requires_grad = False
            
        self.device = next(self.llama_model.parameters()).device
        _, self.clip_preprocess = clip.load("ViT-B/16", device='cpu')
        
        # 缓存embedding层
        self.embedding_layer = get_embedding_layer(self.llama_model)
        
        print(f"Minimal model initialized")

    def encode_images_batch_minimal(self, image_paths_list):
        """最简化的图像处理 - 无缓存，无优化"""
        batch_features = []
        device = next(self.clip_model.parameters()).device
        
        for image_paths in image_paths_list:
            sample_features = []
            
            for img_path in image_paths[:self.max_images]:
                try:
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            clip_features = self.clip_model.encode_image(image_tensor).float()
                        
                        sample_features.append(clip_features.cpu())
                        del image_tensor, clip_features
                    else:
                        sample_features.append(torch.zeros(1, self.clip_model.visual.output_dim))
                        
                except Exception:
                    sample_features.append(torch.zeros(1, self.clip_model.visual.output_dim))
            
            # 填充
            while len(sample_features) < self.max_images:
                sample_features.append(torch.zeros(1, self.clip_model.visual.output_dim))
            
            # 简单拼接
            sample_tensor = torch.cat(sample_features, dim=0)  # [3, 512]
            batch_features.append(sample_tensor)
        
        return torch.stack(batch_features).to(self.device)

    def forward(self, image_features, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        
        # 投影
        proj_features = self.projection_layer(image_features)
        
        # 文本embedding
        with torch.no_grad():
            text_embeddings = self.embedding_layer(input_ids)
        
        # 拼接
        combined_embeddings = torch.cat([proj_features, text_embeddings], dim=1)
        
        # attention mask
        total_image_tokens = proj_features.shape[1]
        image_attention = torch.ones(batch_size, total_image_tokens, 
                                   dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        # forward
        outputs = self.llama_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            use_cache=False
        )
        
        loss = None
        if labels is not None:
            image_labels = torch.full((batch_size, total_image_tokens), -100, 
                                    dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([image_labels, labels], dim=1)
            
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = combined_labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return type("ModelOutput", (), {"loss": loss})

class MinimalDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256):  # 进一步减少长度
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        image_paths = sample['images']
        conversations = sample['conversations']
        
        human_msg = None
        assistant_msg = None
        for conv in conversations:
            if conv['from'] == 'human':
                human_msg = re.sub(r'<image>', '', conv['value']).strip()
            elif conv['from'] == 'gpt':
                assistant_msg = conv['value'].strip()
        
        if not human_msg or not assistant_msg:
            raise ValueError(f"Invalid conversation in sample {idx}")
        
        prompt = f"Human: {human_msg}\n\nAssistant:"
        full_text = f"{prompt} {assistant_msg}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        
        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(prompt, add_special_tokens=False)
        prompt_length = len(prompt_encoding.input_ids)
        labels[:prompt_length] = -100
        
        return {
            'image_paths': image_paths,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def minimal_collate_fn(batch):
    return {
        'image_paths_batch': [item['image_paths'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def setup_minimal_lora(model, rank=16):  # 减小rank以加速
    """最小LoRA配置"""
    print("Setting up minimal LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank,  # alpha = rank
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # 只针对两个最重要的层
        bias="none"
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    return peft_model

def quick_test(model, dataloader, tokenizer, device):
    """快速测试函数"""
    model.eval()
    
    print(f"\n🧪 Quick Test")
    
    for batch in dataloader:
        try:
            image_features = model.encode_images_batch_minimal(batch['image_paths_batch'])
            
            prompt = "Human: Will this patient develop atrial fibrillation? A. Yes B. No\n\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            single_image_feat = image_features[0:1]
            
            with torch.no_grad():
                text_embeddings = model.embedding_layer(inputs.input_ids)
                combined_embeddings = torch.cat([single_image_feat, text_embeddings], dim=1)
                
                image_attention = torch.ones(1, single_image_feat.shape[1], 
                                           dtype=inputs.attention_mask.dtype, 
                                           device=inputs.attention_mask.device)
                combined_attention_mask = torch.cat([image_attention, inputs.attention_mask], dim=1)
                
                try:
                    outputs = model.llama_model.generate(
                        inputs_embeds=combined_embeddings,
                        attention_mask=combined_attention_mask,
                        max_new_tokens=3,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False
                    )
                    
                    generated_ids = outputs[0][combined_embeddings.shape[1]:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    print(f"Sample response: '{response}'")
                    
                except Exception as e:
                    print(f"Generation failed: {e}")
            
            break  # 只测试一个batch
            
        except Exception as e:
            print(f"Test failed: {e}")
            break
    
    model.train()

def minimal_training(model, train_dataloader, val_dataloader, num_epochs, learning_rate, output_dir):
    """最简化训练循环 - 无混合精度，无复杂优化"""
    
    device = next(model.parameters()).device
    
    # 最简单的optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    print(f"MINIMAL training - no FP16, no complex optimizations")
    print(f"Learning rate: {learning_rate}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        valid_steps = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            try:
                optimizer.zero_grad()
                
                # 不使用autocast，直接FP32
                image_features = model.encode_images_batch_minimal(batch['image_paths_batch'])
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(image_features, input_ids, attention_mask, labels)
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                # 简单梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                valid_steps += 1
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{total_train_loss/valid_steps:.4f}'
                })
                
                # 清理
                del image_features, input_ids, attention_mask, labels, outputs, loss
                
                if step % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error at step {step}: {e}")
                continue
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_train_loss / max(valid_steps, 1)
        steps_per_sec = valid_steps / epoch_time
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Speed: {steps_per_sec:.2f} steps/sec")
        
        # 快速测试
        quick_test(model, val_dataloader, model.tokenizer, device)
        
        # 保存模型
        checkpoint_path = os.path.join(output_dir, f'epoch_{epoch+1}')
        os.makedirs(checkpoint_path, exist_ok=True)
        model.llama_model.save_pretrained(checkpoint_path)
        print(f"  Model saved to {checkpoint_path}")
        
        torch.cuda.empty_cache()
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, 
                       default='./cardiac_mri_train.json')
    parser.add_argument('--val_data', type=str,
                       default='./cardiac_mri_val.json')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--projection_path', type=str, 
                       default='./projection/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./minimal_model')
    parser.add_argument('--batch_size', type=int, default=2)  # 回到小batch size
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)  # 回到保守学习率
    parser.add_argument('--max_length', type=int, default=256)  # 更短序列
    parser.add_argument('--lora_rank', type=int, default=16)  # 更小rank
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型 - 完全标准的方式
    print("Loading models (standard way)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # 使用FP32避免精度问题
        low_cpu_mem_usage=True
    ).to(device)
    
    llama_model = setup_minimal_lora(llama_model, rank=args.lora_rank)
    
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    embed_layer = get_embedding_layer(llama_model)
    embed_dim = embed_layer.embedding_dim
    clip_dim = clip_model.visual.output_dim
    
    proj_config = {
        "mm_projector_type": "mlp2x_gelu",
        "mm_hidden_size": clip_dim,
        "hidden_size": embed_dim
    }
    projection_layer = build_vision_projector(proj_config).float().to(device)
    
    if os.path.exists(args.projection_path):
        proj_state = torch.load(args.projection_path, map_location=device)
        if isinstance(proj_state, dict):
            if "model_state_dict" in proj_state:
                projection_layer.load_state_dict(proj_state["model_state_dict"], strict=False)
            else:
                projection_layer.load_state_dict(proj_state, strict=False)
        else:
            projection_layer.load_state_dict(proj_state, strict=False)
        print("✓ Projection loaded")
    
    model = MinimalMultimodalModel(
        llama_model=llama_model,
        clip_model=clip_model,
        projection_layer=projection_layer,
        tokenizer=tokenizer
    )
    
    print("Loading datasets...")
    train_dataset = MinimalDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = MinimalDataset(args.val_data, tokenizer, args.max_length)
    
    # 最简单的dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=minimal_collate_fn,
        num_workers=0,  # 不使用多进程
        pin_memory=False  # 不使用pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=minimal_collate_fn,
        num_workers=0
    )
    
    print("Starting MINIMAL training...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"NO mixed precision, NO complex optimizations")
    
    minimal_training(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    print("Minimal training completed!")

if __name__ == "__main__":
    main()