#!/usr/bin/env python3
"""
完整推理测试脚本 - 与训练脚本完全匹配
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import clip
from PIL import Image
import re
import argparse

def build_vision_projector(config):
    """创建projection layer - 与训练脚本一致"""
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

def get_embedding_layer(model):
    """获取embedding层"""
    for name, module in model.named_modules():
        if 'embed_tokens' in name:
            return module
    raise ValueError("Could not find embedding layer")

class CardiacLLaVAInference:
    def __init__(
        self, 
        base_model_path="meta-llama/Llama-3.2-1B",
        lora_model_path=None,
        projection_path=None,
        device="cuda"
    ):
        """初始化推理模型"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # 与训练一致的配置
        self.num_image_tokens_per_image = 1
        self.max_images = 3
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        print("🧠 Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Load LoRA weights
        if lora_model_path and os.path.exists(lora_model_path):
            print(f"🔗 Loading LoRA from {lora_model_path}...")
            try:
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    lora_model_path,
                    torch_dtype=torch.float32
                )
                print("✅ LoRA loaded successfully!")
            except Exception as e:
                print(f"⚠️  LoRA loading failed: {e}")
                self.model = self.base_model
        else:
            print(f"⚠️  No LoRA path provided, using base model")
            self.model = self.base_model
        
        self.model.eval()
        
        # Load CLIP
        print("👁️  Loading CLIP...")
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.clip_model.eval()
        
        # 获取embedding层
        self.embed_layer = get_embedding_layer(self.model)
        embed_dim = self.embed_layer.embedding_dim
        
        # Load projection layer
        print("🔗 Loading projection layer...")
        clip_dim = self.clip_model.visual.output_dim
        proj_config = {
            "mm_projector_type": "mlp2x_gelu",
            "mm_hidden_size": clip_dim,
            "hidden_size": embed_dim
        }
        self.projection = build_vision_projector(proj_config).float().to(self.device)
        
        if projection_path and os.path.exists(projection_path):
            try:
                proj_state = torch.load(projection_path, map_location=self.device)
                if isinstance(proj_state, dict):
                    if "model_state_dict" in proj_state:
                        self.projection.load_state_dict(proj_state["model_state_dict"], strict=False)
                    else:
                        self.projection.load_state_dict(proj_state, strict=False)
                else:
                    self.projection.load_state_dict(proj_state, strict=False)
                self.projection.eval()
                print("✅ Projection loaded successfully!")
            except Exception as e:
                print(f"⚠️  Projection loading failed: {e}")
        
        print("\n" + "="*60)
        print("🏥 Cardiac MRI LLaVA Ready for Testing!")
        print("="*60)
    
    def encode_images_inference(self, image_paths):
        """编码图像 - 与训练完全一致"""
        device = next(self.clip_model.parameters()).device
        sample_features = []
        
        # 处理图像（最多3张）
        for img_path in image_paths[:self.max_images]:
            try:
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        clip_features = self.clip_model.encode_image(image_tensor).float()
                        # 确保是正确的形状 [1, 512]
                        if clip_features.dim() == 1:
                            clip_features = clip_features.unsqueeze(0)
                    
                    sample_features.append(clip_features.cpu())
                    del image_tensor, clip_features
                else:
                    # 创建零特征 [1, 512]
                    sample_features.append(torch.zeros(1, self.clip_model.visual.output_dim))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                sample_features.append(torch.zeros(1, self.clip_model.visual.output_dim))
        
        # 填充到最大数量
        while len(sample_features) < self.max_images:
            sample_features.append(torch.zeros(1, self.clip_model.visual.output_dim))
        
        # 拼接成 [3, 512]
        sample_tensor = torch.cat(sample_features, dim=0)  # [3, 512]
        
        # 转换为batch格式 [1, 3, 512]
        batch_tensor = sample_tensor.unsqueeze(0).to(self.device)
        
        # 投影到文本空间 [1, 3, 2048]
        with torch.no_grad():
            projected_features = self.projection(batch_tensor)
        
        return projected_features
    
    def generate_response(self, image_paths, question, max_new_tokens=10, temperature=0.1):
        """生成回复 - 与训练完全匹配"""
        
        print(f"🚀 Question: {question}")
        
        # 1. 编码图像
        image_features = self.encode_images_inference(image_paths)
        print(f"📊 Image features shape: {image_features.shape}")
        
        # 2. 准备文本 - 与训练时格式完全一致
        prompt = f"Human: {question}\n\nAssistant:"
        
        # 3. Tokenize文本
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 4. 获取文本embeddings
        with torch.no_grad():
            text_embeddings = self.embed_layer(inputs.input_ids)
        
        print(f"📊 Text embeddings shape: {text_embeddings.shape}")
        
        # 5. 拼接图像和文本embeddings（与训练时一致）
        combined_embeddings = torch.cat([image_features, text_embeddings], dim=1)
        
        # 6. 创建对应的attention mask
        batch_size = image_features.shape[0]
        image_attention_mask = torch.ones(batch_size, image_features.shape[1], 
                                        dtype=inputs.attention_mask.dtype, 
                                        device=inputs.attention_mask.device)
        combined_attention_mask = torch.cat([image_attention_mask, inputs.attention_mask], dim=1)
        
        print(f"📊 Combined embeddings shape: {combined_embeddings.shape}")
        
        # 7. 生成
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=combined_embeddings,
                    attention_mask=combined_attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False
                )
                
                # 8. 解码新生成的tokens
                generated_ids = outputs[0][combined_embeddings.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                print(f"✅ Generated {len(generated_ids)} tokens")
                print(f"🔍 Generated token IDs: {generated_ids.tolist()}")
                
                return response
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_multiple_questions(self, image_paths, questions):
        """测试多个问题"""
        
        print("\n" + "="*70)
        print("🧪 COMPREHENSIVE TESTING")
        print("="*70)
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*20} Test {i}/{len(questions)} {'='*20}")
            print(f"Q: {question}")
            
            try:
                response = self.generate_response(image_paths, question)
                print(f"A: '{response}'")
                
                # 分析回答
                if response:
                    answer_lower = response.lower()
                    if any(x in answer_lower for x in ['a', 'yes']):
                        print("🔴 → HIGH RISK (A/Yes)")
                    elif any(x in answer_lower for x in ['b', 'no']):
                        print("🟢 → LOW RISK (B/No)")
                    else:
                        print("❓ → UNCLEAR/OTHER")
                        print(f"🔍 Full response: '{answer_lower}'")
                else:
                    print("❌ → NO RESPONSE")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 70)
    
    def evaluate_on_examples(self, test_examples):
        """在给定的例子上评估"""
        
        print("\n" + "="*70)
        print("📊 EVALUATION ON EXAMPLES")
        print("="*70)
        
        correct = 0
        total = 0
        
        for i, example in enumerate(test_examples, 1):
            image_paths = example['images']
            question = example['question']
            expected = example['expected'].lower()
            
            print(f"\n--- Example {i}/{len(test_examples)} ---")
            print(f"Question: {question}")
            print(f"Expected: {expected}")
            
            try:
                response = self.generate_response(image_paths, question, max_new_tokens=5)
                
                if response:
                    response_lower = response.lower()
                    
                    # 简单的准确性检查
                    if expected in ['a', 'yes'] and any(x in response_lower for x in ['a', 'yes']):
                        correct += 1
                        print(f"✅ Correct! Generated: '{response}'")
                    elif expected in ['b', 'no'] and any(x in response_lower for x in ['b', 'no']):
                        correct += 1
                        print(f"✅ Correct! Generated: '{response}'")
                    else:
                        print(f"❌ Wrong! Generated: '{response}'")
                else:
                    print(f"❌ No response generated")
                
                total += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 ACCURACY: {correct}/{total} = {accuracy:.2%}")
        
        return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--lora_model_path', type=str, 
                       default='./complete_cardiac_model/best_model')
    parser.add_argument('--projection_path', type=str, 
                       default='./projection/best_model.pth')
    
    args = parser.parse_args()
    
    # 示例图像路径
    test_images = [
        r".\dataset\44\SAMPLE_SUBJECT\lax_2c_rand.png",
        r".\dataset\44\SAMPLE_SUBJECT\lax_3c_rand.png",
        r".\dataset\44\SAMPLE_SUBJECT\lax_4c_rand.png"
    ]
    
    print("🧪 Initializing Cardiac LLaVA for Testing...")
    
    try:
        model = CardiacLLaVAInference(
            base_model_path=args.model_name,
            lora_model_path=args.lora_model_path,
            projection_path=args.projection_path
        )
        
        # 基本测试问题
        basic_questions = [
            "Will this patient develop atrial fibrillation? A. Yes B. No",
            "Based on these cardiac MRI images, what is the AF risk? A. Yes B. No",
            "Does this cardiac MR indicate future atrial fibrillation development? A. Yes B. No",
            "What is the atrial fibrillation risk? A. High B. Low",
        ]
        
        # 测试基本问题
        model.test_multiple_questions(test_images, basic_questions)
        
        # 创建一些测试样例
        test_examples = [
            {
                'images': test_images,
                'question': 'Will this patient develop atrial fibrillation? A. Yes B. No',
                'expected': 'a'  # 根据你的训练数据调整
            },
            {
                'images': test_images,
                'question': 'Based on these cardiac MRI images, what is the AF risk? A. Yes B. No',
                'expected': 'a'
            }
        ]
        
        # 评估准确性
        model.evaluate_on_examples(test_examples)
        
        # 交互模式
        print(f"\n{'='*70}")
        print("🎮 INTERACTIVE MODE")
        print("Type your questions (or 'exit' to quit)")
        print("="*70)
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() == 'exit':
                    break
                    
                if not question:
                    continue
                
                response = model.generate_response(test_images, question)
                print(f"🤖 Response: '{response}'")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Testing completed!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()