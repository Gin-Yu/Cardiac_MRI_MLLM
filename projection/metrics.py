#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py

直接运行：python metrics.py
自动从脚本同目录加载 best_model.pth、pretrain_pairs_eval.npy 和 50–60 文件夹中的图片，
并计算 Exact Match Accuracy 与 Corpus BLEU。
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
import clip
from sacrebleu import corpus_bleu
from pathlib import Path

from projectionTrain import (
    Config,
    build_vision_projector,
    PretrainPairsDataset,
    MultimodalModel,
)

# —— 自动定位脚本及数据 —— 
BASE_DIR         = Path(__file__).resolve().parent
PROJ_CKPT_PATH   = BASE_DIR / "best_model.pth"
PAIRS_NPY_PATH   = BASE_DIR / "pretrain_pairs_eval.npy"

CLIP_MODEL_NAME  = "ViT-B/16"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
BATCH_SIZE       = 1
MAX_NEW_TOKENS   = 50
NUM_BEAMS        = 5
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ——————————————————————————————

def main():
    device = torch.device(DEVICE)
    print(f"[Info] Device: {device}")
    print(f"[Info] Loading projection weights: {PROJ_CKPT_PATH}")
    print(f"[Info] Loading eval pairs:       {PAIRS_NPY_PATH}")

    # 1. 加载 Llama Tokenizer & 模型
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llama = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_NAME).to(device)
    llama.eval()

    # 2. 加载 CLIP 模型
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=device)
    clip_model.eval()

    # 3. 重建并加载视觉投影层
    embed_dim = llama.config.hidden_size
    clip_dim  = clip_model.visual.output_dim
    proj_cfg  = Config(
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=clip_dim,
        hidden_size=embed_dim,
    )
    projection = build_vision_projector(proj_cfg).to(device)
    projection.load_state_dict(torch.load(PROJ_CKPT_PATH, map_location=device))
    projection.eval()

    # 4. 组装 MultimodalModel
    multi_model = MultimodalModel(
        llama_model=llama,
        clip_model=clip_model,
        projection_layer=projection,
        tokenizer=tokenizer,
    ).to(device)
    multi_model.eval()

    # 5. 构造 DataLoader
    dataset = PretrainPairsDataset(
        pairs_npy_path=str(PAIRS_NPY_PATH),
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [x[0] for x in batch],  # image_paths
            [x[1] for x in batch],  # prompts
            [x[2] for x in batch],  # targets
        ),
    )
    print(f"[Info] Loaded {len(dataset)} samples")

    # 原始路径前缀（来自 npy）
    old_prefix = "./dataset_eval"
    new_prefix = str(BASE_DIR)

    all_hyps = []
    all_refs = []
    correct  = 0
    total    = 0

    with torch.no_grad():
        for img_paths, prompts, targets in loader:
            # 修正路径到本地 projection/50–60 目录
            img_paths = [p.replace(old_prefix, new_prefix) for p in img_paths]

            # a) CLIP 特征 (float16 -> float32)
            clip_feats = multi_model.encode_images_batch(img_paths, batch_size=1).float()

            # b) 文本编码
            tok = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_ids      = tok["input_ids"]
            attention_mask = tok["attention_mask"]

            # c) 文本嵌入
            text_embeds = llama.model.embed_tokens(input_ids)

            # d) 拼接视觉+文本嵌入
            vis_embeds     = projection(clip_feats).unsqueeze(1)
            inputs_embeds  = torch.cat([vis_embeds, text_embeds], dim=1)

            # 构造 position_ids，使 RoPE 与总序列长度对齐
            batch_size, seq_len, _ = inputs_embeds.size()
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            # 扩展 attention_mask 以包含视觉 token
            vis_mask            = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            new_attention_mask  = torch.cat([vis_mask, attention_mask], dim=1)

            # e) 生成
            outputs = llama.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # f) 解码 & 统计
            for out_ids, ref in zip(outputs, targets):
                hyp = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                all_hyps.append(hyp)
                all_refs.append([ref])
                if hyp == ref.strip():
                    correct += 1
                total += 1

    # 6. 计算并打印指标
    accuracy = (correct / total * 100) if total > 0 else 0.0
    bleu     = corpus_bleu(all_hyps, all_refs).score if total > 0 else 0.0

    print("\n=== Evaluation Results ===")
    print(f"Samples Processed    : {total}")
    print(f"Exact Match Accuracy : {accuracy:.2f}%")
    print(f"Corpus BLEU Score    : {bleu:.2f}")

if __name__ == "__main__":
    main()
