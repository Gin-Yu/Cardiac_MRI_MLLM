import json
import random
from sklearn.model_selection import train_test_split

def split_dataset(input_file, train_ratio=0.9):
    """Split dataset into train and validation sets"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按患者ID分组，确保同一患者的数据在同一集合中
    patient_groups = {}
    for item in data:
        patient_id = item['id'].split('_')[0]  # 提取患者ID
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(item)
    
    # 分割患者ID
    patient_ids = list(patient_groups.keys())
    train_ids, val_ids = train_test_split(
        patient_ids, 
        train_size=train_ratio, 
        random_state=42
    )
    
    # 构建训练集和验证集
    train_data = []
    val_data = []
    
    for pid in train_ids:
        train_data.extend(patient_groups[pid])
    
    for pid in val_ids:
        val_data.extend(patient_groups[pid])
    
    # 保存
    with open('cardiac_mri_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('cardiac_mri_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    return train_data, val_data

# 使用
split_dataset('cardiac_mri_train.json', train_ratio=0.9)