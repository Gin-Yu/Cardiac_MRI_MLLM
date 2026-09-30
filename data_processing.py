import json
import os
from pathlib import Path
from tqdm import tqdm

def preprocess_medical_dataset(
    input_json_path,
    output_json_path,
    image_base_dir,
    valid_folders=range(40, 61)  # 40-60文件夹
):
    """
    Preprocess medical dataset to match image availability
    """
    # Load original dataset
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    missing_count = 0
    
    for item in tqdm(data, desc="Processing dataset"):
        # Extract patient ID from item
        patient_id = item["id"]  # e.g., "SAMPLE_SUBJECT"
        
        # Extract folder number (first 2 digits)
        folder_num = int(patient_id[:2])
        
        # Check if in valid range
        if folder_num not in valid_folders:
            missing_count += 1
            continue
        
        # Construct image paths
        image_folder = Path(image_base_dir) / str(folder_num) / patient_id
        
        # Check if all three images exist
        image_files = [
            image_folder / "lax_2c_rand.png",
            image_folder / "lax_3c_rand.png", 
            image_folder / "lax_4c_rand.png"
        ]
        
        if all(img.exists() for img in image_files):
            # Update conversation to include three images
            new_conversations = []
            
            for conv in item["conversations"]:
                if conv["from"] == "human":
                    # Replace single <image> with three <image> tags
                    new_value = conv["value"].replace(
                        "<image>",
                        "<image><image><image>"
                    )
                    new_conversations.append({
                        "from": "human",
                        "value": new_value
                    })
                else:
                    new_conversations.append(conv)
            
            # Create new item with updated paths
            new_item = {
                "id": item["id"],
                "images": [str(img) for img in image_files],  # Three image paths
                "conversations": new_conversations
            }
            
            processed_data.append(new_item)
        else:
            missing_count += 1
    
    # Save processed dataset
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(processed_data)} valid samples")
    print(f"Skipped {missing_count} samples without complete images")
    
    return len(processed_data)

# Usage
if __name__ == "__main__":
    IMAGE_BASE = r".\dataset"
    
    # Process training set
    preprocess_medical_dataset(
        input_json_path="finetune.json",
        output_json_path="cardiac_mri_train.json",
        image_base_dir=IMAGE_BASE,
        valid_folders=range(40, 61)
    )
    
    # Process validation set if exists
    if os.path.exists("finetune_val.json"):
        preprocess_medical_dataset(
            input_json_path="finetune_val.json",
            output_json_path="cardiac_mri_val.json",
            image_base_dir=IMAGE_BASE,
            valid_folders=range(40, 61)
        )