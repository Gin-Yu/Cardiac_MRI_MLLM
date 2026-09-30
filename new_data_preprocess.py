import os
import json
import glob
from pathlib import Path
import shutil

def extract_id_from_path(image_path):
    """Extract patient ID from image path."""
    if isinstance(image_path, str) and "/" in image_path:
        parts = image_path.split("/")
        if len(parts) >= 2:
            return parts[-1]
    return None

def find_images_for_id(patient_id, dataset_dir):
    """Find all images for a given patient ID (2c, 3c, 4c views)."""
    search_pattern = os.path.join(dataset_dir, "**", patient_id, "*.png")
    all_imgs = glob.glob(search_pattern, recursive=True)
    # Keep only 2c, 3c, 4c
    imgs = [p for p in all_imgs if any(view in os.path.basename(p) for view in ["2c", "3c", "4c"])]
    # Sort by view order
    imgs.sort(key=lambda x: (0 if "2c" in x else 1 if "3c" in x else 2))
    return imgs

def process_dataset(original_json_path, dataset_dir, output_json_path, use_single_image_token=True):
    """Process dataset, filtering IDs by prefix (40-60), building new JSON."""
    with open(original_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    processed_data = []
    missing_count = 0
    processed_count = 0

    for item in original_data:
        # ID extraction
        if 'image' in item:
            pid = extract_id_from_path(item['image'])
        else:
            pid = item.get('id')
        if not pid or not str(pid)[:2].isdigit():
            continue
        prefix = int(str(pid)[:2])
        # filter out prefixes outside 40-60
        if prefix < 40 or prefix > 60:
            continue

        imgs = find_images_for_id(pid, dataset_dir)
        if not imgs:
            missing_count += 1
            continue

        # adjust conversation tokens
        new_convs = []
        for conv in item['conversations']:
            if conv['from'] == 'human':
                val = conv['value']
                if use_single_image_token:
                    # ensure exactly one <image>
                    parts = val.split('<image>')
                    base = ''.join(parts).strip()
                    # reinsert single token before Options if present
                    if "Options:" in base:
                        pre, post = base.split('Options:', 1)
                        val = f"{pre.strip()}\n<image>\nOptions:{post}"
                    else:
                        val = f"{base}\n<image>"
                else:
                    # original multi-token logic (unchanged)
                    count = val.count('<image>')
                    num_tokens = min(len(imgs), 3)
                    if count == 0:
                        if "Options:" in val:
                            pre, post = val.split('Options:', 1)
                            val = f"{pre.strip()}\n" + '<image>' * num_tokens + f"\nOptions:{post}"
                        else:
                            val = val + "\n" + '<image>' * num_tokens
                    elif count == 1 and len(imgs) > 1:
                        val = val.replace('<image>', '<image>' * num_tokens)
                new_convs.append({'from': 'human', 'value': val})
            else:
                new_convs.append(conv)

        processed_data.append({'id': pid, 'images': imgs[:3], 'conversations': new_convs})
        processed_count += 1

    # save and summary
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Processing complete! Processed items: {processed_count}, Missing images: {missing_count}")
    print(f"Output saved to: {output_json_path}")
    return processed_data

if __name__ == "__main__":
    orig = "finetune.json"
    data_dir = r".\dataset"
    out_json = r".\processed_finetune.json"
    processed = process_dataset(orig, data_dir, out_json, use_single_image_token=True)
    # train/val split omitted for brevity if not needed
