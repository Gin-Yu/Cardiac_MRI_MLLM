# Cardiac MRI Multimodal LLM

Selected research scripts for a CLIP-to-LLaMA cardiac MRI vision-language pipeline: image/text preprocessing, projection training and evaluation, LoRA fine-tuning, and inference. These files document an exploratory MSc research project rather than a clinical system.

The original research used UK Biobank data under controlled access. No MRI scans, patient records, paired-data JSON files, model weights, caches, virtual environments, or third-party source trees are included. Users need their own authorised data and model access and must adapt the scripts to their local environment.

The reported multimodal accuracy was 51.7% in the dissertation; this model is not suitable for clinical use. The scripts are shared as a research portfolio and have not been validated as a standalone reproducible release.

## Structure

- Root preprocessing scripts: transform source records and split datasets
- `projection/`: train and evaluate the vision-language projector
- `finetune.py`: LoRA-based model adaptation
- `llava_baseline.py` and `llava_test.py`: inference and diagnostic comparison

## Safety and data

Do not commit real medical images, patient-level metadata, access tokens, checkpoints, or full model downloads. Keep any credentials in environment variables or your existing model registry login.
