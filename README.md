# RLHF (Reinforcement Learning from Human Feedback) Pipeline

A complete implementation of RLHF for text summarization, including reward model training, PPO-based policy optimization, and inference.

## Overview

This pipeline implements the RLHF process with three main steps:

1. **Reward Model Training**: Train a model to predict human preferences between summaries
2. **RLHF Training (PPO)**: Fine-tune a language model using Proximal Policy Optimization with the reward model
3. **Inference**: Generate summaries using the trained RLHF model

## Project Structure

```
RLHF/
├── dataset/
│   ├── sample_preference.json    # Preference data (chosen/rejected pairs)
│   └── sample_prompt.json        # Prompt data for training/inference
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── reward_model.py          # Reward model implementation
│   ├── rlhf_trainer.py          # RLHF trainer with PPO
│   ├── inference.py             # Inference script
│   └── pipeline.py              # Main pipeline orchestrator
├── models/                      # Saved models (created during training)
├── outputs/                     # Output files (created during training)
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone or navigate to the project directory:
```bash
cd RLHF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download models beforehand:
```bash
python download_models.py
```
This will download the default models (DistilGPT2 and DistilBERT) from Hugging Face. See `MODEL_DOWNLOAD.md` for more options.

## Dataset Format

### Preference Dataset (`sample_preference.json`)
```json
[
  {
    "input_text": "Original text... [summary]: ",
    "candidate_0": "First candidate summary",
    "candidate_1": "Second candidate summary",
    "choice": 1
  }
]
```

### Prompt Dataset (`sample_prompt.json`)
```json
[
  {
    "input_text": "Text to summarize... [summary]: "
  }
]
```

## Usage

### Option 1: Run Full Pipeline

Run all steps sequentially:
```bash
python src/pipeline.py --step all
```

### Option 2: Run Individual Steps

#### Step 1: Train Reward Model
```bash
python src/pipeline.py --step reward \
    --preference_data dataset/sample_preference.json \
    --reward_model_dir models/reward_model \
    --reward_epochs 3 \
    --batch_size 4
```

#### Step 2: Train RLHF Model
```bash
python src/pipeline.py --step rlhf \
    --prompt_data dataset/sample_prompt.json \
    --reward_model_dir models/reward_model \
    --rlhf_model_dir models/rlhf_model \
    --rlhf_epochs 3 \
    --batch_size 4
```

#### Step 3: Run Inference
```bash
python src/pipeline.py --step inference \
    --rlhf_model_dir models/rlhf_model \
    --prompt_data dataset/sample_prompt.json \
    --inference_output outputs/inference_results.json
```

### Option 3: Use Python API

```python
from src.pipeline import RLHFPipeline

# Initialize pipeline
pipeline = RLHFPipeline(
    base_model_name="gpt2",
    reward_model_name="distilbert-base-uncased",
    device="cuda"
)

# Run full pipeline
pipeline.run_full_pipeline(
    preference_data_path="dataset/sample_preference.json",
    prompt_data_path="dataset/sample_prompt.json",
    reward_model_dir="models/reward_model",
    rlhf_model_dir="models/rlhf_model",
    inference_output="outputs/inference_results.json"
)
```

## Command Line Arguments

- `--preference_data`: Path to preference dataset (default: `dataset/sample_preference.json`)
- `--prompt_data`: Path to prompt dataset (default: `dataset/sample_prompt.json`)
- `--base_model`: Base model for policy (default: `gpt2`)
- `--reward_model`: Base model for reward model (default: `distilbert-base-uncased`)
- `--reward_model_dir`: Directory to save reward model (default: `models/reward_model`)
- `--rlhf_model_dir`: Directory to save RLHF model (default: `models/rlhf_model`)
- `--inference_output`: Path to save inference results (default: `outputs/inference_results.json`)
- `--reward_epochs`: Number of epochs for reward model training (default: 3)
- `--rlhf_epochs`: Number of epochs for RLHF training (default: 3)
- `--batch_size`: Batch size (default: 4)
- `--device`: Device to use - cuda or cpu (default: `cuda`)
- `--step`: Which step to run - reward, rlhf, inference, or all (default: `all`)

## Components

### 1. Reward Model (`src/reward_model.py`)
- Trains a model to score summaries based on human preferences
- Uses ranking loss to ensure chosen summaries score higher than rejected ones
- Can be saved and loaded for reuse

### 2. RLHF Trainer (`src/rlhf_trainer.py`)
- Implements PPO (Proximal Policy Optimization) for RLHF
- Uses the reward model to provide feedback during training
- Fine-tunes the language model to generate better summaries

### 3. Inference (`src/inference.py`)
- Generates summaries using the trained RLHF model
- Supports batch processing
- Can evaluate on datasets and save results

### 4. Data Loader (`src/data_loader.py`)
- Handles loading and preprocessing of preference and prompt datasets
- Provides PyTorch Dataset classes for easy integration

## Model Recommendations

### For Base Model (Policy):
- **Small**: `distilgpt2` (default, 82M params, fastest), `gpt2` (124M params)
- **Very Small**: `EleutherAI/pythia-70m` (70M params, very fast)
- **Medium**: `gpt2-medium`, `gpt2-large`
- **Large**: `gpt2-xl` (better quality, more memory)

**Note**: Default model is now `distilgpt2` (smaller and faster than GPT-2). See `MODEL_DOWNLOAD.md` for downloading models.

### For Reward Model:
- **Small**: `distilbert-base-uncased` (recommended for most cases)
- **Medium**: `bert-base-uncased`
- **Large**: `bert-large-uncased` (better accuracy, more memory)

## Tips

1. **Start Small**: Begin with smaller models (gpt2, distilbert) to test the pipeline
2. **Batch Size**: Adjust based on GPU memory (reduce if OOM errors occur)
3. **Epochs**: Start with 3 epochs and increase if needed
4. **Data**: More preference data = better reward model = better RLHF results
5. **GPU**: CUDA is recommended for faster training

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size`
- Use smaller models (distilgpt2, distilbert)
- Reduce `max_length` in config

### Slow Training
- Use GPU (CUDA)
- Reduce batch size if causing memory issues
- Use smaller models

### Poor Results
- Increase training epochs
- Add more preference data
- Try larger models
- Adjust learning rates

## License

This project is provided as-is for educational and research purposes.

## References

- [TRL Library](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - Training language models to follow instructions
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization

