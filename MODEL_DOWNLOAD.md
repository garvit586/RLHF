# Model Download Guide

## Quick Download

Download the default models (DistilGPT2 and DistilBERT):
```bash
python download_models.py
```

## Custom Models

### Download Specific Models
```bash
# Download custom policy and reward models
python download_models.py --policy_model EleutherAI/pythia-70m --reward_model bert-base-uncased
```

### Save Models Locally
```bash
# Download and save to local directory
python download_models.py --save_local --local_dir my_models
```

## Available Small Models

### Policy Models (Causal Language Models)
- **distilgpt2** (default) - 82M parameters, ~350MB
- **gpt2** - 124M parameters, ~500MB
- **EleutherAI/pythia-70m** - 70M parameters, ~300MB
- **EleutherAI/pythia-160m** - 160M parameters, ~650MB
- **microsoft/DialoGPT-small** - 117M parameters, ~470MB

### Reward Models (Encoder Models)
- **distilbert-base-uncased** (default) - 66M parameters, ~260MB
- **bert-base-uncased** - 110M parameters, ~440MB
- **distilroberta-base** - 82M parameters, ~330MB

## Examples

### Example 1: Download Default Models
```bash
python download_models.py
```

### Example 2: Download and Save Locally
```bash
python download_models.py --save_local --local_dir downloaded_models
```

### Example 3: Download Different Models
```bash
python download_models.py \
    --policy_model EleutherAI/pythia-70m \
    --reward_model bert-base-uncased \
    --save_local
```

### Example 4: Use Custom Cache Directory
```bash
python download_models.py --cache_dir ./my_cache
```

## Using Downloaded Models

After downloading, models are automatically cached by Hugging Face. When you run the pipeline, it will use the cached models.

If you saved models locally with `--save_local`, you can use them like this:

```python
from src.pipeline import RLHFPipeline

pipeline = RLHFPipeline(
    base_model_name="./downloaded_models/policy_model",  # Local path
    reward_model_name="./downloaded_models/reward_model"
)
```

## Model Sizes

| Model | Parameters | Size (approx) | Speed |
|-------|-----------|---------------|-------|
| distilgpt2 | 82M | 350MB | Fast |
| gpt2 | 124M | 500MB | Medium |
| pythia-70m | 70M | 300MB | Fast |
| distilbert-base-uncased | 66M | 260MB | Fast |
| bert-base-uncased | 110M | 440MB | Medium |

## Troubleshooting

### Out of Memory
- Use smaller models: `distilgpt2` or `pythia-70m`
- Download models separately if needed

### Slow Download
- Models are cached after first download
- Use `--save_local` to keep a local copy

### Authentication Required
Some models may require Hugging Face login:
```bash
huggingface-cli login
```

## Notes

- Models are downloaded from [Hugging Face Hub](https://huggingface.co/)
- First download may take time depending on internet speed
- Models are cached in `~/.cache/huggingface/hub/` (or `C:\Users\<username>\.cache\huggingface\hub\` on Windows)
- Use `--save_local` to create a local backup

