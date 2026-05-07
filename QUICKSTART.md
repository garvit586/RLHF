# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Download models beforehand
python download_models.py

# Test setup
python test_setup.py
```

## Quick Run

### Option 1: Simple Script
```bash
python run_pipeline.py
```

### Option 2: Command Line
```bash
# Full pipeline
python src/pipeline.py --step all

# Or individual steps
python src/pipeline.py --step reward    # Train reward model
python src/pipeline.py --step rlhf      # Train RLHF model
python src/pipeline.py --step inference # Run inference
```

### Option 3: Reward Model Inference Only
```bash
# Interactive mode
python reward_inference.py

# Score single text
python reward_inference.py --text "Your text here"

# Score from file
python reward_inference.py --input_file texts.json --output_file scores.json
```

### Option 3: Python API
```python
from src.pipeline import RLHFPipeline

pipeline = RLHFPipeline()
pipeline.run_full_pipeline(
    preference_data_path="dataset/sample_preference.json",
    prompt_data_path="dataset/sample_prompt.json"
)
```

## What Gets Created

After running the pipeline:

```
models/
├── reward_model/          # Trained reward model
└── rlhf_model/            # Trained RLHF model

outputs/
└── inference_results.json # Generated summaries
```

## Customization

Edit `config.yaml` or use command-line arguments to customize:
- Model sizes (gpt2, distilgpt2, etc.)
- Training epochs
- Batch sizes
- Learning rates

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 2`
- Use smaller models: `--base_model distilgpt2`

**Slow Training?**
- Use GPU: `--device cuda`
- Reduce epochs for testing

**Need Help?**
- Check `README.md` for detailed documentation
- Run `python test_setup.py` to verify setup

