"""
Simple script to run the RLHF pipeline
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import RLHFPipeline

def main():
    """Run the RLHF pipeline"""
    print("Starting RLHF Pipeline...")
    print("="*70)
    
    # Initialize pipeline
    pipeline = RLHFPipeline(
        base_model_name="distilgpt2",  # Using DistilGPT2 - smaller and faster than GPT-2
        reward_model_name="distilbert-base-uncased",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    # Run full pipeline
    pipeline.run_full_pipeline(
        preference_data_path="dataset/sample_preference.json",
        prompt_data_path="dataset/sample_prompt.json",
        reward_model_dir="models/reward_model",
        rlhf_model_dir="models/rlhf_model",
        inference_output="outputs/inference_results.json",
        reward_epochs=3,
        rlhf_epochs=3,
        batch_size=2  # Smaller batch size for sample data
    )

if __name__ == "__main__":
    main()

