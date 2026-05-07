"""
Script to download models from Hugging Face Hub
Downloads both the policy model and reward model
"""
import os
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import argparse


def download_model(model_name: str, model_type: str = "auto", cache_dir: str = None):
    """
    Download a model from Hugging Face Hub
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        model_type: Type of model - "causal" (for GPT-like), "encoder" (for BERT-like), or "auto"
        cache_dir: Custom cache directory (optional)
    """
    print(f"\n{'='*60}")
    print(f"Downloading {model_type.upper()} Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        if model_type == "causal":
            print("Loading causal language model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        elif model_type == "encoder":
            print("Loading encoder model...")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        else:
            print("Auto-detecting model type...")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        
        print(f"✓ Successfully downloaded {model_name}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Model size: ~{sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {str(e)}")
        return None, None


def download_models(
    policy_model: str = "distilgpt2",
    reward_model: str = "distilbert-base-uncased",
    cache_dir: str = None,
    save_to_local: bool = True,
    local_dir: str = "models"
):
    """
    Download both policy and reward models
    
    Args:
        policy_model: Name of the policy model (causal LM)
        reward_model: Name of the reward model (encoder)
        cache_dir: Custom cache directory
        save_to_local: Whether to save models to local directory
        local_dir: Local directory to save models
    """
    print("\n" + "="*60)
    print("RLHF Model Downloader")
    print("="*60)
    
    # Download policy model
    policy_model_obj, policy_tokenizer = download_model(
        policy_model,
        model_type="causal",
        cache_dir=cache_dir
    )
    
    if save_to_local and policy_model_obj:
        local_policy_path = os.path.join(local_dir, "base_policy_model")
        os.makedirs(local_policy_path, exist_ok=True)
        print(f"\nSaving policy model to {local_policy_path}...")
        policy_model_obj.save_pretrained(local_policy_path)
        policy_tokenizer.save_pretrained(local_policy_path)
        print(f"✓ Policy model saved to {local_policy_path}")
    
    # Download reward model
    reward_model_obj, reward_tokenizer = download_model(
        reward_model,
        model_type="encoder",
        cache_dir=cache_dir
    )
    
    if save_to_local and reward_model_obj:
        local_reward_path = os.path.join(local_dir, "base_reward_model")
        os.makedirs(local_reward_path, exist_ok=True)
        print(f"\nSaving reward model to {local_reward_path}...")
        reward_model_obj.save_pretrained(local_reward_path)
        reward_tokenizer.save_pretrained(local_reward_path)
        print(f"✓ Reward model saved to {local_reward_path}")
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    
    if save_to_local:
        print(f"\nModels are available at:")
        print(f"  Policy: {os.path.join(local_dir, 'base_policy_model')}")
        print(f"  Reward: {os.path.join(local_dir, 'base_reward_model')}")
    else:
        print("\nModels are cached in Hugging Face cache directory")
        print("They will be automatically loaded when you run the pipeline")


def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--policy_model",
        type=str,
        default="distilgpt2",
        help="Policy model name (default: distilgpt2)"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="distilbert-base-uncased",
        help="Reward model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--no_save_local",
        action="store_true",
        help="Don't save models to local directory (use cache only)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="models",
        help="Local directory to save models (default: models)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom cache directory for Hugging Face"
    )
    
    args = parser.parse_args()
    
    download_models(
        policy_model=args.policy_model,
        reward_model=args.reward_model,
        cache_dir=args.cache_dir,
        save_to_local=not args.no_save_local,
        local_dir=args.local_dir
    )


if __name__ == "__main__":
    main()

