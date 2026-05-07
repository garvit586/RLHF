"""
Main RLHF Pipeline Orchestrator
Coordinates the entire RLHF training process
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_loader import PreferenceDataset, PromptDataset
from reward_model import RewardModel, train_reward_model
from rlhf_trainer import RLHFTrainer
from inference import RLHFInference


class RLHFPipeline:
    """Complete RLHF Pipeline"""
    
    def __init__(
        self,
        base_model_name: str = "distilgpt2",
        reward_model_name: str = "distilbert-base-uncased",
        device: str = "cuda",
        max_length: int = 512
    ):
        self.base_model_name = base_model_name
        self.reward_model_name = reward_model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        
        print(f"Using device: {self.device}")
    
    def step1_train_reward_model(
        self,
        preference_data_path: str,
        output_dir: str = "models/reward_model",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5
    ):
        """Step 1: Train reward model on preference data"""
        print("\n" + "="*50)
        print("STEP 1: Training Reward Model")
        print("="*50)
        
        # Initialize reward model
        reward_model = RewardModel(model_name=self.reward_model_name, device=self.device)
        
        # Load tokenizer for data loading
        tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load preference dataset
        print(f"Loading preference data from {preference_data_path}")
        dataset = PreferenceDataset(preference_data_path, tokenizer, max_length=self.max_length)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Training reward model for {num_epochs} epochs...")
        
        # Train reward model
        train_reward_model(
            model=reward_model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_path=output_dir
        )
        
        print(f"\nReward model saved to {output_dir}")
        return reward_model
    
    def step2_train_rlhf(
        self,
        prompt_data_path: str,
        reward_model_path: str,
        output_dir: str = "models/rlhf_model",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1.41e-5
    ):
        """Step 2: Train policy model using RLHF (PPO)"""
        print("\n" + "="*50)
        print("STEP 2: Training RLHF Model (PPO)")
        print("="*50)
        
        # Initialize RLHF trainer
        trainer = RLHFTrainer(
            model_name=self.base_model_name,
            reward_model_path=reward_model_path,
            device=self.device,
            max_length=self.max_length
        )
        
        # Load prompt dataset
        print(f"Loading prompt data from {prompt_data_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = PromptDataset(prompt_data_path, tokenizer, max_length=self.max_length)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Training RLHF model for {num_epochs} epochs...")
        
        # Train using PPO
        trainer.train_ppo(
            train_loader=train_loader,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir
        )
        
        print(f"\nRLHF model saved to {output_dir}")
        return trainer
    
    def step3_inference(
        self,
        model_path: str,
        dataset_path: str,
        output_path: str = "outputs/inference_results.json"
    ):
        """Step 3: Run inference on trained model"""
        print("\n" + "="*50)
        print("STEP 3: Running Inference")
        print("="*50)
        
        # Initialize inference
        inference = RLHFInference(model_path, device=self.device, max_length=self.max_length)
        
        # Evaluate on dataset
        results = inference.evaluate_on_dataset(dataset_path, output_path)
        
        print(f"\nGenerated {len(results)} summaries")
        print(f"Results saved to {output_path}")
        
        return results
    
    def run_full_pipeline(
        self,
        preference_data_path: str,
        prompt_data_path: str,
        reward_model_dir: str = "models/reward_model",
        rlhf_model_dir: str = "models/rlhf_model",
        inference_output: str = "outputs/inference_results.json",
        reward_epochs: int = 3,
        rlhf_epochs: int = 3,
        batch_size: int = 4
    ):
        """Run the complete RLHF pipeline"""
        print("\n" + "="*70)
        print("RLHF PIPELINE - FULL RUN")
        print("="*70)
        
        # Step 1: Train reward model
        self.step1_train_reward_model(
            preference_data_path=preference_data_path,
            output_dir=reward_model_dir,
            num_epochs=reward_epochs,
            batch_size=batch_size
        )
        
        # Step 2: Train RLHF model
        self.step2_train_rlhf(
            prompt_data_path=prompt_data_path,
            reward_model_path=reward_model_dir,
            output_dir=rlhf_model_dir,
            num_epochs=rlhf_epochs,
            batch_size=batch_size
        )
        
        # Step 3: Run inference
        self.step3_inference(
            model_path=rlhf_model_dir,
            dataset_path=prompt_data_path,
            output_path=inference_output
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"Reward model: {reward_model_dir}")
        print(f"RLHF model: {rlhf_model_dir}")
        print(f"Inference results: {inference_output}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RLHF Pipeline")
    
    # Data paths
    parser.add_argument("--preference_data", type=str, default="dataset/sample_preference.json",
                       help="Path to preference dataset")
    parser.add_argument("--prompt_data", type=str, default="dataset/sample_prompt.json",
                       help="Path to prompt dataset")
    
    # Model configuration
    parser.add_argument("--base_model", type=str, default="distilgpt2",
                       help="Base model for policy (e.g., distilgpt2, gpt2, EleutherAI/pythia-70m)")
    parser.add_argument("--reward_model", type=str, default="distilbert-base-uncased",
                       help="Base model for reward model")
    
    # Output paths
    parser.add_argument("--reward_model_dir", type=str, default="models/reward_model",
                       help="Directory to save reward model")
    parser.add_argument("--rlhf_model_dir", type=str, default="models/rlhf_model",
                       help="Directory to save RLHF model")
    parser.add_argument("--inference_output", type=str, default="outputs/inference_results.json",
                       help="Path to save inference results")
    
    # Training parameters
    parser.add_argument("--reward_epochs", type=int, default=3,
                       help="Number of epochs for reward model training")
    parser.add_argument("--rlhf_epochs", type=int, default=3,
                       help="Number of epochs for RLHF training")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    # Pipeline steps
    parser.add_argument("--step", type=str, choices=["reward", "rlhf", "inference", "all"],
                       default="all", help="Which step to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RLHFPipeline(
        base_model_name=args.base_model,
        reward_model_name=args.reward_model,
        device=args.device
    )
    
    # Run selected step(s)
    if args.step == "reward" or args.step == "all":
        pipeline.step1_train_reward_model(
            preference_data_path=args.preference_data,
            output_dir=args.reward_model_dir,
            num_epochs=args.reward_epochs,
            batch_size=args.batch_size
        )
    
    if args.step == "rlhf" or args.step == "all":
        pipeline.step2_train_rlhf(
            prompt_data_path=args.prompt_data,
            reward_model_path=args.reward_model_dir,
            output_dir=args.rlhf_model_dir,
            num_epochs=args.rlhf_epochs,
            batch_size=args.batch_size
        )
    
    if args.step == "inference" or args.step == "all":
        pipeline.step3_inference(
            model_path=args.rlhf_model_dir,
            dataset_path=args.prompt_data,
            output_path=args.inference_output
        )


if __name__ == "__main__":
    main()

