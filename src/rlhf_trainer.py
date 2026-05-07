"""
RLHF Trainer using PPO (Proximal Policy Optimization)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import random
from typing import Optional
try:
    from .reward_model import RewardModel
except ImportError:
    from reward_model import RewardModel

# LengthSampler implementation (in case trl.core is not available)
try:
    from trl.core import LengthSampler
except ImportError:
    # Fallback implementation
    class LengthSampler:
        """Sample sequence lengths for generation"""
        def __init__(self, min_value: int, max_value: int, seed: int = None):
            self.min_value = min_value
            self.max_value = max_value
            if seed is not None:
                random.seed(seed)
        
        def __call__(self) -> int:
            return random.randint(self.min_value, self.max_value)


class RLHFTrainer:
    """RLHF Trainer using PPO"""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        reward_model_path: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 512
    ):
        self.device = device
        self.max_length = max_length
        self.model_name = model_name  # Store model name for PPO config
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # --- COMPATIBILITY FIXES ---
        # Ensure the wrapper exposes necessary attributes from the underlying model
        
        # 1. Gradient checkpointing
        if not hasattr(self.model, 'is_gradient_checkpointing'):
            if hasattr(self.model, 'pretrained_model') and hasattr(self.model.pretrained_model, 'is_gradient_checkpointing'):
                self.model.is_gradient_checkpointing = self.model.pretrained_model.is_gradient_checkpointing
            else:
                self.model.is_gradient_checkpointing = False
                
        # 2. Generation config
        if not hasattr(self.model, 'generation_config') or self.model.generation_config is None:
            # Try to get from the underlying pretrained_model if it exists
            if hasattr(self.model, 'pretrained_model') and hasattr(self.model.pretrained_model, 'generation_config'):
                self.model.generation_config = self.model.pretrained_model.generation_config
            else:
                # Create a default generation config
                try:
                    self.model.generation_config = GenerationConfig.from_model_config(self.model.config)
                except Exception:
                    # Fallback if config is also wrapped/hidden
                    self.model.generation_config = GenerationConfig()
                
                # Ensure token IDs are set
                if hasattr(self.model.config, 'eos_token_id'):
                    self.model.generation_config.eos_token_id = self.model.config.eos_token_id
                if hasattr(self.model.config, 'pad_token_id'):
                    self.model.generation_config.pad_token_id = self.model.config.pad_token_id
        
        # --- SPEED OPTIMIZATION FOR CPU ---
        # Limit the generation length so it doesn't freeze on CPU
        # Setting this to 50 instead of default (512) makes it 10x faster
        self.model.generation_config.max_new_tokens = 50
        self.model.generation_config.min_new_tokens = 5
        
        self.model.to(device)
        
        # Load reward model
        if reward_model_path:
            self.reward_model = RewardModel(device=device)
            self.reward_model.load(reward_model_path)
            self.reward_model.eval()
        else:
            self.reward_model = None
    
    def generate_summary(self, query: str, max_new_tokens: int = 100) -> str:
        """Generate summary for a given query"""
        self.model.eval()
        
        # Tokenize query
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input query from the output
        summary = generated_text[len(query):].strip()
        return summary
    
    def compute_reward(self, query: str, summary: str) -> float:
        """Compute reward for a summary using reward model"""
        if self.reward_model is None:
            # Simple heuristic: length-based reward (can be replaced)
            return len(summary.split()) / 100.0
        
        # Use reward model
        full_text = query + " " + summary
        reward = self.reward_model.get_reward(full_text)
        return reward
    
    def train_ppo(
        self,
        train_loader: DataLoader,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1.41e-5,
        output_dir: Optional[str] = None
    ):
        """Train using PPO"""
        
        # Extract queries from train_loader to build a Dataset
        # We need to collect all data first because PPOTrainer handles the dataloader internally
        all_queries = []
        for batch in train_loader:
            # batch['input_ids'] is [batch_size, seq_len]
            for input_ids in batch['input_ids']:
                all_queries.append(input_ids)
        
        # Custom dataset class for PPO
        class PPODataset(Dataset):
            def __init__(self, queries):
                self.queries = queries
            
            def __len__(self):
                return len(self.queries)
            
            def __getitem__(self, idx):
                return {"input_ids": self.queries[idx]}
        
        train_dataset = PPODataset(all_queries)
        
        # Data collator (simple padding)
        def collate_fn(data):
            input_ids = [d['input_ids'] for d in data]
            padded = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            return {"input_ids": padded}

        # PPO Configuration
        ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_ppo_epochs=4,
            seed=42,
            bf16=False,
            fp16=False,
        )
        
        # For value_model, we need to pass the underlying transformer model
        if hasattr(self.model, 'pretrained_model'):
            value_model = self.model.pretrained_model
        else:
            value_model = self.model
            
        # For reward_model, ensure it has the required base_model_prefix for TRL
        if self.reward_model:
            # If using our custom RewardModel, ensure it has base_model_prefix
            # Our RewardModel wraps a transformer in .base_model
            if not hasattr(self.reward_model, 'base_model_prefix'):
                # TRL expects this to find the backbone. 
                # We can point it to 'base_model' since that's our attribute name for the transformer
                self.reward_model.base_model_prefix = 'base_model'
        
        # Initialize PPO trainer
        # Note: PPOTrainer takes ownership of the model training loop
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=None,  # Can use a reference model for KL penalty
            reward_model=self.reward_model,
            train_dataset=train_dataset,
            value_model=value_model,
            data_collator=collate_fn,
        )
        
        # Run training
        print(f"\nStarting PPO training for {num_epochs} epochs (handled by PPOTrainer)...")
        # Note: PPOTrainer.train() runs for the configured number of steps/epochs in args
        # We override the total steps/epochs in config if needed, but here we rely on dataset size
        
        ppo_trainer.train()
        
        # Save model
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"\nModel saved to {output_dir}")
    
    def save(self, path: str):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load a trained model"""
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
