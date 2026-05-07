"""
Reward Model for RLHF
Trains a model to predict human preferences between summaries
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class RewardModel(nn.Module):
    """Reward model that scores summaries based on human preferences"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reward head: maps hidden states to scalar reward
        hidden_size = self.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.to(device)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass to compute reward scores"""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token or mean pooling
        if hasattr(self.base_model.config, 'model_type') and 'bert' in self.base_model.config.model_type.lower():
            # For BERT models, use [CLS] token
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            # Mean pooling
            pooled_output = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        reward = self.reward_head(pooled_output)
        return reward.squeeze(-1)
    
    def score(self, hidden_states):
        """Compute score from hidden states (for TRL compatibility)"""
        # hidden_states: [batch_size, seq_len, hidden_size]
        # We need to pool these states. Since we don't have attention_mask here,
        # we'll assume the last token is the one to use (standard for causal LMs in TRL)
        # OR we can just use the hidden states directly if the head expects it.
        
        # However, my reward head expects [batch_size, hidden_size].
        # TRL passes `output.hidden_states[-1]` which is [batch_size, seq_len, hidden_size].
        
        # For compatibility with TRL's get_reward:
        # reward_logits = model.score(output.hidden_states[-1])
        # It expects `score` to take hidden states and return logits for each token?
        # TRL's get_reward selects the reward at the end index.
        
        # So `score` should return [batch_size, seq_len, 1] ideally?
        # Let's check TRL's get_reward again.
        # reward_logits = model.score(output.hidden_states[-1])
        # sequence_lengths = ...
        # return reward_logits[..., sequence_lengths, ...].squeeze(-1)
        
        # Yes, it picks the score at the end of the sequence.
        # So `score` must process the whole sequence.
        
        return self.reward_head(hidden_states)
    
    def get_reward(self, text: str) -> float:
        """Get reward score for a single text"""
        self.eval()
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            reward = self.forward(encoding['input_ids'], encoding['attention_mask'])
        return reward.item()
    
    def save(self, path: str):
        """Save the reward model"""
        os.makedirs(path, exist_ok=True)
        self.base_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.reward_head.state_dict(), os.path.join(path, 'reward_head.pt'))
    
    def load(self, path: str):
        """Load the reward model"""
        self.base_model = AutoModel.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.reward_head.load_state_dict(torch.load(os.path.join(path, 'reward_head.pt')))
        self.to(self.device)


def train_reward_model(
    model: RewardModel,
    train_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    save_path: Optional[str] = None
):
    """Train the reward model on preference data"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            chosen_ids = batch['chosen_input_ids'].to(model.device)
            chosen_mask = batch['chosen_attention_mask'].to(model.device)
            rejected_ids = batch['rejected_input_ids'].to(model.device)
            rejected_mask = batch['rejected_attention_mask'].to(model.device)
            
            # Get rewards for chosen and rejected summaries
            # We concatenate input + summary for context
            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)
            
            # Loss: chosen should have higher reward than rejected
            # We use ranking loss: max(0, margin - (chosen_reward - rejected_reward))
            margin = 1.0
            loss = torch.mean(torch.clamp(margin - (chosen_rewards - rejected_rewards), min=0.0))
            
            # Alternative: MSE loss with target rewards (chosen=1, rejected=0)
            # chosen_target = torch.ones_like(chosen_rewards)
            # rejected_target = torch.zeros_like(rejected_rewards)
            # loss = criterion(chosen_rewards, chosen_target) + criterion(rejected_rewards, rejected_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    if save_path:
        model.save(save_path)
        print(f"Reward model saved to {save_path}")
    
    return model

