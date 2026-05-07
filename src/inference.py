"""
Inference script for generating summaries with trained RLHF model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json
from tqdm import tqdm


class RLHFInference:
    """Inference class for RLHF model"""
    
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 512):
        self.device = device
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def generate_summary(self, query: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate summary for a given query"""
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
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input query from the output
        summary = generated_text[len(query):].strip()
        return summary
    
    def batch_generate(self, queries: List[str], max_new_tokens: int = 100) -> List[str]:
        """Generate summaries for multiple queries"""
        summaries = []
        for query in tqdm(queries, desc="Generating summaries"):
            summary = self.generate_summary(query, max_new_tokens)
            summaries.append(summary)
        return summaries
    
    def evaluate_on_dataset(self, dataset_path: str, output_path: str = None):
        """Evaluate model on a dataset and save results"""
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in tqdm(data, desc="Evaluating"):
            query = item['input_text'].replace('[summary]:', '').strip()
            summary = self.generate_summary(query)
            
            result = {
                'input_text': query,
                'generated_summary': summary
            }
            results.append(result)
        
        # Save results
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", type=str, default="outputs/inference_results.json", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = RLHFInference(args.model_path, device=args.device)
    
    # Evaluate
    results = inference.evaluate_on_dataset(args.dataset_path, args.output_path)
    
    print(f"\nGenerated {len(results)} summaries")
    print("\nSample results:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Input: {result['input_text'][:100]}...")
        print(f"   Summary: {result['generated_summary']}")


if __name__ == "__main__":
    main()

