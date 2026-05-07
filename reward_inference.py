"""
Command-line script for reward model inference
Score texts using a trained reward model
"""
import sys
import os
import argparse
import torch
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reward_model import RewardModel


def main():
    parser = argparse.ArgumentParser(
        description="Reward Model Inference - Score texts using trained reward model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python reward_inference.py --model_path models/reward_model
  
  # Score single text
  python reward_inference.py --text "This is a good summary"
  
  # Score from file
  python reward_inference.py --input_file texts.json --output_file scores.json
  
  # Score from file (JSON array format)
  python reward_inference.py --input_file texts.json --output_file scores.json
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/reward_model",
        help="Path to trained reward model (default: models/reward_model)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to score"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="JSON file with texts to score. Can be:\n"
             "- List of strings: [\"text1\", \"text2\", ...]\n"
             "- List of dicts with 'text' key: [{\"text\": \"...\"}, ...]\n"
             "- Single string: \"text\""
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch processing for faster inference (when using input_file)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist!")
        print(f"Please train a reward model first or check the path.")
        print(f"\nTo train a reward model, run:")
        print(f"  python src/pipeline.py --step reward")
        return
    
    # Load model
    print("="*60)
    print("Reward Model Inference")
    print("="*60)
    print(f"Loading reward model from: {args.model_path}")
    print(f"Using device: {device}")
    print()
    
    try:
        reward_model = RewardModel(device=device)
        reward_model.load(args.model_path)
        reward_model.eval()
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("\nMake sure the model path contains:")
        print("  - config.json")
        print("  - model.safetensors (or model.bin)")
        print("  - tokenizer files")
        print("  - reward_head.pt")
        return
    
    results = []
    
    # Single text mode
    if args.text:
        print("Scoring single text...")
        print("-"*60)
        score = reward_model.get_reward(args.text)
        print(f"Text: {args.text}")
        print(f"Reward Score: {score:.4f}")
        print()
        results.append({"text": args.text, "reward": float(score)})
    
    # File input mode
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file '{args.input_file}' does not exist!")
            return
        
        print(f"Loading texts from: {args.input_file}")
        
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            return
        
        # Handle different input formats
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and 'text' in data[0]:
                texts = [item['text'] for item in data]
            else:
                texts = [str(item) for item in data]
        elif isinstance(data, dict) and 'text' in data:
            texts = [data['text']]
        else:
            texts = [str(data)]
        
        print(f"Found {len(texts)} text(s) to score\n")
        
        if args.batch and len(texts) > 1:
            # Batch processing
            print("Using batch processing...")
            print("-"*60)
            
            # Tokenize all texts
            encodings = reward_model.tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Get rewards for all texts at once
            with torch.no_grad():
                rewards = reward_model.forward(
                    encodings['input_ids'],
                    encodings['attention_mask']
                )
            
            # Convert to list
            scores = rewards.cpu().tolist()
            
            # Print and store results
            for i, (text, score) in enumerate(zip(texts, scores), 1):
                print(f"{i}. Score: {score:.4f} | Text: {text[:60]}...")
                results.append({"text": text, "reward": float(score)})
        else:
            # Sequential processing
            print("Scoring texts sequentially...")
            print("-"*60)
            for i, text in enumerate(texts, 1):
                score = reward_model.get_reward(text)
                print(f"{i}. Score: {score:.4f} | Text: {text[:60]}...")
                results.append({"text": text, "reward": float(score)})
        
        print()
    
    # Interactive mode
    else:
        print("Interactive mode - Enter text to score")
        print("(Type 'quit', 'exit', or 'q' to exit)")
        print("-"*60)
        print()
        
        while True:
            try:
                text = input("Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if text:
                    score = reward_model.get_reward(text)
                    print(f"Reward Score: {score:.4f}\n")
                    results.append({"text": text, "reward": float(score)})
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")
    
    # Save results
    if args.output_file and results:
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to {args.output_file}")
    
    # Summary
    if results:
        scores = [r['reward'] for r in results]
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Total texts scored: {len(results)}")
        print(f"Average score: {sum(scores)/len(scores):.4f}")
        print(f"Min score: {min(scores):.4f}")
        print(f"Max score: {max(scores):.4f}")
        print("="*60)


if __name__ == "__main__":
    main()

