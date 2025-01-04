import torch
from model import blues
from config import config
from data import encoder
import argparse
from optimize import optimize_model_for_inference  # Now this import will work
import os

# Define checkpoint directory if not in config
checkpoint_dir = getattr(config, 'checkpoint_dir', os.path.join(os.path.dirname(__file__), 'checkpoints'))

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint file integrity"""
    try:
        if not os.path.exists(checkpoint_path):
            return False, "Checkpoint file does not exist"
            
        if os.path.getsize(checkpoint_path) == 0:
            return False, "Checkpoint file is empty"
            
        # Try loading with weights_only
        try:
            torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            return True, None
        except Exception:
            return False, "Invalid checkpoint format"
    except Exception as e:
        return False, f"Error verifying checkpoint: {str(e)}"

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    # Resolve checkpoint path
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    
    # Verify checkpoint
    is_valid, error_msg = verify_checkpoint(checkpoint_path)
    if not is_valid:
        raise ValueError(f"Invalid checkpoint: {error_msg}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = blues(config, encoder)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try to find a valid backup checkpoint
        for backup in ['best_model.pt', 'latest.pt']:
            backup_path = os.path.join(checkpoint_dir, backup)
            is_valid, _ = verify_checkpoint(backup_path)
            if is_valid:
                print(f"Attempting to load backup checkpoint: {backup}")
                return load_model(backup_path)
        raise
        
    model.to(config.device)
    model.eval()  # Set to evaluation mode
    
    # Apply inference optimizations
    with optimize_model_for_inference(model):
        return model

def sample_text(model, prompt, max_length=100, temperature=0.8, top_p=0.9, top_k=50):
    """Generate text from a prompt"""
    print(f"\nPrompt: {prompt}")
    print("\nGenerating...")
    
    output = model.generate(
        prompt=prompt,
        output_len=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    
    print("\nGenerated text:")
    print("=" * 50)
    print(output)
    print("=" * 50)
    return output

def evaluate_perplexity(model, text):
    """Calculate perplexity on given text"""
    tokens = torch.tensor(encoder.encode(text), device=config.device).unsqueeze(0)
    with torch.no_grad():
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    return torch.exp(loss).item()

def main():
    parser = argparse.ArgumentParser(description='Text generation with Blues model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--eval_text', type=str, default=None, help='Text to evaluate perplexity on')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Generate text
    generated_text = sample_text(
        model,
        args.prompt,
        args.max_length,
        args.temperature,
        args.top_p,
        args.top_k
    )
    
    # Calculate perplexity if eval_text is provided
    if args.eval_text:
        perplexity = evaluate_perplexity(model, args.eval_text)
        print(f"\nPerplexity on evaluation text: {perplexity:.2f}")

if __name__ == "__main__":
    main()
