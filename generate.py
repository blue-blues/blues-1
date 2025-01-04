import torch
from model import blues
from config import config
from data import encoder
import argparse

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    model = blues(config, encoder)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
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
