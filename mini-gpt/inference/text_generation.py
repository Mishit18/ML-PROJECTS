"""
Text generation script.

Loads trained model and generates text from prompts.
"""

import torch
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import create_tokenizer
from model.utils import set_seed, get_device


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        model_config_dict = checkpoint['config'].get('model', {})
    else:
        model_config_dict = {
            'vocab_size': 50257,
            'max_seq_len': 1024,
            'd_model': 384,
            'num_layers': 6,
            'num_heads': 6,
            'd_ff': 1536,
        }
    
    model_config = GPTConfig(**model_config_dict)
    model = GPT(model_config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {model.get_num_params():,}")
    
    return model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    use_cache: bool = True,
    device: torch.device = torch.device('cpu'),
) -> str:
    """
    Generate text from prompt.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        use_cache: Use KV caching
        device: Device to run on
    
    Returns:
        Generated text
    """
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_new_tokens} tokens...")
    print(f"Settings: temperature={temperature}, top_k={top_k}, top_p={top_p}, use_cache={use_cache}")
    print("-" * 80)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )
    
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def interactive_generation(model, tokenizer, device):
    """
    Interactive text generation loop.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer
        device: Device to run on
    """
    print("\n" + "="*80)
    print("Interactive Text Generation")
    print("="*80)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Commands:")
    print("  - 'quit': Exit")
    print("  - 'settings': Change generation settings")
    print("-"*80)
    
    settings = {
        'max_new_tokens': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'use_cache': True,
    }
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if prompt.lower() == 'settings':
            print("\nCurrent settings:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
            
            for key in settings.keys():
                new_value = input(f"New {key} (press Enter to keep current): ").strip()
                if new_value:
                    if key == 'use_cache':
                        settings[key] = new_value.lower() in ['true', '1', 'yes']
                    elif key == 'max_new_tokens' or key == 'top_k':
                        settings[key] = int(new_value)
                    else:
                        settings[key] = float(new_value)
            continue
        
        if not prompt:
            continue
        
        generated_text = generate_text(
            model,
            tokenizer,
            prompt,
            **settings,
            device=device,
        )
        
        print("\nGenerated text:")
        print(generated_text)
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained GPT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Input prompt (if not provided, enters interactive mode)')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable KV caching')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = create_tokenizer()
    
    model = load_model(args.checkpoint, device)
    
    if args.prompt:
        generated_text = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
            device=device,
        )
        
        print("\nGenerated text:")
        print(generated_text)
    else:
        interactive_generation(model, tokenizer, device)


if __name__ == '__main__':
    main()
