import os
import torch
from transformer import MiniTransformerLM

# config parameters(consistent with training parameters)
block_size = 128
embedding_dim = 64
n_heads = 4
n_layers = 2
hidden_dim = 128
dropout = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word2vec = {}
vec2word = {}

def load_vocab(data_path="data/input.txt"):
    """load vocabulary list"""
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return False
    
    word = {}
    with open(data_path, "r", encoding="utf-8") as f:
        full_content = f.read()
        for char in full_content:
            word[char] = word.get(char, 0) + 1
    
    sorted_items = sorted(word.items(), key=lambda item: item[1], reverse=True)
    sorted_words = dict(sorted_items)
    
    for i, ch in enumerate(sorted_words):
        word2vec[ch] = i
        vec2word[i] = ch
    
    print(f"Vocabulary loaded: {len(word2vec)} characters")
    return True

def load_model(model_path="result/best_model.pt"):
    """load trained model"""
    vocab_size = len(word2vec)
    model = MiniTransformerLM(
        vocal_size=vocab_size,
        d_model=embedding_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=block_size,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: {model_path} not found! Using untrained model.")
    
    model.to(device)
    model.eval()
    return model

def generate_text(model, start_text, max_length=500, temperature=0.8, top_k=None, top_p=None):
    """
    generate text
    params:
        model: trained model
        start_text: the text of begining
        max_length: the max length of generating text
        temperature: control randomness
        top_k: Top-K sampling
        top_p: Nucleus sampling
    """
    model.eval()
    
    # transfor begining text to token ids
    context = []
    for ch in start_text:
        if ch in word2vec:
            context.append(word2vec[ch])
        else:
            print(f"Warning: Character '{ch}' not in vocabulary, skipping...")
    
    if len(context) == 0:
        print("Error: No valid characters in start_text!")
        return start_text
    
    print(f"Generating with temperature={temperature}, top_k={top_k}, top_p={top_p}...")
    print("=" * 60)
    print(start_text, end="", flush=True)
    
    with torch.no_grad():
        for i in range(max_length):
            context_window = context[-block_size:]
            context_tensor = torch.tensor(context_window, dtype=torch.long).unsqueeze(0).to(device)
            
            logits = model(context_tensor)
            logits = logits[0, -1, :] / temperature
            
            # Top-K
            if top_k is not None:
                top_k_vals, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered[top_k_indices] = top_k_vals
                logits = logits_filtered
            
            # Top-P (Nucleus)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # calculate probability and sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # add to context
            context.append(next_token)
            
            print(vec2word[next_token], end="", flush=True)
    
    print("\n" + "=" * 60)
    
    generated = ''.join([vec2word[idx] for idx in context])
    return generated

def interactive_generation(model):
    print("\n" + "=" * 60)
    print("Interactive Text Generation Mode")
    print("=" * 60)
    print("Commands:")
    print("  - Enter text to use as prompt")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'config' to adjust parameters")
    print("=" * 60 + "\n")
    
    temperature = 0.8
    max_length = 300
    top_k = None
    top_p = None
    
    while True:
        start_text = input("\nEnter prompt (or command): ").strip()
        
        if start_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if start_text.lower() == 'config':
            try:
                temperature = float(input(f"Temperature [{temperature}]: ") or temperature)
                max_length = int(input(f"Max length [{max_length}]: ") or max_length)
                top_k_input = input(f"Top-K (None for off) [{top_k}]: ").strip()
                top_k = int(top_k_input) if top_k_input and top_k_input.lower() != 'none' else None
                top_p_input = input(f"Top-P (None for off) [{top_p}]: ").strip()
                top_p = float(top_p_input) if top_p_input and top_p_input.lower() != 'none' else None
                print(f"\nConfig updated: temp={temperature}, len={max_length}, top_k={top_k}, top_p={top_p}")
            except ValueError:
                print("Invalid input, keeping previous config.")
            continue
        
        if not start_text:
            print("Please enter a prompt!")
            continue
        
        generate_text(model, start_text, max_length, temperature, top_k, top_p)

def main():
    print("=" * 60)
    print("Shakespeare Text Generator")
    print("=" * 60)
    
    if not load_vocab():
        return
    
    model = load_model()
    
    examples = [
        ("ROMEO:\n", 300, 0.8),
        ("JULIET:\n", 300, 0.8),
        ("First Citizen:\n", 200, 0.7),
        ("KING:\n", 250, 0.9),
    ]
    
    print("\n" + "=" * 60)
    print("Running example generations...")
    print("=" * 60)
    
    # running examples
    for start_text, length, temp in examples:
        print(f"\n### Example: '{start_text.strip()}' (temp={temp}, len={length}) ###")
        generate_text(model, start_text, max_length=length, temperature=temp)
        print()
    
    # enter interactive mode
    interactive_generation(model)

if __name__ == "__main__":
    main()