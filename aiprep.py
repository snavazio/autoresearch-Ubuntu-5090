import os
import sys
import numpy as np

def aiprep(input_file, train_ratio=0.9):
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return

    print(f"📖 Reading {input_file} for Thing 2...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    # Create a character-level vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size} unique characters")

    # Mapping characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    print(f"🔢 Tokenizing {input_file}...")
    tokens = [stoi[c] for c in data]
    tokens_np = np.array(tokens, dtype=np.uint16)
    
    # Split for training and validation
    n = len(tokens_np)
    train_data = tokens_np[:int(n * train_ratio)]
    val_data = tokens_np[int(n * train_ratio):]
    
    # Save the binary files
    train_data.tofile('train.bin')
    val_data.tofile('val.bin')
    
    print(f"✅ Success!")
    print(f"Total tokens: {n:,}")
    print(f"Binary files 'train.bin' and 'val.bin' are ready for your 5090.")

if __name__ == "__main__":
    # Check if a filename was provided as an argument
    if len(sys.argv) < 2:
        print("❌ Usage: uv run aiprep.py <filename.txt>")
    else:
        target_file = sys.argv[1]
        aiprep(target_file)