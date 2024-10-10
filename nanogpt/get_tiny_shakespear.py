#
# Author: Denis Tananaev
# Date: 09.10.2024
#
import os
import urllib.request
import torch

def get_dataset(split=0.9):
     # Get data
    current_dir = os.getcwd()
    dataset = get_tiny_shakespear(output_directory=current_dir)
    # Preprocess
    chars = sorted(list(set(dataset)))
    vocab_size = len(chars)
    # Create simple tokenizer
    s2i = {ch: idx  for  idx, ch  in enumerate(chars)}
    i2s = {idx: ch for idx, ch in enumerate(chars)}

    # encoder and decoder 
    # Google uses sentencepiece:  https://github.com/google/sentencepiece
    # OpenAI uses tiktoken: https://github.com/openai/tiktoken
    encode = lambda s: [s2i[ch]  for ch in s] # encoder: take a string and outputs list of tokens
    decode = lambda i: "".join([i2s[num] for num in i]) # decoder: gets list of tokens and outputs string

    data = torch.tensor(encode(dataset), dtype=torch.long)

    # Dataset split
    n = int(split * len(data))
    train_data =data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, encode, decode


def get_tiny_shakespear(output_directory: str, filename: str = "input.txt"):
    """Loads tiny shakespear data from web."""

    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    output_path = os.path.join(output_directory,filename )
    urllib.request.urlretrieve(url, output_path)

    with open(output_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
