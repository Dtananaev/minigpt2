#
# Author: Denis Tananaev
# Date:  09.10.2024
#
import os
import torch
from nanogpt.get_tiny_shakespear import get_tiny_shakespear



def train():
    """Train script."""


    # Get data
    current_dir = os.getcwd()
    dataset = get_tiny_shakespear(output_directory=current_dir)
    print(f"Dataset length is  {len(dataset)}")
    # Preprocess
    chars = sorted(list(set(dataset)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(f"vocab_size {vocab_size}")

    # Create simple tokenizer
    s2i = {ch: idx  for  idx, ch  in enumerate(chars)}
    i2s = {idx: ch for idx, ch in enumerate(chars)}

    # encoder and decoder 
    # Google uses sentencepiece:  https://github.com/google/sentencepiece
    # OpenAI uses tiktoken: https://github.com/openai/tiktoken
    encode = lambda s: [s2i[ch]  for ch in s] # encoder: take a string and outputs list of tokens
    decode = lambda i: "".join([i2s[num] for num in i]) # decoder: gets list of tokens and outputs string

    print(f"encode {encode('Hi all!')}")
    print(f"decode {decode(encode('Hi all!'))}")

    data = torch.tensor(encode(dataset), dtype=torch.long)

    print(f"data {data.shape}, type {data.type}")
    print(f"data {data[:1000]}")

    n = int(0.9 * len(data))
    train_data =data[:n]
    val_data = data[n:]


if __name__ =="__main__":
    train()
