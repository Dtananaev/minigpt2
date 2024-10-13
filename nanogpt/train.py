#
# Author: Denis Tananaev
# Date:  09.10.2024
#
import os
import torch
from functools import partial
from nanogpt.get_tiny_shakespear import get_dataset
from nanogpt.gpt_model import GPTLanguageModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(data_tuple, batch_size, block_size, split):
    """Generate a small batch of data of inputs x and targets y.
    
    Args:
        data: the data list
        batch_size: size of batch
        block_size: seq length

    Returns:
        x: the inputs of the shape [batch, block_size]
        y: the targets of the shape [batch, block_size]
    """

    train, val = data_tuple
    data = train if split=="train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model, get_batch_func, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_func(split=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(batch_size = 64,
        block_size =256,
        max_iters = 5000,
        eval_interval = 500,
        learning_rate = 3e-4,
        eval_iters = 200,
        dropout=0.2, 
        n_embed=384,
        n_heads=6,
        n_layers=6):
    """Train script.
    
    Args:
        batch_size: how many independent sequences will we process in parallel?
        block_size: what is the maximum context length for predictions?
    """
    torch.manual_seed(1337)
    train_data, val_data, vocab_size, encode, decode = get_dataset(split=0.9)
    get_batch_func  = partial(get_batch, data_tuple=(train_data, val_data), batch_size=batch_size, block_size=block_size)
    model = GPTLanguageModel(vocab_size=vocab_size,block_size=block_size,  n_embd=n_embed, n_head=n_heads, n_layer=n_layers, dropout=dropout).to(DEVICE)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Training:  The initial loss should be -ln(1/vocab_size) 
    for iter in range(max_iters):


        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
           losses = estimate_loss(model, get_batch_func=get_batch_func, eval_iters=eval_iters)
           print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch_func(split="train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx  = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(idx, max_new_tokens=400)[0].tolist())) 

if __name__ =="__main__":
    train()
