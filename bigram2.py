
from typing import List

import torch
from torch.backends import cuda, mps
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

def get_device():
    if cuda.is_built():
        return 'cuda'

    if mps.is_built():
        return 'mps'

    return 'cpu'

class Config(object):
    def __init__(self,
                 batch_size: int,
                 block_size: int,
                 max_iters: int,
                 eval_interval: int,
                 learning_rate: float,
                 device: str,
                 eval_iters: int):
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters

def get_input_data() -> str:
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        return f.read()

class Tokenizer(object):
    def __init__(self, text: str):
        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, s: str) -> List[int]:
        # encoder: take a string, output a list of integers
        return [self.stoi[c] for c in s]

    def decode(self, indices: List[int]) -> str:
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[i] for i in indices])


class Loader(object):
    def __init__(self, config: Config, text: str):
        self.tokenizer = Tokenizer(text)

        # Train and test splits
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        n = int(0.9*len(data)) # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.config = config

    def get_tokenizer(self):
        return self.tokenizer

    # data loading
    def get_batch(self, split: str):
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device
        
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model: nn.Module):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def main():
    # hyperparameters
    config = Config(
        batch_size = 32, # how many independent sequences will we process in parallel?
        block_size = 8, # what is the maximum context length for predictions?
        max_iters = 3000,
        eval_interval = 300,
        learning_rate = 1e-2,
        device = get_device(),
        eval_iters = 200
    )

    text = get_input_data()
    loader = Loader(config, text)

    model = BigramLanguageModel(loader.get_tokenizer().vocab_size)
    m = model.to(config.device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0:
            losses = loader.estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = loader.get_batch('train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(loader.get_tokenizer().decode(m.generate(context, max_new_tokens=500)[0].tolist()))

if __name__ == '__main__':
    main()
