
from typing import List, Tuple
import random

import torch
from torch import Tensor
from torch.backends import cuda, mps
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

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


class CustomDataset(Dataset):
    def __init__(self, data: Tensor, config: Config):
        self.data = data.to(config.device)
        self.config = config

    def __len__(self):
        return self.data.shape[0] - self.config.block_size

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        block_size = self.config.block_size
        device = self.config.device

        # generate a small batch of data of inputs x and targets y
        x = self.data[index:index+block_size]
        y = self.data[index+1:index+block_size+1]

        x, y = x.to(device), y.to(device)
        return x, y


def get_train_val_datasets(text: str, tokenizer: Tokenizer, config: Config):
    # Train and test splits
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return CustomDataset(train_data, config), CustomDataset(val_data, config)


def get_train_val_dataloaders(text: str, tokenizer: Tokenizer, config: Config):
    train_dataset, val_dataset = get_train_val_datasets(text, tokenizer, config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    return train_loader, val_loader


@torch.no_grad()
def estimate_loss(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  config: Config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        losses = torch.zeros(config.eval_iters)

        k = 0
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= config.eval_iters:
                break
            _, loss = model(x, y)
            losses[k] = loss.item()
            k += 1
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
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def train(model: nn.Module,
          text: str,
          tokenizer: Tokenizer,
          config: Config):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_loader, val_loader = get_train_val_dataloaders(text, tokenizer, config)

    loss_count = 0
    loss_sum = 0

    for batch_index, (xb, yb) in enumerate(train_loader):
        if batch_index >= config.max_iters:
            break

        # every once in a while evaluate the loss on train and val sets
        if batch_index % config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, config)
            print(f"batch {batch_index}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_count += 1
        loss_sum += loss.item()

        if batch_index % config.eval_interval == 0:
            print(f"Batch: {batch_index}, Average training loss: {loss_sum/loss_count:.4f}")
            print()


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
    tokenizer = Tokenizer(text)

    model = BigramLanguageModel(tokenizer.vocab_size)
    m = model.to(config.device)

    train(model, text, tokenizer, config)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == '__main__':
    main()
