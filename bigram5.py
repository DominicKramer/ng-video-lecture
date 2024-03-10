
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.backends import cuda, mps
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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
                 learning_rate: float,
                 num_epochs: int,
                 device: str,
                 at_num_iterations_estimate_loss: int,
                 max_iterations_per_epoch: Optional[int],
                 max_iterations_used_to_estimate_val_loss: Optional[int]):
        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.at_num_iterations_estimate_loss = at_num_iterations_estimate_loss
        self.max_iterations_per_epoch = max_iterations_per_epoch
        self.max_iterations_used_to_estimate_val_loss = max_iterations_used_to_estimate_val_loss


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


class CustomDataset(TensorDataset):
    def __init__(self, data: Tensor, config: Config):
        self.data = data.to(config.device)
        self.config = config

    def __len__(self):
        return self.data.shape[0] - self.config.block_size - 1

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        block_size = self.config.block_size
        device = self.config.device

        # generate a small batch of data of inputs x and targets y
        x = self.data[index:index+block_size]
        y = self.data[index+1:index+block_size+1]

        x, y = x.to(device), y.to(device)
        return x, y


def get_train_val_datasets(text: str,
                           tokenizer: Tokenizer,
                           config: Config) -> Tuple[TensorDataset, TensorDataset]:
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
        batch_size=config.batch_size
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size
    )
    return train_loader, val_loader


def estimate_loss(model: nn.Module,
                  loader: DataLoader,
                  config: Config):
    max_batch = config.max_iterations_used_to_estimate_val_loss
    if max_batch is None:
        return None

    model.eval()
    with torch.no_grad():
        running_loss = 0
        running_loss_count = 0
        
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= max_batch:
                break
            _, loss = model(x, y)
            running_loss += loss.item()
            running_loss_count += 1
    model.train()
    return running_loss/running_loss_count


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

    for epoch in range(config.num_epochs):
        print(f"Processing epoch {epoch + 1}")

        running_loss = 0
        running_loss_count = 0

        max_batch = config.max_iterations_per_epoch
        for batch_index, (xb, yb) in enumerate(train_loader):
            if max_batch is not None and batch_index >= max_batch:
                break

            if (batch_index + 1) % config.at_num_iterations_estimate_loss == 0:
                train_loss = running_loss/running_loss_count
                val_loss = estimate_loss(model, val_loader, config)
                print(f"Batch {batch_index + 1}: Training loss {train_loss:.4f}, Val loss {val_loss:.4f}")

            # evaluate the loss
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_count += 1


def generate(text: str, tokenizer: Tokenizer,
             model: BigramLanguageModel, config: Config,
             max_new_tokens: int):
    context = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=config.device)
    return tokenizer.decode(model.generate(context, max_new_tokens)[0].tolist())


def main():
    # hyperparameters
    config = Config(
        batch_size = 32, # how many independent sequences will we process in parallel?
        block_size = 8, # what is the maximum context length for predictions?
        learning_rate = 1e-2,
        num_epochs=2,
        device = get_device(),
        at_num_iterations_estimate_loss = 300,
        max_iterations_per_epoch = 3000,
        max_iterations_used_to_estimate_val_loss = 200
    )

    text = get_input_data()
    tokenizer = Tokenizer(text)

    model = BigramLanguageModel(tokenizer.vocab_size)
    model = model.to(config.device)

    train(model, text, tokenizer, config)
    print(generate("Romeo", tokenizer, model, config, 500))


if __name__ == '__main__':
    main()
