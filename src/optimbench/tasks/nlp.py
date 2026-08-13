import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimbench.tasks import Task


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, c // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, block_size)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size, dim, depth, n_head):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, dim))
        self.blocks = nn.Sequential(*[Block(dim, n_head, block_size) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        t = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb[:, :t]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text, block_size):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

    @property
    def vocab_size(self):
        return len(self.stoi)


class CharGPTTask(Task):
    higher_is_better = False
    max_steps_cap = 20000
    checkpoint_every = 200

    def __init__(self, text_path, name, dim=128, depth=4, n_head=4, block_size=128, batch_size=64):
        self.text_path = text_path
        self.name = name
        self.dim = dim
        self.depth = depth
        self.n_head = n_head
        self.block_size = block_size
        self.batch_size = batch_size

    def build_data(self, seed):
        with open(self.text_path, "r", encoding="utf-8") as f:
            text = f.read()
        n = int(len(text) * 0.9)
        train_ds = CharDataset(text[:n], self.block_size)
        val_ds = CharDataset(text[n:], self.block_size)
        self.vocab_size = train_ds.vocab_size
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def build_model(self, seed):
        torch.manual_seed(seed)
        return NanoGPT(self.vocab_size, self.block_size, self.dim, self.depth, self.n_head)

    def loss_and_metric(self, model, batch):
        x, y = batch
        logits = model(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    def evaluate(self, model, val_loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)
