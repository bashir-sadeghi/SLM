# wikitext2_rnn_lm.py
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from datasets import load_dataset  # HuggingFace datasets


# -------------------------------
# 1. Simple dataset utilities
# -------------------------------

def build_vocab(tokens, vocab_size=10000):
    """
    Build a word-level vocab from tokens.
    Returns: stoi (str->int), itos (list of str)
    """
    counter = Counter(tokens)
    # reserve 0 for <unk>
    most_common = counter.most_common(vocab_size - 1)
    itos = ["<unk>"] + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def tokenize(text):
    # ultra-simple whitespace tokenizer
    return text.strip().split()


def encode(tokens, stoi):
    unk_id = stoi["<unk>"]
    return [stoi.get(t, unk_id) for t in tokens]


class LanguageModelingDataset(Dataset):
    """
    From a long list of token IDs, create fixed-length
    (x, y) sequences for next-token prediction.

    x: [block_size]
    y: [block_size]
    """
    def __init__(self, token_ids, block_size):
        self.block_size = block_size
        # we need at least block_size + 1 tokens to create one (x,y)
        total_len = len(token_ids)
        usable_len = (total_len - 1) // block_size * block_size
        self.token_ids = token_ids[: usable_len + 1]  # +1 for targets
        self.num_sequences = usable_len // block_size

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1:end + 1]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def build_wikitext2_datasets(block_size=32, vocab_size=10000):
    """
    Load WikiText-2 (raw) and create train/val/test datasets
    of fixed-length sequences.
    """
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Concatenate all lines for each split
    train_text = "\n".join(raw["train"]["text"])
    val_text = "\n".join(raw["validation"]["text"])
    test_text = "\n".join(raw["test"]["text"])

    # Tokenize
    train_tokens = tokenize(train_text)
    val_tokens = tokenize(val_text)
    test_tokens = tokenize(test_text)

    # Build vocab from train only
    stoi, itos = build_vocab(train_tokens, vocab_size=vocab_size)

    # Encode
    train_ids = encode(train_tokens, stoi)
    val_ids = encode(val_tokens, stoi)
    test_ids = encode(test_tokens, stoi)

    train_ds = LanguageModelingDataset(train_ids, block_size)
    val_ds = LanguageModelingDataset(val_ids, block_size)
    test_ds = LanguageModelingDataset(test_ids, block_size)

    return train_ds, val_ds, test_ds, stoi, itos


# -------------------------------
# 2. LightningModule: RNN LM
# -------------------------------

class RNNLanguageModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: [B, T] (token IDs)
        returns logits: [B, T, V]
        """
        emb = self.embed(x)           # [B, T, E]
        out, _ = self.lstm(emb)       # [B, T, H]
        logits = self.fc(out)         # [B, T, V]
        return logits

    def _step(self, batch, stage="train"):
        x, y = batch          # x: [B,T], y: [B,T]
        logits = self(x)      # [B,T,V]
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            y.view(B * T),
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, stage="test")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -------------------------------
# 3. Main: build data + train
# -------------------------------

def main():
    block_size = 32
    batch_size = 32
    vocab_size = 10000

    print("Loading WikiText-2 and building datasets...")
    train_ds, val_ds, test_ds, stoi, itos = build_wikitext2_datasets(
        block_size=block_size,
        vocab_size=vocab_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = RNNLanguageModel(vocab_size=len(itos), embed_dim=128, hidden_dim=256)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
