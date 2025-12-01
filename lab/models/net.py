# net.py
import pdb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
# import control.sean.utils as ut

__all__ = ['RNN_SLM']

class RNN_SLM(nn.Module):
    """
    Simple RNN-based small language model.

    - Input:  x  of shape [batch_size, seq_len] with token IDs (LongTensor)
    - Output: logits of shape [batch_size, seq_len, vocab_size],
              and the final hidden state (h_n, c_n) if you want it.

    You’ll handle loss (CrossEntropy) in your training code.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.0):
        super(RNN_SLM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # token ID -> embedding vector
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)

        # RNN core (you can switch to GRU if you want)
        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)

        # map hidden states to vocab logits
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: LongTensor [batch_size, seq_len] with token IDs.
        hidden: optional (h_0, c_0) for LSTM; if None, LSTM uses zeros.

        returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden: (h_n, c_n)
        """
        # [B, T] -> [B, T, E]
        emb = self.embedding(x)

        # [B, T, E] -> [B, T, H]
        out, hidden = self.rnn(emb, hidden)

        # [B, T, H] -> [B, T, V]
        logits = self.fc_out(out)
        # For CrossEntropyLoss: [B, V, T] with targets [B, T]
        logits = logits.permute(0, 2, 1)

        return logits, hidden
