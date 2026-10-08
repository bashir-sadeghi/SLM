# net.py
import pdb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
# import control.sean.utils as ut

__all__ = ['LSTM_SLM', 'Transformer_SLM']

class LSTM_SLM(nn.Module):
    """
    Simple RNN-based small language model.

    - Input:  x  of shape [batch_size, seq_len] with token IDs (LongTensor)
    - Output: logits of shape [batch_size, seq_len, vocab_size],
              and the final hidden state (h_n, c_n) if you want it.

    You’ll handle loss (CrossEntropy) in your training code.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.0):
        super().__init__()

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

class Transformer_SLM(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, max_seq_len=256, attention_dim=128):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.eps = 1e-8

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                            embedding_dim=embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) / 100)

        self.w_q = nn.Parameter(torch.randn(embed_dim, attention_dim) / 100)
        self.w_k = nn.Parameter(torch.randn(embed_dim, attention_dim) / 100)
        self.w_v = nn.Parameter(torch.randn(embed_dim, attention_dim) / 100)
        self.w_o = nn.Parameter(torch.randn(attention_dim, embed_dim) / 100)

        self.gamma1 = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.beta1 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = 4 * embed_dim),
            nn.GELU(),
            nn.Linear(in_features = 4 * embed_dim, out_features = embed_dim)
        )

        self.gamma2 = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.beta2 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        


    def forward(self, x):
        T = x.size(1)

        # [B, T] -> [B, T, E]
        emb = self.embedding(x)
        

        # Positional embedding [B, T, E]
        z = emb + self.pos_embedding[:,:T,:]

        # [B, T, P]
        q = z @ self.w_q
        k = z @ self.w_k
        v = z @ self.w_v

        # [B, T ,T]
        scores = (q @ k.transpose(1, 2)) / (self.attention_dim ** 0.5) # [B, T, T]

        # defining the causal mask
        rows = torch.arange(T, device=q.device).reshape(1, T, 1)
        cols = torch.arange(T, device=q.device).reshape(1, 1, T)

        mask = rows >= cols                           # [1, T, T]

        masked_scores = scores.masked_fill(~mask, float("-inf"))


        a = torch.softmax(masked_scores, dim=-1)

        # [B, T, P] @ [P, E] -> [B, T, E]
        out_att = (a @ v) @ self.w_o

        residual1 = z + out_att                         # (B, T, E)
        mean1 = residual1.mean(dim=-1, keepdim=True)      # (B, T, 1)
        var1 = residual1.var(dim=-1, keepdim=True, correction=0) # (B, T, 1)

        out1_normalized = (residual1 - mean1) / torch.sqrt(var1 + self.eps) # (B, T, E)
        out1_rescaled = out1_normalized * self.gamma1 + self.beta1 # (B, T, E)

        f1 = self.feed_forward(out1_rescaled) # (B, T, E)

        residual2 = out1_rescaled + f1
        mean2 = residual2.mean(dim=-1, keepdim=True)      # (B, T, 1)
        var2 = residual2.var(dim=-1, keepdim=True, correction=0) # (B, T, 1)

        out2_normalized = (residual2 - mean2) / torch.sqrt(var2 + self.eps) # (B, T, E)
        out2_rescaled = out2_normalized * self.gamma2 + self.beta2 # (B, T, E)

        logits = self.fc_out(out2_rescaled)  # (B, T, vocab_size)
        return logits.transpose(1, 2), None # (B, vocab_size, T)

