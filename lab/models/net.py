# net.py
import pdb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
# import control.sean.utils as ut

__all__ = ['SAEFit', 'RNN_SLM']

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

    
class SAEFit(nn.Module):

    def __init__(self, ndim, nout, r, hdl_enc, hdl_tgt, hdl_est, n_param, t1_m=0.4,t2_m=0.5):
        super(SAEFit, self).__init__()
        self.t1_m = t1_m
        self.t2_m = t2_m

        self.t1_lim = [0.0, 1.0]
        self.a1s_lim = [0.0, 10.0]
        self.a1_lim = [0.0, 10.0]
        self.k1_lim = [0.0, 20.0]
        self.r1_lim = [0.0, 200.0]

        self.t2_lim = [0.000, 1.0]
        self.a2_lim = [0.0, 10.0]
        self.k2_lim = [0.0, 20.0]
        self.r2_lim = [0.0, 200.0]

        self.bounds = [self.t1_lim, self.a1s_lim, self.a1_lim ,self.k1_lim, self.r1_lim,
                       self.t2_lim,  self.a2_lim, self.k2_lim, self.r2_lim]


        self.encoder = nn.Sequential(
            nn.Linear(ndim, 2 * hdl_enc),
            nn.ReLU(),
            nn.Linear(2 * hdl_enc, 2 * hdl_enc),
            nn.ReLU(),
            nn.Linear(2 * hdl_enc, 2 * hdl_enc),
            nn.ReLU(),
            # nn.Linear(2 * hdl_enc, 1 * hdl_enc),
            # nn.ReLU(),
            nn.Linear(2 * hdl_enc, r),
            # nn.ReLU(),
        )
        # self.encoder = nn.LSTM(ndim, r, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(r, 1 * hdl_tgt),
            nn.ReLU(),
            nn.Linear(1 * hdl_tgt, 2 * hdl_tgt),
            nn.ReLU(),
            # nn.Linear(2 * hdl_tgt, 2 * hdl_tgt),
            # nn.ReLU(),
            nn.Linear(2 * hdl_tgt, nout),
        )

        self.estimator = nn.Sequential(
            nn.Linear(r, 2 * hdl_est),
            nn.ReLU(),
            # nn.Linear(2 * hdl_est, 2 * hdl_est),
            # nn.ReLU(),
            nn.Linear(2 * hdl_est, 1 * hdl_est),
            nn.ReLU(),
            nn.Linear(1 * hdl_est, n_param),

        )

        self.classifier = nn.Sequential(
            nn.Linear(r, 2 * hdl_est),
            nn.ReLU(),
            # nn.Linear(2 * hdl_est, hdl_est),
            # nn.ReLU(),
            nn.Linear(2 * hdl_est, 2)
        )

        self.base = nn.Parameter(torch.tensor([self.t1_m, 0.5, 0.5, 0.5, 0.5, self.t2_m, 0.5, 0.5, 0.5]).unsqueeze(1))
                                                # t1      a1s   a1  k1   r1       t2      a2   k2   r2
    def boundary_conditions(self, bases, boundries):
        outs = []
        for i in range(len(boundries)):
            if bases.shape[1] == 1:
                outs.append(torch.clamp(bases[i], min=boundries[i][0], max=boundries[i][1]))
            else:
                outs.append(torch.clamp(bases[:,i], min=boundries[i][0], max=boundries[i][1]).unsqueeze(1))
                
        return outs
    
    def forward(self, x, control=False):

        z = self.encoder(x)
        out = self.decoder(z)
        var = self.estimator(z)
        clas = self.classifier(z)

        
        if control:
            [t1, a1s, a1, k1, r1, t2, a2, k2, r2] = self.boundary_conditions(self.base, self.bounds)

        else:
            [t1, a1s, a1, k1, r1, t2, a2, k2, r2] = self.boundary_conditions(var + self.base.T, self.bounds)


        # [t1, a1s, a1, k1, k2, ofs1, t2, a2, k3, k4, ofs2] = self.boundary_conditions(var, self.bounds)

        return out, clas,\
        t1, a1s, a1, k1, r1, t2, a2, k2, r2


