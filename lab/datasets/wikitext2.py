from curses import raw
import pdb
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset  # HuggingFace datasets
from collections import Counter

__all__ = ['Wikitext2']


class DataPrepare(torch.utils.data.Dataset):

    # class-level storage so all instances share vocab & encoded data
    _prepared = False
    _x_train = None
    _y_train = None
    _x_val = None
    _y_val = None
    _x_test = None
    _y_test = None
    stoi = None
    itos = None

    def __init__(self, data_dir, ratio_train, ratio_test, seed, block_size, train, val):

        # Prepare everything only once (first time __init__ is called)
        if not DataPrepare._prepared:
            # raw = load_dataset("wikitext", "wikitext-2-raw-v1")
            raw = load_dataset("wikitext", "wikitext-2-v1")

            train_text = "\n".join(raw["train"]["text"])
            val_text   = "\n".join(raw["validation"]["text"])
            test_text  = "\n".join(raw["test"]["text"])

            # Tokenize: each word is a token
            train_tokens = train_text.strip().split()
            val_tokens   = val_text.strip().split()
            test_tokens  = test_text.strip().split()

            # --- build vocab from TRAIN tokens only ---
            counter = Counter(train_tokens)
            # simple: keep all distinct tokens
            vocab_tokens = list(counter.keys())

            # make sure <unk> exists
            if "<unk>" not in vocab_tokens:
                vocab_tokens.append("<unk>")

            # assign ids 0..V-1
            DataPrepare.itos = vocab_tokens
            DataPrepare.stoi = {tok: i for i, tok in enumerate(vocab_tokens)}

            unk_id = DataPrepare.stoi["<unk>"]

            # train: all tokens are in vocab
            train_ids = [DataPrepare.stoi[t] for t in train_tokens]

            # val/test: may have unseen tokens → map to <unk>
            val_ids   = [DataPrepare.stoi.get(t, unk_id) for t in val_tokens]
            test_ids  = [DataPrepare.stoi.get(t, unk_id) for t in test_tokens]

            # --- make fixed-length (x, y) sequences for each split ---
            def make_xy(id_list):
                ids = torch.tensor(id_list, dtype=torch.long)
                total_len = ids.size(0)
                usable_len = (total_len - 1) // block_size * block_size
                ids = ids[:usable_len + 1]  # keep +1 for targets
                xs = []
                ys = []
                for start in range(0, usable_len, block_size):
                    end = start + block_size
                    x_seq = ids[start:end]          # [block_size]
                    y_seq = ids[start + 1:end + 1]  # shifted by 1
                    xs.append(x_seq)
                    ys.append(y_seq)
                x = torch.stack(xs, dim=0)  # [num_seq, block_size]
                y = torch.stack(ys, dim=0)
                return x, y

            DataPrepare._x_train, DataPrepare._y_train = make_xy(train_ids)
            DataPrepare._x_val,   DataPrepare._y_val   = make_xy(val_ids)
            DataPrepare._x_test,  DataPrepare._y_test  = make_xy(test_ids)

            DataPrepare._prepared = True
            # pdb.set_trace()
            # pdb.set_trace()  # uncomment if you want to inspect once

        # ---------------------------------------------------------
        # At this point, class-level tensors are ready.
        # Just pick the right split for this instance.
        # ---------------------------------------------------------
        if train:
            self.x = DataPrepare._x_train
            self.y = DataPrepare._y_train
        elif val:
            self.x = DataPrepare._x_val
            self.y = DataPrepare._y_val
        else:
            self.x = DataPrepare._x_test
            self.y = DataPrepare._y_test

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]   # [block_size] each, dtype long
        return x, y


class Wikitext2(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.data_dir = opts.features_path
        self.ratio_train = opts.dataset_options['ratio_train']
        self.ratio_test = opts.dataset_options['ratio_test']
        self.seed = opts.dataset_options['seed']
        self.block_size = opts.dataset_options['block_size']

        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        dataset = DataPrepare(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            block_size=self.block_size,
            train=True,
            val=False,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            # persistent_workers=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        batch_size = self.opts.batch_size_val
        dataset = DataPrepare(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            block_size=self.block_size,
            train=False,
            val=True,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            # persistent_workers=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        batch_size = self.opts.batch_size_test
        dataset = DataPrepare(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            block_size=self.block_size,
            train=False,
            val=False,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_train_dataloader(self):
        batch_size = self.opts.batch_size_test
        dataset = DataPrepare(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            block_size=self.block_size,
            train=True,
            val=False,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader