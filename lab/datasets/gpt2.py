import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer
import pdb

import torch

__all__ = ['GPT2']

class DataPrepare2(torch.utils.data.Dataset):
    _prepared = False
    _x_train = _y_train = None
    _x_val   = _y_val   = None
    _x_test  = _y_test  = None
    stoi = None
    itos = None
    tokenizer = None

    def __init__(self, data_dir, ratio_train, ratio_test, seed, vocab_size, block_size, train, val):

        # Prepare everything only once (first time __init__ is called)
        if not DataPrepare2._prepared:
            raw = load_dataset("wikitext", "wikitext-2-raw-v1")
            # raw = load_dataset("wikitext", "wikitext-2-v1")


            if DataPrepare2.tokenizer is None:
                # 1) base tokenizer (just for pre-tokenization rules)
                base_tok = AutoTokenizer.from_pretrained("gpt2")

                # 2) iterator over all or train WikiText lines
                def text_iterator():
                    # for split in ["train", "validation", "test"]:
                    for split in ["train"]:
                        for txt in raw[split]["text"]:
                            if txt and len(txt.strip()) > 0:
                                yield txt

                # 3) train new, smaller BPE vocab (e.g. 8000)
                DataPrepare2.tokenizer = base_tok.train_new_from_iterator(
                    text_iterator(),
                    vocab_size=vocab_size,   # <<< change this number if you want
                )
                DataPrepare2.tokenizer.pad_token = DataPrepare2.tokenizer.eos_token

            tok = DataPrepare2.tokenizer

                        
            def encode_split(text_list):
                all_ids = []
                for txt in text_list:
                    if not txt or not txt.strip():
                        continue
                    # encode this line only
                    ids = tok.encode(txt, add_special_tokens=False)
                    all_ids.extend(ids)
                    # optional: add an EOS between lines
                    # all_ids.append(tok.eos_token_id)
                return all_ids


            train_ids = encode_split(raw["train"]["text"])
            val_ids   = encode_split(raw["validation"]["text"])
            test_ids  = encode_split(raw["test"]["text"])

            # vocab (id <-> token) from tokenizer
            vocab = tok.get_vocab()          # token -> id
            DataPrepare2.stoi = vocab
            print(f"GPT2 vocab size: {len(vocab)}")
            # pdb.set_trace()
            # exit
            # DataPrepare2.itos = {i: t for t, i in vocab.items()}

            def make_xy(id_list):
                ids = torch.tensor(id_list, dtype=torch.long)
                total_len = ids.size(0)
                usable_len = ((total_len - 1) // block_size) * block_size
                ids = ids[:usable_len + 1]
                xs, ys = [], []
                for start in range(0, usable_len, block_size):
                    end = start + block_size
                    xs.append(ids[start:end])
                    ys.append(ids[start+1:end+1])
                x = torch.stack(xs, dim=0)
                y = torch.stack(ys, dim=0)
                return x, y

            DataPrepare2._x_train, DataPrepare2._y_train = make_xy(train_ids)
            DataPrepare2._x_val,   DataPrepare2._y_val   = make_xy(val_ids)
            DataPrepare2._x_test,  DataPrepare2._y_test  = make_xy(test_ids)

            DataPrepare2._prepared = True
        if train:
            self.x, self.y = DataPrepare2._x_train, DataPrepare2._y_train
        elif val:
            self.x, self.y = DataPrepare2._x_val, DataPrepare2._y_val
        else:
            self.x, self.y = DataPrepare2._x_test, DataPrepare2._y_test

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class GPT2(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.data_dir = opts.features_path
        self.ratio_train = opts.dataset_options['ratio_train']
        self.ratio_test = opts.dataset_options['ratio_test']
        self.seed = opts.dataset_options['seed']
        self.vocab_size = opts.model_options['vocab_size']
        self.block_size = opts.dataset_options['block_size']

        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        dataset = DataPrepare2(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            vocab_size=self.vocab_size,
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
        dataset = DataPrepare2(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            vocab_size=self.vocab_size,
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
        dataset = DataPrepare2(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            vocab_size=self.vocab_size,
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
        dataset = DataPrepare2(
            data_dir=self.data_dir,
            ratio_train=self.ratio_train,
            ratio_test=self.ratio_test,
            seed=self.seed,
            vocab_size=self.vocab_size,
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