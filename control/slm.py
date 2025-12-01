'''
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 <Your Name or Organization>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-- Bashir Sadeghi
'''

import torch
import pytorch_lightning as pl
import os
import csv

import lab.models as models
import lab.losses as losses
import lab.metrics as metrics
import lab.schedulers as schedulers_custom
import matplotlib.pyplot as plt

import numpy as np

import pdb
from lab.datasets.wikitext2 import DataPrepare
from lab.datasets.gpt2 import DataPrepare2


__all__ = ['RNN_SLM']


class RNN_SLM(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.data_flag = None
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['dataloader'])


        self.opts = opts
        self.r = opts.r
        self.best_loss = 10.0
        self.n_t_samples = opts.ndim

        # For optional text logging (decode ids -> tokens)
        self.itos = getattr(DataPrepare, "itos", None)
        ## Logging Separately in CSV
        # self.loss_log_file = 'epoch_metrics_train.csv'
        self.csv_initialized = False  # flag to prevent reinitializing header each epoch

        #################################################################################
        #                                  Models                                       #
        #################################################################################
        # opts.model_type should be something like "RNN_SLM" from lab.models
        self.model = getattr(models, opts.model_type)(**opts.model_options)

        #################################################################################
        #                          Losses and Accuracies                                #
        #################################################################################
        self.criterion = {}

        self.criterion['train_lossc'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_lossc']   = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_lossc']  = getattr(losses, opts.loss_type)(**opts.loss_options)

        self.acc_trn  = getattr(metrics, opts.evaluation_type_1)(**opts.evaluation_options_1)
        self.acc_val  = getattr(metrics, opts.evaluation_type_1)(**opts.evaluation_options_1)
        self.acc_test = getattr(metrics, opts.evaluation_type_1)(**opts.evaluation_options_1)

    ################################################# Training ########################################
    def training_step(self, batch, batch_idx):
        x, y = batch                      # [B, T]
        y_hat, _ = self.model(x)         # [B, T, V]
        # pdb.set_trace()
        # print(f"x shape: {x.shape}, y shape: {y.shape}, y_hat shape: {y_hat.shape}")
        loss = self.criterion['train_lossc'](y_hat, y)

        self.acc_trn.update(y_hat, y)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)

        if self.opts.gradient_clip > 0.0:
            self.clip_gradients(
            opt,
            gradient_clip_val=self.opts.gradient_clip,
            gradient_clip_algorithm="norm",
            )

        opt.step()

        return loss

    def on_train_epoch_end(self):
        # Step scheduler (manual opt)
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        loss = self.criterion['train_lossc'].compute()
        acc = self.acc_trn.compute() * 100

        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        self.log('train_acc', acc, on_epoch=True, prog_bar=False)

        self.criterion['train_lossc'].reset()
        self.acc_trn.reset()

    ################################ Validation #######################################################################
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)   # [B, V, T] in your current model

        loss = self.criterion['val_lossc'](y_hat, y)
        self.acc_val.update(y_hat, y)

        # Only log for first batch of each val epoch
        if (
            batch_idx == 0
            and self.logger is not None
            and hasattr(self.logger, "experiment")
        ):
            print(">>> logging val_sample text")  # debug check

            with torch.no_grad():
                B = x.size(0)

                tok = DataPrepare2.tokenizer
                V = tok.vocab_size if tok is not None else None

                # detect [B, V, T] vs [B, T, V]
                if y_hat.dim() == 3 and V is not None and y_hat.shape[1] == V:
                    # [B, V, T]
                    pred_ids_full = torch.argmax(y_hat, dim=1)  # [B, T]
                else:
                    # assume [B, T, V]
                    pred_ids_full = torch.argmax(y_hat, dim=-1)  # [B, T]

                def decode(ids):
                    tok = DataPrepare2.tokenizer
                    if tok is None:
                        return " ".join(str(i) for i in ids)
                    return tok.decode(ids)

                max_examples = min(3, B)
                blocks = []
                for i in range(max_examples):
                    inp_ids  = x[i].detach().cpu().tolist()
                    tgt_ids  = y[i].detach().cpu().tolist()
                    pred_ids = pred_ids_full[i].detach().cpu().tolist()

                    input_text  = decode(inp_ids)
                    target_text = decode(tgt_ids)
                    pred_text   = decode(pred_ids)

                    blocks.append(
                        f"#### Example {i}\n"
                        f"Input:  {input_text}\n\n"
                        f"Target: {target_text}\n\n"
                        f"Pred:   {pred_text}"
                    )

                text_blob = "\n\n".join(blocks)

                tb = self.logger.experiment  # SummaryWriter
                tb.add_text(
                    "val_sample",
                    text_blob,
                    global_step=self.current_epoch,
                )

        return loss

    def on_validation_epoch_end(self):
        loss = self.criterion['val_lossc'].compute()
        acc = self.acc_val.compute() * 100

        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        self.criterion['val_lossc'].reset()
        self.acc_val.reset()

        # If you want a generic "val_loss" key as well:
        self.log('val_loss', loss, on_epoch=True, prog_bar=False)

    ################################ Testing ##########################################################################
    def test_step(self, batch, batch_idx):
        x, y = batch                         # [B, T]
        y_hat, _ = self.model(x)             # [B, T, V]

        loss = self.criterion['test_lossc'](y_hat, y)

        self.acc_test.update(y_hat, y)

        return loss

    def on_test_epoch_end(self):
        loss = self.criterion['test_lossc'].compute()
        acc = self.acc_test.compute() * 100

        self.log('test_loss', loss, on_epoch=True, prog_bar=False)
        self.log('test_acc', acc, on_epoch=True, prog_bar=False)

        self.criterion['test_lossc'].reset()
        self.acc_test.reset()

    ################################# Optimization #############################
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options
        )

        if self.opts.scheduler_method == "CustomMultiGammaLR":
            scheduler_gen = schedulers_custom.CustomMultiGammaLR(
                optimizer,
                **self.opts.scheduler_options
            )
            return [optimizer], [scheduler_gen]
        elif self.opts.scheduler_method is not None:
            scheduler_gen = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer, **self.opts.scheduler_options
            )
            return [optimizer], [scheduler_gen]
        else:
            return [optimizer]
