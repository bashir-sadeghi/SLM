# main.py
 
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
 
import os
import sys
import config
import traceback
from lab.utils import misc
import lab.models as models
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import control
import lab.datasets as datasets
 
def main():

    args = config.parse_args()
    pl.seed_everything(args.manual_seed, workers=True)
    logger = TensorBoardLogger(
        save_dir=args.out_dir,
        # log_graph=True,
        name=args.project_name
    )
    dataloader = getattr(datasets, args.dataset)(args)
    model = getattr(control, args.control_type)(args, dataloader)
    if args.resume is not None:
        # import pdb; pdb.set_trace()
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
 
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, 'checkpoints'),
        filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
        monitor='val_loss',
        save_top_k=1)
 
    if torch.backends.mps.is_available():
        # Apple Silicon (single device)
        accelerator = "mps"
        devices = 1
        precision = 32
        strategy = None
        sync_bn = False
 
    elif torch.cuda.is_available() and getattr(args, "ngpu", 0) > 0:
        accelerator = "gpu"
        precision = args.precision
        devices = int(args.ngpu)
        if devices > 1:
            strategy = "ddp"
            sync_bn = True
        else:
            strategy = None
            sync_bn = False
        # keep args.precision (fp16/bf16) only for CUDA

    else:
        accelerator = "cpu"
        devices = 1
        precision = 32
        strategy = None
        sync_bn = False
 
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        sync_batchnorm=sync_bn,
        benchmark=True,
        callbacks=[checkpoint_callback],
        logger=logger,
        num_sanity_val_steps=0,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=precision,
        check_val_every_n_epoch=args.check_val_every_n_epochs
    )
 
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
 
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, dataloader)

    if args.nepochs == 0:
        trainer.test(dataloaders=dataloader.test_dataloader(), ckpt_path=args.resume)
    else:
        # trainer.test(dataloaders=dataloader.test_dataloader())
        trainer.test(dataloaders=dataloader.test_dataloader(), ckpt_path='best')
 
if __name__ == "__main__":
    misc.setup_graceful_exit()

    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    finally:
        misc.cleanup()
 