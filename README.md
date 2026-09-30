# Small Language Models on WikiText-2

This repository contains small language-modeling experiments on WikiText-2 using
PyTorch Lightning. It includes:

- an LSTM language model baseline
- a compact causal self-attention / Transformer-style model
- a WikiText-2 data pipeline
- a GPT-2-compatible tokenizer interface with a newly trained small BPE vocabulary
- TensorBoard logging for losses, accuracy, and validation text samples

The goal is educational and experimental: to study the mechanics of tokenization,
sequence modeling, causal prediction, optimization, and training diagnostics in a
small, readable codebase.

## Repository Structure

```text
.
+-- main.py                       # Training entry point
+-- config.py                     # INI + CLI configuration parser
+-- exps/
|   +-- lstm_slm.ini              # LSTM experiment config
|   +-- transformer_slm.ini       # Transformer-style experiment config
+-- control/
|   +-- slm.py                    # PyTorch Lightning training module
+-- lab/
    +-- datasets/                 # WikiText-2 and GPT-2-style BPE data modules
    +-- losses/                   # Classification / cross-entropy loss wrappers
    +-- metrics/                  # TorchMetrics integration
    +-- models/                   # LSTM_SLM and Transformer_SLM
    +-- schedulers/               # Custom scheduler utilities
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install torch pytorch-lightning torchmetrics transformers datasets tensorboard
```

On the first run, Hugging Face `datasets` and `transformers` download WikiText-2
and the GPT-2 tokenizer files. They are cached locally by default.

## Data

The default experiments use WikiText-2:

- Dataset: `wikitext`, `wikitext-2-raw-v1`
- Tokenizer source: GPT-2 tokenizer rules
- Vocabulary: a newly trained smaller BPE vocabulary, controlled by
  `model_options["vocab_size"]`
- Sequence format: fixed-length next-token prediction blocks

The active dataset is selected in the experiment config:

```ini
dataset = GPT2
dataset_options = {"seed": 0, "ratio_train": 0.7, "ratio_test": 0.2, "block_size": 256}
```

Note: the `GPT2` data module uses the official train, validation, and test splits
from WikiText-2. The `ratio_train` and `ratio_test` fields are kept in the config
interface but are not used by that data module.

## Models

### LSTM_SLM

`LSTM_SLM` is a small recurrent language model:

- token embedding
- LSTM sequence encoder
- linear vocabulary projection
- cross-entropy next-token objective

Default config:

```bash
python main.py --config exps/lstm_slm.ini
```

### Transformer_SLM

`Transformer_SLM` is a compact causal self-attention model implemented directly
in PyTorch:

- token and positional embeddings
- single causal self-attention block
- residual connections
- layer normalization implemented with learned scale and shift parameters
- feed-forward block
- vocabulary projection

Default config:

```bash
python main.py --config exps/transformer_slm.ini
```

## Training

Run an experiment from the repository root:

```bash
python main.py --config exps/lstm_slm.ini
```

or:

```bash
python main.py --config exps/transformer_slm.ini
```

The training loop uses a PyTorch Lightning module in `control/slm.py`. The module
uses manual optimization, so optimizer stepping, gradient clipping, and scheduler
stepping are handled explicitly in `training_step` and `on_train_epoch_end`.

Useful config fields:

```ini
model_type = LSTM_SLM
control_type = RNN_SLM
loss_type = Classification
evaluation_type_1 = Accuracy

precision = 16
batch_size_train = 256
batch_size_val = 256
batch_size_test = 256

optim_method = AdamW
learning_rate = 1e-3
optim_options = {"weight_decay": 1e-4}
gradient_clip = 3.0

scheduler_method = StepLR
scheduler_options = {"step_size": 50, "gamma": 0.8}
```

The code automatically selects CUDA, Apple MPS, or CPU depending on availability.

## Logging and Results

Training outputs are written under `results/`, with subdirectories controlled by
the experiment config. Checkpoints are saved with the best validation loss.

Start TensorBoard with:

```bash
tensorboard --logdir results --port 6006
```

Then open:

[http://localhost:6006](http://localhost:6006)

TensorBoard logs include:

- `train_loss`
- `train_acc`
- `val_loss`
- `val_acc`
- `test_loss`
- `test_acc`
- validation text samples under the Text tab

The text samples show input, target, and predicted token sequences, which makes it
easier to inspect qualitative learning progress during training.

## Notes

- This is a compact research/learning codebase, not a production language model.
- The Transformer-style model is intentionally minimal and implemented from core
  tensor operations for clarity.
- The LSTM and Transformer configs share the same Lightning control module.
- The code is most useful for studying tokenizer training, next-token prediction,
  causal masking, optimizer behavior, and sequence-model diagnostics.
