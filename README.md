
# RNN Language Model on WikiText-2

This project trains a tiny recurrent language model (LSTM) on the WikiText-2 dataset using PyTorch Lightning and a custom BPE tokenizer (trained from GPT-2’s tokenizer).

The goal is **educational**: understand tokenization, embeddings, RNN LMs, and training mechanics – not to be state-of-the-art.

---

## 1. Setup

### 1.1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install --upgrade pip
````

### 1.2. Install dependencies

```bash
pip install \
  torch \
  pytorch-lightning \
  transformers \
  datasets \
  tensorboard
```

---

## 2. Data

We use HuggingFace’s **WikiText-2**:

* Loaded automatically via `datasets.load_dataset("wikitext", "wikitext-2-v1")`
* No manual download needed; it will be cached in `~/.cache/huggingface` by default.

There are two DataModules in the code:

* `Wikitext2`: simple word-level tokenizer.
* `GPT2`: BPE tokenizer trained on WikiText-2 with a smaller vocab (e.g. 4k).

In the config we use:

```ini
dataset = GPT2
```

so the BPE version is active by default.

---

## 3. Configuration

Training is controlled via an INI config (e.g. `exps/slm.ini`) with an `[Arguments]` section.
A typical setup (the one used for the best run) looks like:

```ini
[Arguments]

project_name   = RNN_SLM
result_subdir  = RNN_SLM-Wikitext2

model_type = RNN_SLM
model_options = {
    "vocab_size": 4000,
    "embed_dim": 256,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1
}

; dataset = Wikitext2
dataset = GPT2
; dataset = Pulse_Actual_Double

dataset_options = {
    "seed": 0,
    "ratio_train": 0.7,
    "ratio_test": 0.2,
    "block_size": 256
}

features_path = ./data
ndim = 130

precision = 16
batch_size_train = 256
batch_size_val   = 256
batch_size_test  = 256

# EarlyStopping
EarlyStopping = True
earlystop_options = {
    "monitor": "val_loss",
    "min_delta": 0.001,
    "patience": 7,
    "verbose": 1,
    "mode": "min"
}

lam = 0.01

loss_type = Classification

evaluation_type_1   = Accuracy
evaluation_options_1 = {"task": "multiclass", "num_classes": 4000}

control_type = RNN_SLM

manual_seed = 0
nepochs     = 500
init_epochs = 15
check_val_every_n_epochs = 2

# Optimization
optim_method  = AdamW
learning_rate = 1e-3
optim_options = {"weight_decay": 1e-4}

scheduler_method  = StepLR
scheduler_options = {"step_size": 50, "gamma": 0.8}

# cpu/gpu settings
ngpu     = 1
nthreads = 1
```

Adjust paths / options as needed (especially `features_path`, `result_subdir`, and `exps/slm.ini`).

---

## 4. Training

From the repo root:

```bash
python main.py --args exps/slm.ini
```

During training you get:

* **Train / val loss & perplexity** via PyTorch Lightning logs.
* **Validation text samples** (`Input / Target / Pred`) logged to TensorBoard’s **TEXT** tab.

---

## 5. TensorBoard

### 5.1. Start TensorBoard

By default, PyTorch Lightning’s `TensorBoardLogger` saves logs under `lightning_logs/` (or whatever you configured as `save_dir`). From the repo root:

```bash
tensorboard --logdir results --port 6006
```

Then open in your browser:

* [http://localhost:6006](http://localhost:6006)

### 5.2. What you can see

* In the **SCALARS** tab:

  * `train_loss` / `val_loss` (or `train_loss_c` / `val_loss_c`)
  * Any accuracy metrics you log.
* In the **TEXT** tab:

  * A few validation examples per epoch with:

    * `Input:`
    * `Target:`
    * `Pred:`
  * This lets you visually inspect how the model’s predictions evolve over training.

---

## 6. Notes

* This project uses **manual optimization** in Lightning (`automatic_optimization = False`), so:

  * Optimizer steps, optional gradient clipping, and LR scheduler stepping are all handled inside `training_step`.
* The best run so far (2-layer LSTM, 4k BPE, `block_size = 256`, batch size 256) reaches:

  * **Validation cross-entropy ≈ 3.69**
  * **Perplexity ≈ 40** on WikiText-2
    which is very respectable for a tiny educational RNN language model.
