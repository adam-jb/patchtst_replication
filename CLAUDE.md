# CLAUDE.md — Build PatchTST from Scratch

you are using an M1 pro cpu. Optimise for that.

outputs should be nice and graphed in html files for easy comparison to the originals and to see scaling effects.

you can update what you do in this file as its the long term memory you will use. avoid changing core instructions.

this project exists to help the user better understand the findings of the paper and expand on them/



## Goal
Replicate the PatchTST paper ("A Time Series is Worth 64 Words") from scratch in PyTorch. No HuggingFace, no tsai, no copying from the official repo. Pure PyTorch so every component is understood. Reproduce their reported MSE/MAE numbers on ETTh1 supervised forecasting as the primary validation target.

## Reference
- Paper: https://arxiv.org/abs/2211.14730
- Official repo (for comparison only, not for copying): https://github.com/yuqinie98/PatchTST
- Datasets: download from https://github.com/zhouhaoyi/Informer2020 (the `ETT-small` folder contains ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv)

## Project Structure
```
patchtst/
├── CLAUDE.md
├── data/
│   └── ETTh1.csv          # downloaded
├── src/
│   ├── dataset.py          # data loading, splitting, normalisation, patching
│   ├── model.py            # PatchTST model
│   ├── train_supervised.py # supervised training loop
│   ├── train_selfsup.py    # self-supervised pretraining + fine-tuning
│   └── utils.py            # metrics, helpers
├── scripts/
│   ├── run_supervised.sh
│   └── run_selfsup.sh
└── results/
```

## Implementation Steps (do these in order)

### Step 1: Data Pipeline (`dataset.py`)

1. Load ETTh1.csv. Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT. That's 7 channels.
2. Train/val/test split by index (standard for this dataset):
   - Train: months 0–12 (rows 0:8640)
   - Val: months 12–16 (rows 8640:11520)
   - Test: months 16–20 (rows 11520:14400)
3. Normalisation: fit StandardScaler on train split only. Transform val and test with train statistics. Per-channel normalisation.
4. Create sliding window dataset:
   - Input: `(batch, channels, seq_len)` where seq_len = L (lookback window)
   - Target: `(batch, channels, pred_len)` where pred_len = T (prediction horizon)
   - Stride of 1 between windows
   - Channel-independent: at training time, each channel is a separate sample. So one multivariate window of M=7 channels becomes 7 univariate training samples. The dataset should handle this reshaping.
5. Use PyTorch DataLoader. Batch size 32.

### Step 2: Model (`model.py`)

Build these components bottom-up:

#### 2a: Patch Embedding
- Input: `(batch, 1, seq_len)` — one univariate series
- Unfold into patches: patch_len P=16, stride S=8 (overlapping patches)
- Number of patches: N = floor((seq_len - P) / S) + 1
- Linear projection: `nn.Linear(P, D)` where D=128 (d_model)
- Add learnable positional encoding: `nn.Parameter(torch.randn(1, N, D))`
- Output: `(batch, N, D)`

#### 2b: Transformer Encoder
- Use `nn.TransformerEncoderLayer` with:
  - d_model = 128
  - nhead = 16
  - dim_feedforward = 256
  - dropout = 0.2
  - activation = 'gelu'
  - **batch_first = True**
  - **norm_first = False** (post-norm, but see note below)
- Stack 3 layers via `nn.TransformerEncoder`
- NOTE: The paper uses BatchNorm instead of LayerNorm. To replicate this, you need a custom encoder layer that replaces LayerNorm with BatchNorm1d. Implement a custom `PatchTSTEncoderLayer` that is identical to `nn.TransformerEncoderLayer` but swaps `nn.LayerNorm` for `nn.BatchNorm1d(D)`. BatchNorm1d expects `(batch, D)` or `(batch, D, seq)`, so you'll need to permute before/after the norm.

#### 2c: Prediction Head (Supervised)
- Take encoder output: `(batch, N, D)`
- Flatten: `(batch, N*D)`
- Linear projection: `nn.Linear(N*D, pred_len)` — predicts the full forecast horizon in one shot
- Output: `(batch, pred_len)`
- During forward pass, process all M channels independently (reshape batch dimension), then reshape back to `(batch_orig, M, pred_len)`

#### 2d: Full Model Forward Pass
```
input (batch, M, L)
  → reshape to (batch*M, 1, L)           # channel independence
  → patch + project + pos_enc            # (batch*M, N, D)
  → transformer encoder                  # (batch*M, N, D)
  → flatten → linear head                # (batch*M, pred_len)
  → reshape to (batch, M, pred_len)      # restore channels
```

#### 2e: Hyperparameters for ETTh1
ETTh1 is a small dataset. The paper uses reduced hyperparameters:
- H=4 (heads), D=16 (d_model), F=128 (feedforward dim)
- NOT the default H=16, D=128, F=256 — those are for larger datasets
- Still 3 encoder layers, dropout 0.2, patch_len 16, stride 8

### Step 3: Supervised Training (`train_supervised.py`)

1. Loss: MSE, averaged over all channels and all prediction timesteps
2. Optimiser: Adam, lr=1e-4, batch size 32 (supervised) / 128 (self-supervised)
3. Random seed: 2021
4. Train for 100 epochs with early stopping (patience ~10 epochs on val MSE)
5. Learning rate: the paper doesn't specify a scheduler explicitly for supervised; use constant or ReduceLROnPlateau
6. Evaluate on test set after training. Report MSE and MAE.
7. Test with prediction horizons T ∈ {96, 192, 336, 720}. Lookback L=336 for ETTh1 supervised (check paper Table 2 footnotes).

Expected results for ETTh1 supervised (from paper Table 2, PatchTST/42):
| Pred Len | MSE   | MAE   |
|----------|-------|-------|
| 96       | 0.370 | 0.400 |
| 192      | 0.413 | 0.429 |
| 336      | 0.422 | 0.440 |
| 720      | 0.447 | 0.468 |

You don't need to match these exactly. Within 5-10% is good — confirms the implementation is correct. Larger deviations mean a bug.

### Step 4: Self-Supervised Pretraining (`train_selfsup.py`)

#### 4a: Masked Patch Prediction
- Take the patch sequence `(batch, N, D)` after embedding
- Randomly mask 40% of patches (replace with zeros — the paper explicitly says "masked with zero values")
- Feed masked sequence through encoder
- Reconstruction head: `nn.Linear(D, P)` — predict the raw patch values (patch_len numbers) for each masked position only
- Loss: MSE between predicted and actual patch values, computed only on masked patches

#### 4b: Pretraining Loop
- Pretrain on ETTh1 train set (or Electricity dataset for transfer learning experiments)
- 100 epochs, Adam, lr=1e-4, batch size 128
- Random seed 2021
- No prediction head attached yet — just encoder + reconstruction head
- NOTE: self-supervised uses NON-OVERLAPPING patches (stride = patch_len = 16), unlike supervised which uses overlapping (stride 8)

#### 4c: Fine-tuning
Two evaluation modes after pretraining:

**Linear probe:**
- Freeze all encoder weights
- Attach the supervised prediction head from Step 2c
- Train only the head for 20 epochs
- lr=1e-4, Adam

**End-to-end fine-tuning:**
- Attach the supervised prediction head
- Freeze encoder, train head for 10 epochs (linear probing phase)
- Unfreeze encoder, train entire network for 20 more epochs (fine-tuning phase)
- Same lr=1e-4 throughout, Adam
- NOTE: the paper does NOT use differential learning rates for encoder vs head. Single lr for everything.

### Step 5: Validation & Debugging

1. **Sanity checks before real training:**
   - Overfit on a single batch (loss should go to ~0)
   - Check output shapes at every stage
   - Verify channel independence: output for channel k should not change if you modify channel j's input
   
2. **Compare with paper:**
   - Run supervised ETTh1 with T=96 first. If MSE is in 0.35-0.42 range, you're correct.
   - If MSE is >0.6 or <0.2, something is wrong.

3. **Common bugs to watch for:**
   - Normalisation data leakage (fitting scaler on val/test)
   - Patches computed incorrectly (off-by-one in unfold)
   - Positional encoding not matching number of patches
   - BatchNorm behaving differently in train vs eval mode (call model.train() / model.eval())
   - Channel independence not actually independent (accidentally mixing channels in batch dimension)

## Code Style Rules
- Pure PyTorch. No external ML frameworks.
- Minimal code. No unnecessary abstractions.
- Print statements for logging (no logging library).
- Add time profiling to the training loop (per-epoch timing).
- Use argparse for hyperparameters so different configs can be run from CLI.
- Type hints on function signatures.
- Single file per component, no classes where functions suffice.

## Order of Execution
1. Download ETTh1.csv into data/
2. Implement and test dataset.py (print shapes, verify splits)
3. Implement model.py, test with random input (verify shapes)
4. Implement train_supervised.py, run T=96
5. Compare MSE/MAE with paper
6. If correct, run T=192, 336, 720
7. Implement train_selfsup.py, run pretraining + linear probe
8. Compare self-supervised results with paper

## Assumptions Not In Paper
These details are not explicitly stated in the paper text. Items marked [code] are confirmed from the official repo (https://github.com/yuqinie98/PatchTST) or equivalent authoritative sources. Items marked [guess] need verification against the repo.

1. **Single learning rate during fine-tuning.** [guess] The paper says "end-to-end fine-tuning for 20 epochs" but doesn't state whether encoder and head use the same lr. ~70% confidence it's a single lr. Check `patchtst_finetune.py` for parameter groups.
2. **lr=1e-4 for all stages.** [code] Confirmed across HuggingFace blog, Nixtla reimplementation, and third-party repos using the official code.
3. **Batch size 128 for self-supervised.** [code] Confirmed from third-party repo invoking official code: `python patchtst_pretrain.py ... --batch_size 128`.
4. **Batch size 32 for supervised.** [code] Confirmed from official training scripts and Nixtla defaults.
5. **Masking uses literal zeros, NOT a learnable mask token.** [paper + code] The paper explicitly says "masked with zero values." Do NOT use `nn.Parameter` for the mask token. Just zero out the selected patches.
6. **Early stopping patience.** [guess] The code uses an `EarlyStopping` class but the patience value wasn't visible in search results. Check `run_longExp.py` argparse defaults.
7. **BatchNorm (not LayerNorm).** [code] Confirmed from backbone defaults: `norm:str='BatchNorm'`.
8. **Train/val/test split indices for ETTh1.** [code] From `data_loader.py`: boundaries computed as `12*30*24 = 8640` (train end), `+4*30*24 = 2880` each for val and test. So train=0:8640, val≈8640:11520, test≈11520:14400. NOTE: val/test start indices are shifted back by `seq_len` so the first window's lookback doesn't leak into the previous split.
9. **LR scheduler.** [guess] Code calls `adjust_learning_rate` but the implementation wasn't visible. Check `utils/tools.py` in the repo.

## Do NOT
- Copy code from the official PatchTST repo
- Use HuggingFace transformers library
- Add features not described here
- Use wandb or other experiment trackers
- Create separate config files — use argparse defaults


