"""Self-supervised PatchTST: masked patch prediction pretraining + fine-tuning on ETTh1."""

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import PatchTST, PatchTSTForPretrain
from utils import set_seed, mse, mae, EarlyStopping, save_results


def pretrain_one_epoch(model: nn.Module, loader, criterion, optimizer,
                       device: str, n_channels: int) -> float:
    """One pretraining epoch: each channel is an independent univariate sample."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, _ in loader:  # y (targets) not used in pretraining
        x = x.to(device)
        B, M, L = x.shape
        # Flatten channels into batch for channel independence
        x_flat = x.reshape(B * M, 1, L)  # (B*M, 1, L)

        pred_patches, target_patches, mask = model(x_flat)
        loss = criterion(pred_patches, target_patches)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def pretrain_evaluate(model: nn.Module, loader, criterion,
                      device: str, n_channels: int) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, _ in loader:
        x = x.to(device)
        B, M, L = x.shape
        x_flat = x.reshape(B * M, 1, L)
        pred_patches, target_patches, mask = model(x_flat)
        loss = criterion(pred_patches, target_patches)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def train_supervised_epoch(model: nn.Module, loader, criterion, optimizer,
                           device: str) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate_supervised(model: nn.Module, loader, criterion, device: str) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def test_metrics(model: nn.Module, loader, device: str) -> tuple[float, float]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return mse(preds, targets), mae(preds, targets)


def transfer_weights(pretrain_model: PatchTSTForPretrain, sup_model: PatchTST) -> None:
    """Transfer encoder weights from pretrained model to supervised model.

    Handles mismatches between pretrain and supervised architectures:
    - Patch projection: transferred only if patch_len matches, else re-initialized
    - Positional encoding: transferred only if patch count matches, else re-initialized
    - Encoder layers: always transferred (same d_model/nhead/ff)
    """
    # Transfer patch projection weights (only if patch_len matches)
    pretrain_proj = pretrain_model.patch_embed.proj
    sup_proj = sup_model.patch_embed.proj
    if pretrain_proj.in_features == sup_proj.in_features:
        sup_proj.load_state_dict(pretrain_proj.state_dict())
        print(f"  Transferred patch projection weights (P={sup_proj.in_features})")
    else:
        print(f"  Patch projection: pretrain P={pretrain_proj.in_features} != "
              f"supervised P={sup_proj.in_features} — re-initialized")

    # Transfer encoder weights
    sup_model.encoder.load_state_dict(pretrain_model.encoder.state_dict())
    print(f"  Transferred encoder weights ({len(sup_model.encoder.layers)} layers)")

    # Positional encoding: transfer only if patch count matches
    n_pretrain = pretrain_model.patch_embed.n_patches
    n_sup = sup_model.patch_embed.n_patches
    if n_pretrain != n_sup:
        print(f"  Positional encoding: pretrain has {n_pretrain} patches, "
              f"supervised has {n_sup} — re-initialized")
    else:
        sup_model.patch_embed.pos_enc.data.copy_(pretrain_model.patch_embed.pos_enc.data)
        print(f"  Transferred positional encoding ({n_sup} patches)")


def _ds_prefix(args) -> str:
    return "weather" if "weather" in args.data_path.lower() else "etth1"


def run_pretrain(args) -> str:
    """Run masked patch prediction pretraining. Returns path to saved checkpoint."""
    set_seed(args.seed)
    device = args.device
    ds = _ds_prefix(args)

    print(f"\n{'='*60}")
    dataset_name = "Weather" if "weather" in args.data_path.lower() else "ETTh1"
    print(f"Self-Supervised Pretraining | {dataset_name}")
    print(f"{'='*60}")

    # Data — use larger batch size and pretrain_seq_len for pretraining
    pretrain_seq_len = args.pretrain_seq_len
    pretrain_patch_len = args.pretrain_patch_len
    train_loader, val_loader, _, scaler, _ = get_dataloaders(
        args.data_path, pretrain_seq_len, args.pred_len, args.pretrain_batch_size
    )
    print(f"Train: {len(train_loader.dataset)} windows, Val: {len(val_loader.dataset)}")
    print(f"Pretrain seq_len={pretrain_seq_len}, patch_len={pretrain_patch_len} (non-overlapping)")

    # Pretrain model with non-overlapping patches (stride = patch_len)
    model = PatchTSTForPretrain(
        seq_len=pretrain_seq_len, patch_len=pretrain_patch_len, stride=pretrain_patch_len,
        d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, n_layers=args.n_layers, mask_ratio=args.mask_ratio
    ).to(device)
    print(f"Pretrain model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Patches: {model.patch_embed.n_patches} (P={pretrain_patch_len}, S={pretrain_patch_len}), "
          f"Mask ratio: {args.mask_ratio}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs("results", exist_ok=True)
    checkpoint_path = f"results/checkpoint_pretrain_{ds}.pt"
    early_stopping = EarlyStopping(patience=args.patience, save_path=checkpoint_path)

    history = []
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Time':>6}")
    print("-" * 42)

    for epoch in range(1, args.pretrain_epochs + 1):
        t0 = time.time()
        train_loss = pretrain_one_epoch(model, train_loader, criterion, optimizer,
                                        device, args.n_channels)
        val_loss = pretrain_evaluate(model, val_loader, criterion, device, args.n_channels)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "time_s": round(elapsed, 1)
        })
        print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {elapsed:5.1f}s")

        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break

    early_stopping.load_best(model)
    save_results({"pretrain_history": history}, f"results/{ds}_pretrain_history.json")
    return checkpoint_path


def run_finetune(args, pretrain_checkpoint: str, mode: str = "finetune") -> dict:
    """Fine-tune or linear-probe a pretrained model.

    Args:
        mode: 'linear_probe' (freeze encoder, train head only) or
              'finetune' (freeze 10 epochs, then unfreeze 20 epochs)
    Returns:
        Results dict with test metrics and history.
    """
    set_seed(args.seed)
    device = args.device

    print(f"\n{'='*60}")
    print(f"{'Linear Probe' if mode == 'linear_probe' else 'Fine-tuning'} | "
          f"pred_len={args.pred_len}")
    print(f"{'='*60}")

    # Data — supervised batch size
    train_loader, val_loader, test_loader, scaler, _ = get_dataloaders(
        args.data_path, args.seq_len, args.pred_len, args.batch_size
    )

    # Load pretrained model with PRETRAIN architecture (P=12, S=12, L=512)
    pretrain_model = PatchTSTForPretrain(
        seq_len=args.pretrain_seq_len, patch_len=args.pretrain_patch_len,
        stride=args.pretrain_patch_len,
        d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, n_layers=args.n_layers, mask_ratio=args.mask_ratio
    )
    pretrain_model.load_state_dict(torch.load(pretrain_checkpoint, weights_only=True))

    # Create supervised model and transfer weights
    sup_model = PatchTST(
        seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
        stride=args.stride, d_model=args.d_model, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        n_layers=args.n_layers, n_channels=args.n_channels
    ).to(device)
    transfer_weights(pretrain_model, sup_model)
    del pretrain_model

    criterion = nn.MSELoss()
    os.makedirs("results", exist_ok=True)
    ds = _ds_prefix(args)
    checkpoint_path = f"results/checkpoint_{ds}_{mode}_T{args.pred_len}.pt"
    history = []

    if mode == "linear_probe":
        # Freeze encoder, train head only
        for param in sup_model.patch_embed.parameters():
            param.requires_grad = False
        for param in sup_model.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in sup_model.parameters() if p.requires_grad)
        print(f"Trainable params (head only): {trainable:,}")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sup_model.parameters()),
                                     lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, save_path=checkpoint_path)

        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Time':>6}")
        print("-" * 42)

        for epoch in range(1, args.probe_epochs + 1):
            t0 = time.time()
            train_loss = train_supervised_epoch(sup_model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_supervised(sup_model, val_loader, criterion, device)
            elapsed = time.time() - t0
            history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                           "val_loss": round(val_loss, 6), "time_s": round(elapsed, 1),
                           "phase": "linear_probe"})
            print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {elapsed:5.1f}s")
            if early_stopping(val_loss, sup_model):
                print(f"Early stopping at epoch {epoch}")
                break

    elif mode == "finetune":
        # Phase 1: Freeze encoder, train head for 10 epochs
        for param in sup_model.patch_embed.parameters():
            param.requires_grad = False
        for param in sup_model.encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sup_model.parameters()),
                                     lr=args.lr)
        print(f"\n--- Phase 1: Linear probing (encoder frozen, {args.ft_freeze_epochs} epochs) ---")
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Time':>6}")
        print("-" * 42)

        for epoch in range(1, args.ft_freeze_epochs + 1):
            t0 = time.time()
            train_loss = train_supervised_epoch(sup_model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_supervised(sup_model, val_loader, criterion, device)
            elapsed = time.time() - t0
            history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                           "val_loss": round(val_loss, 6), "time_s": round(elapsed, 1),
                           "phase": "frozen"})
            print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {elapsed:5.1f}s")

        # Phase 2: Unfreeze all, fine-tune for 20 epochs
        for param in sup_model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(sup_model.parameters(), lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, save_path=checkpoint_path)

        print(f"\n--- Phase 2: Full fine-tuning ({args.ft_unfreeze_epochs} epochs) ---")
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Time':>6}")
        print("-" * 42)

        for epoch in range(1, args.ft_unfreeze_epochs + 1):
            t0 = time.time()
            train_loss = train_supervised_epoch(sup_model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_supervised(sup_model, val_loader, criterion, device)
            elapsed = time.time() - t0
            total_epoch = args.ft_freeze_epochs + epoch
            history.append({"epoch": total_epoch, "train_loss": round(train_loss, 6),
                           "val_loss": round(val_loss, 6), "time_s": round(elapsed, 1),
                           "phase": "unfrozen"})
            print(f"{total_epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {elapsed:5.1f}s")
            if early_stopping(val_loss, sup_model):
                print(f"Early stopping at epoch {total_epoch}")
                break

    # Test evaluation
    early_stopping.load_best(sup_model)
    test_mse_val, test_mae_val = test_metrics(sup_model, test_loader, device)

    # Paper Table 4 reference values (dataset-aware)
    paper_refs = {
        "weather": {
            "linear_probe": {96: (0.158, 0.209), 192: (0.203, 0.249), 336: (0.251, 0.285), 720: (0.321, 0.336)},
            "finetune": {96: (0.144, 0.193), 192: (0.190, 0.236), 336: (0.244, 0.280), 720: (0.320, 0.335)},
            "supervised": {96: (0.152, 0.199), 192: (0.197, 0.243), 336: (0.249, 0.283), 720: (0.320, 0.335)},
        },
        "etth1": {
            # No self-supervised in paper for ETTh1; show supervised for reference
            "linear_probe": {96: (0.375, 0.399), 192: (0.414, 0.421), 336: (0.431, 0.436), 720: (0.449, 0.466)},
            "finetune": {96: (0.375, 0.399), 192: (0.414, 0.421), 336: (0.431, 0.436), 720: (0.449, 0.466)},
            "supervised": {96: (0.375, 0.399), 192: (0.414, 0.421), 336: (0.431, 0.436), 720: (0.449, 0.466)},
        },
    }
    ds_refs = paper_refs.get(ds, paper_refs["etth1"])
    mode_refs = ds_refs.get(mode, ds_refs["supervised"])
    paper_mse, paper_mae = mode_refs.get(args.pred_len, (None, None))
    sup_mse, sup_mae = ds_refs["supervised"].get(args.pred_len, (None, None))

    print(f"\nTEST RESULTS ({mode}) | pred_len={args.pred_len}")
    print(f"  MSE: {test_mse_val:.4f}  (paper {mode}: {paper_mse}, paper supervised: {sup_mse})")
    print(f"  MAE: {test_mae_val:.4f}  (paper {mode}: {paper_mae}, paper supervised: {sup_mae})")

    results = {
        "mode": mode,
        "config": {k: v for k, v in vars(args).items()},
        "train_history": history,
        "test_metrics": {"mse": round(test_mse_val, 6), "mae": round(test_mae_val, 6)},
        "paper_metrics": {"mse": paper_mse, "mae": paper_mae},
    }
    save_results(results, f"results/{ds}_selfsup_{mode}_T{args.pred_len}.json")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="PatchTST Self-Supervised on ETTh1")
    parser.add_argument("--data_path", type=str, default="data/ETTh1.csv")
    parser.add_argument("--pred_len", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pretrain_seq_len", type=int, default=512, help="Lookback for pretraining (paper uses 512)")
    parser.add_argument("--patch_len", type=int, default=16, help="Patch len for supervised fine-tune head")
    parser.add_argument("--pretrain_patch_len", type=int, default=12, help="Patch len for pretraining (paper uses 12)")
    parser.add_argument("--stride", type=int, default=8, help="Stride for supervised head (fine-tune)")
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_channels", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--ft_freeze_epochs", type=int, default=10)
    parser.add_argument("--ft_unfreeze_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--skip_pretrain", action="store_true", help="Skip pretraining, use existing checkpoint")
    parser.add_argument("--mode", type=str, default="both", choices=["linear_probe", "finetune", "both"])
    args = parser.parse_args()

    # Step 1: Pretraining
    ds = _ds_prefix(args)
    pretrain_checkpoint = f"results/checkpoint_pretrain_{ds}.pt"
    if not args.skip_pretrain:
        pretrain_checkpoint = run_pretrain(args)
    else:
        print(f"Skipping pretraining, using checkpoint: {pretrain_checkpoint}")

    # Step 2: Fine-tuning / Linear probe
    if args.mode in ("linear_probe", "both"):
        run_finetune(args, pretrain_checkpoint, mode="linear_probe")

    if args.mode in ("finetune", "both"):
        run_finetune(args, pretrain_checkpoint, mode="finetune")

    # Keep pretrain checkpoint for later phases (fine-tuning)
    print(f"Pretrain checkpoint preserved at: {pretrain_checkpoint}")


if __name__ == "__main__":
    main()
