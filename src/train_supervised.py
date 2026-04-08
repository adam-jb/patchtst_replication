"""Supervised PatchTST training on ETTh1."""

import argparse
import sys
import time
import json
import os

# Allow running from project root: python src/train_supervised.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import PatchTST
from utils import set_seed, mse, mae, EarlyStopping, save_results


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: str,
                    max_grad_norm: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: str) -> float:
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
    """Compute MSE and MAE over the full test set (concatenated, not averaged per batch)."""
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


def overfit_single_batch(model: nn.Module, loader, device: str, n_iters: int = 200) -> None:
    """Sanity check: overfit on one batch, loss should go near zero."""
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(n_iters):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"  Overfit iter {i+1}: loss={loss.item():.6f}")
    final_loss = loss.item()
    if final_loss < 0.01:
        print(f"  Overfit check PASSED (loss={final_loss:.6f})")
    else:
        print(f"  WARNING: Overfit check may have issues (loss={final_loss:.6f})")


def main():
    parser = argparse.ArgumentParser(description="PatchTST Supervised Training on ETTh1")
    parser.add_argument("--data_path", type=str, default="data/ETTh1.csv")
    parser.add_argument("--pred_len", type=int, default=96, choices=[96, 192, 336, 720])
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_channels", type=int, default=None, help="Auto-detected from data if not set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--sanity_check", action="store_true", help="Run overfit check before training")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    run_start = time.time()
    print(f"\n{'='*60}")
    dataset_name = "Weather" if "weather" in args.data_path.lower() else "ETTh1"
    print(f"PatchTST Supervised | {dataset_name} | pred_len={args.pred_len}")
    print(f"{'='*60}")
    print(f"Device: {device}, Seed: {args.seed}")

    # Data
    train_loader, val_loader, test_loader, scaler, detected_channels = get_dataloaders(
        args.data_path, args.seq_len, args.pred_len, args.batch_size
    )
    n_channels = detected_channels if args.n_channels is None else args.n_channels
    print(f"Train: {len(train_loader.dataset)} windows, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}, "
          f"Channels: {n_channels}")

    # Model
    model = PatchTST(
        seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
        stride=args.stride, d_model=args.d_model, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        n_layers=args.n_layers, n_channels=n_channels
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sanity check
    if args.sanity_check:
        print("\n--- Overfit sanity check ---")
        sanity_model = PatchTST(
            seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
            stride=args.stride, d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward, dropout=args.dropout,
            n_layers=args.n_layers, n_channels=args.n_channels
        ).to(device)
        overfit_single_batch(sanity_model, train_loader, device)
        del sanity_model
        set_seed(args.seed)  # Reset seed for real training
        print("---\n")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    ds = "weather" if "weather" in args.data_path.lower() else "etth1"
    checkpoint_path = f"results/checkpoint_sup_{ds}_T{args.pred_len}.pt"
    os.makedirs("results", exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, save_path=checkpoint_path)

    # Training loop
    history = []
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": current_lr,
            "time_s": round(elapsed, 1)
        })

        print(f"{epoch:5d} | {train_loss:10.6f} | {val_loss:10.6f} | {current_lr:.2e} | {elapsed:5.1f}s")

        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break

    # Test evaluation
    early_stopping.load_best(model)
    test_mse, test_mae = test_metrics(model, test_loader, device)

    # Paper Table 3 reference results (PatchTST/42 for ETTh1, PatchTST/64 for Weather)
    paper_refs = {
        "etth1": {96: (0.375, 0.399), 192: (0.414, 0.421), 336: (0.431, 0.436), 720: (0.449, 0.466)},
        "weather": {96: (0.152, 0.199), 192: (0.197, 0.243), 336: (0.249, 0.283), 720: (0.320, 0.335)},
    }
    dataset_key = "weather" if "weather" in args.data_path.lower() else "etth1"
    paper_results = paper_refs.get(dataset_key, {})
    paper_mse, paper_mae = paper_results.get(args.pred_len, (None, None))

    total_time = time.time() - run_start
    n_epochs = len(history)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS | pred_len={args.pred_len}")
    print(f"  MSE: {test_mse:.4f}  (paper: {paper_mse})")
    print(f"  MAE: {test_mae:.4f}  (paper: {paper_mae})")
    if paper_mse:
        print(f"  MSE diff: {(test_mse - paper_mse) / paper_mse * 100:+.1f}%")
        print(f"  MAE diff: {(test_mae - paper_mae) / paper_mae * 100:+.1f}%")
    print(f"  Trained {n_epochs} epochs in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}\n")

    # Save results
    # Collect a few sample predictions for visualisation
    model.eval()
    sample_preds, sample_targets, sample_inputs = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 5:
                break
            x_dev = x.to(device)
            pred = model(x_dev).cpu().numpy()
            sample_preds.append(pred[0].tolist())      # first sample in batch
            sample_targets.append(y[0].numpy().tolist())
            sample_inputs.append(x[0].numpy().tolist())

    results = {
        "config": vars(args),
        "train_history": history,
        "test_metrics": {"mse": round(test_mse, 6), "mae": round(test_mae, 6)},
        "paper_metrics": {"mse": paper_mse, "mae": paper_mae},
        "total_time_s": round(total_time, 1),
        "n_epochs": n_epochs,
        "sample_predictions": sample_preds,
        "sample_targets": sample_targets,
        "sample_inputs": sample_inputs,
    }
    save_results(results, f"results/{ds}_supervised_T{args.pred_len}.json")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
