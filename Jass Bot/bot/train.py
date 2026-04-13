"""Train the Differenzler neural network on PIMC self-play data."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.neural_net import DifferenzlerNet, FEATURE_SIZE


def train(data_dir: str = "data", output_path: str = "data/model_poc.pt",
          epochs: int = 30, batch_size: int = 1024, lr: float = 1e-3,
          patience: int = 5):
    # ===== Device setup =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    use_amp = device.type == "cuda"
    print(f"  Mixed precision: {use_amp}")
    print(f"  CPU cores: {os.cpu_count()}")

    # ===== Load data =====
    features = np.load(os.path.join(data_dir, "poc_features.npy"))
    labels_card = np.load(os.path.join(data_dir, "poc_labels_card.npy"))
    labels_dev = np.load(os.path.join(data_dir, "poc_labels_deviation.npy"))

    N = len(features)
    print(f"\nData: {N} samples, feature_size={features.shape[1]}")

    # Convert to tensors, move to device
    X = torch.from_numpy(features).to(device)
    y_card = torch.from_numpy(labels_card).long().to(device)
    y_dev = torch.from_numpy(labels_dev).float().to(device) / 157.0  # normalize

    # Extract legal mask from features (positions 144:180)
    legal_mask = X[:, 144:180] > 0.5

    # Train/val split (90/10)
    perm = torch.randperm(N, device=device)
    n_val = N // 10
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_card_train, y_card_val = y_card[train_idx], y_card[val_idx]
    y_dev_train, y_dev_val = y_dev[train_idx], y_dev[val_idx]
    mask_train, mask_val = legal_mask[train_idx], legal_mask[val_idx]

    n_train = len(X_train)
    n_batches = (n_train + batch_size - 1) // batch_size

    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"  Batches/epoch: {n_batches}, Batch size: {batch_size}")

    # ===== Model =====
    model = DifferenzlerNet(FEATURE_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * n_batches, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ===== Training loop =====
    print(f"\n{'Epoch':>5} {'TrLoss':>8} {'VaLoss':>8} {'PolAcc':>7} {'Top3':>6} "
          f"{'ValMAE':>7} {'LR':>10} {'Time':>6} {'GPU':>6}")
    print("-" * 78)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # Shuffle
        idx = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        train_batches = 0

        for b in range(n_batches):
            bi = idx[b * batch_size:(b + 1) * batch_size]
            xb = X_train[bi]
            yc = y_card_train[bi]
            yd = y_dev_train[bi]
            mb = mask_train[bi]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits, values = model(xb, mb)
                    ploss = F.cross_entropy(logits, yc)
                    vloss = F.mse_loss(values.squeeze(), yd)
                    loss = ploss + 0.5 * vloss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, values = model(xb, mb)
                ploss = F.cross_entropy(logits, yc)
                vloss = F.mse_loss(values.squeeze(), yd)
                loss = ploss + 0.5 * vloss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            train_loss_sum += loss.item()
            train_batches += 1

        # ===== Validation =====
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    vl, vv = model(X_val, mask_val)
                    vploss = F.cross_entropy(vl, y_card_val)
                    vvloss = F.mse_loss(vv.squeeze(), y_dev_val)
                    val_loss = (vploss + 0.5 * vvloss).item()
            else:
                vl, vv = model(X_val, mask_val)
                vploss = F.cross_entropy(vl, y_card_val)
                vvloss = F.mse_loss(vv.squeeze(), y_dev_val)
                val_loss = (vploss + 0.5 * vvloss).item()

            # Metrics
            preds = vl.argmax(dim=1)
            pol_acc = (preds == y_card_val).float().mean().item()
            # Top-3
            top3 = vl.topk(3, dim=1).indices
            top3_hit = (top3 == y_card_val.unsqueeze(1)).any(dim=1).float().mean().item()
            # Value MAE (in original scale)
            val_mae = (vv.squeeze() - y_dev_val).abs().mean().item() * 157

        train_loss = train_loss_sum / train_batches
        cur_lr = optimizer.param_groups[0]["lr"]
        gpu_mb = torch.cuda.memory_allocated() / 1e6 if device.type == "cuda" else 0
        dt = time.time() - t0

        print(f"{epoch:5d} {train_loss:8.4f} {val_loss:8.4f} {pol_acc:6.1%} {top3_hit:5.1%} "
              f"{val_mae:7.1f} {cur_lr:10.6f} {dt:5.1f}s {gpu_mb:5.0f}M")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    # ===== Load best and final eval =====
    model.load_state_dict(torch.load(output_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        vl, vv = model(X_val, mask_val)
        preds = vl.argmax(dim=1)
        pol_acc = (preds == y_card_val).float().mean().item()
        top3 = vl.topk(3, dim=1).indices
        top3_hit = (top3 == y_card_val.unsqueeze(1)).any(dim=1).float().mean().item()
        val_mae = (vv.squeeze() - y_dev_val).abs().mean().item() * 157

    print(f"\nBest model (epoch {best_epoch}):")
    print(f"  Policy accuracy: {pol_acc:.1%}")
    print(f"  Top-3 accuracy:  {top3_hit:.1%}")
    print(f"  Value MAE:       {val_mae:.1f}")

    # ===== Pass/Fail =====
    p1 = pol_acc > 0.35
    p2 = top3_hit > 0.65
    p3 = val_mae < 7.0
    print(f"\nPASS/FAIL CRITERIA:")
    print(f"  Policy acc > 35%: {'PASS' if p1 else 'FAIL'} ({pol_acc:.1%})")
    print(f"  Top-3 acc > 65%:  {'PASS' if p2 else 'FAIL'} ({top3_hit:.1%})")
    print(f"  Value MAE < 7.0:  {'PASS' if p3 else 'FAIL'} ({val_mae:.1f})")

    if p1 and p2 and p3:
        print("\nPOC PASSED — neural approach works")
    else:
        print("\nPOC FAILED — debug before scaling")

    return model


if __name__ == "__main__":
    train()
