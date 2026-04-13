"""Neural declaration network for Differenzler.

Predicts the expected score (optimal declaration) from hand + trump features.
Trained with L1 loss (Bayes-optimal for median estimation, which minimizes
expected absolute deviation).

Usage:
    # Train
    python -m bot.declaration_net --train --data data/decl_features.npy --scores data/decl_scores.npy

    # Evaluate
    python -m bot.declaration_net --eval --model data/declaration_model.pt
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeclarationNet(nn.Module):
    """Neural network for Differenzler declaration prediction.

    Input: 72 features (hand binary + trump + derived features)
    Output: 1 value (predicted score, 0-157)
    """

    def __init__(self, input_size: int = 72, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train(
    features_path: str,
    scores_path: str,
    output_path: str = "data/declaration_model.pt",
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_split: float = 0.1,
    patience: int = 10,
) -> dict:
    """Train the declaration network.

    Returns dict with training metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load data
    features = np.load(features_path)
    scores = np.load(scores_path)
    n = len(features)
    input_size = features.shape[1]
    print(f"Data: {n} samples, {input_size} features")
    print(f"Scores: mean={scores.mean():.1f}, std={scores.std():.1f}, "
          f"min={scores.min():.0f}, max={scores.max():.0f}")

    # Shuffle and split
    idx = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    X_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(scores[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(scores[val_idx], dtype=torch.float32, device=device)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Model
    model = DeclarationNet(input_size=input_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=(len(X_train) + batch_size - 1) // batch_size,
    )

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        train_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            xb = X_shuf[i:i + batch_size]
            yb = y_shuf[i:i + batch_size]

            pred = model(xb)
            loss = F.l1_loss(pred, yb)  # L1 = Bayes-optimal for median

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae = F.l1_loss(val_pred, y_val).item()
            val_bias = (val_pred - y_val).mean().item()

            # Additional metrics
            val_errors = (val_pred - y_val).abs()
            within_5 = (val_errors <= 5).float().mean().item() * 100
            within_10 = (val_errors <= 10).float().mean().item() * 100

        avg_train = train_loss / n_batches

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:>3d}: train_L1={avg_train:.2f}  "
                  f"val_MAE={val_mae:.2f}  bias={val_bias:+.1f}  "
                  f"<5={within_5:.1f}%  <10={within_10:.1f}%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - t0

    # Save best model
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "input_size": input_size,
        "best_val_mae": best_val_mae,
    }, output_path)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_errors = (val_pred - y_val).abs()
        final_mae = val_errors.mean().item()
        final_bias = (val_pred - y_val).mean().item()
        within_5 = (val_errors <= 5).float().mean().item() * 100
        within_10 = (val_errors <= 10).float().mean().item() * 100
        within_15 = (val_errors <= 15).float().mean().item() * 100
        within_20 = (val_errors <= 20).float().mean().item() * 100

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val MAE: {final_mae:.2f}")
    print(f"Bias: {final_bias:+.2f}")
    print(f"Within  5: {within_5:.1f}%")
    print(f"Within 10: {within_10:.1f}%")
    print(f"Within 15: {within_15:.1f}%")
    print(f"Within 20: {within_20:.1f}%")
    print(f"Saved to: {output_path}")

    return {
        "val_mae": final_mae,
        "bias": final_bias,
        "within_5": within_5,
        "within_10": within_10,
        "elapsed": elapsed,
    }


class NeuralDeclarationModel:
    """Drop-in replacement for DeclarationModel using neural network.

    Provides instant declaration without simulation.
    """

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "declaration_model.pt"
            )
        if os.path.exists(model_path):
            self._load(model_path)

    def _load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        input_size = checkpoint.get("input_size", 72)
        self._model = DeclarationNet(input_size=input_size).to(self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

    @property
    def available(self) -> bool:
        return self._model is not None

    def declare(self, hand: list[int], trump: int, **kwargs) -> int:
        """Predict optimal declaration from hand + trump. Instant (<1ms)."""
        if not self.available:
            raise RuntimeError("Neural declaration model not loaded")

        # Import here to avoid circular imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.generate_declaration_data import encode_hand_rich

        feat = encode_hand_rich(hand, trump)
        x = torch.tensor(feat, dtype=torch.float32, device=self._device).unsqueeze(0)

        with torch.no_grad():
            pred = self._model(x).item()

        return max(0, min(157, round(pred)))

    def declare_with_info(self, hand: list[int], trump: int, **kwargs) -> dict:
        """Returns declaration plus diagnostic info."""
        decl = self.declare(hand, trump)
        return {
            "declaration": decl,
            "method": "neural",
            "median": decl,
            "mean": float(decl),
            "std": 0.0,
            "min": decl,
            "max": decl,
            "distribution": [decl],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--data", default="data/decl_features.npy")
    parser.add_argument("--scores", default="data/decl_scores.npy")
    parser.add_argument("--model", default="data/declaration_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        train(args.data, args.scores, args.model,
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.eval:
        model = NeuralDeclarationModel(args.model)
        if not model.available:
            print("Model not found!")
            sys.exit(1)

        # Quick test
        test_hands = [
            ([5, 3, 0, 1, 2, 17, 10, 19, 28], 0, "Puur+Nell+3low+Ace"),
            ([0, 10, 19, 28, 1, 11, 20, 29, 2], 3, "All low cards"),
            ([5, 3, 8, 4, 7, 17, 26, 19, 28], 0, "5 trump + 2 aces"),
        ]
        for hand, trump, desc in test_hands:
            decl = model.declare(hand, trump)
            print(f"  {desc}: declared={decl}")
