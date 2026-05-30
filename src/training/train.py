"""
Train MultiTaskDiagnosticModel on physics-based synthetic dam monitoring data.

Usage:
    python -m src.training.train                    # train with defaults
    python -m src.training.train --epochs 100       # custom epochs
    python -m src.training.train --quick            # fast test run
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.transformer_model import MultiTaskDiagnosticModel, MultiTaskLoss, create_model
from src.core.data_generator import (
    generate_multi_hazard_dataset,
    generate_sliding_window_dataset,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / 'outputs' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_datasets(n_samples=1300, seq_length=30, pred_horizon=10,
                   train_split=0.7, val_split=0.15, seed=42):
    """Generate data and split into train/val/test sets."""
    print("Generating multi-hazard dam monitoring data...")
    positions, full_data, hazard_labels, severity_labels, damage_spans = \
        generate_multi_hazard_dataset(n_samples=n_samples, seed=seed)

    print("Building sliding windows...")
    X, y_sensor, y_hazard, y_severity, y_span = generate_sliding_window_dataset(
        positions, full_data, hazard_labels, severity_labels, damage_spans,
        seq_length=seq_length, pred_horizon=pred_horizon,
    )

    n_total = len(X)
    indices = np.random.RandomState(seed).permutation(n_total)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Fit scaler on training data only
    n_features = X.shape[-1]
    X_train_flat = X[train_idx].reshape(-1, n_features)
    scaler = StandardScaler().fit(X_train_flat)

    def scale(X_arr):
        s = X_arr.shape
        return scaler.transform(X_arr.reshape(-1, n_features)).reshape(s)

    datasets = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        X_s = scale(X[idx])
        y_s = scale(y_sensor[idx])
        datasets[name] = (
            torch.FloatTensor(X_s),
            torch.FloatTensor(y_s),
            torch.LongTensor(y_hazard[idx]),
            torch.LongTensor(y_severity[idx]),
            torch.FloatTensor(y_span[idx]),
        )
        print(f"  {name}: {len(idx)} windows")

    return datasets, scaler


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y_sensor, y_hazard, y_severity, y_span in loader:
        X = X.to(device)
        y_sensor = y_sensor.to(device)
        y_hazard = y_hazard.to(device)
        y_severity = y_severity.to(device)
        y_span = y_span.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, (y_sensor, y_span, y_hazard, y_severity), X)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_hazard = 0
    correct_severity = 0
    total_samples = 0

    for X, y_sensor, y_hazard, y_severity, y_span in loader:
        X = X.to(device)
        y_sensor = y_sensor.to(device)
        y_hazard = y_hazard.to(device)
        y_severity = y_severity.to(device)
        y_span = y_span.to(device)

        outputs = model(X)
        loss = criterion(outputs, (y_sensor, y_span, y_hazard, y_severity), X)
        total_loss += loss.item()

        _, _, hazard_logits, severity_logits = outputs
        correct_hazard += (hazard_logits.argmax(dim=-1) == y_hazard).sum().item()
        correct_severity += (severity_logits.argmax(dim=-1) == y_severity).sum().item()
        total_samples += X.size(0)

    return (total_loss / len(loader),
            correct_hazard / total_samples,
            correct_severity / total_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-samples', type=int, default=1300)
    parser.add_argument('--seq-length', type=int, default=30)
    parser.add_argument('--pred-horizon', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--quick', action='store_true', help='Fast test run with fewer samples')
    parser.add_argument('--profile', type=str, default='xiaowan',
                        choices=['xiaowan', 'goupitan', 'pineflat_gravity'],
                        help='Dam profile for physics calibration')
    args = parser.parse_args()

    if args.quick:
        args.n_samples = 130
        args.epochs = 5

    # Set dam profile before data generation
    import src.core.data_generator as dg
    dg.ACTIVE_PROFILE = args.profile
    profile = dg._get_profile()
    bl = dg._baseline_params()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dam Profile: {profile['name']} | H={profile['height_m']}m L={profile['crest_length_m']}m")
    print(f"  E={profile['elastic_modulus_gpa']}GPa rho={profile['density_kgm3']} nu={profile['poisson_ratio']}")
    print(f"  Baseline: strain={bl['strain']:.1f}ue PP={bl['pore_pressure']:.1f}kPa "
          f"disp={bl['displacement']:.2f}mm")
    print(f"Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, samples={args.n_samples}")

    # Build data
    datasets, scaler = build_datasets(
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        pred_horizon=args.pred_horizon,
    )

    train_loader = DataLoader(
        TensorDataset(*datasets['train']),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(*datasets['val']),
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(*datasets['test']),
        batch_size=args.batch_size, shuffle=False,
    )

    # Model
    n_features = datasets['train'][0].shape[-1]
    model = create_model('multitask', n_features=n_features,
                         pred_horizon=args.pred_horizon)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: MultiTaskDiagnosticModel | {n_params:,} params | device={device}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MultiTaskLoss(learnable_weights=True)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_hazard_acc': [], 'val_severity_acc': []}

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_h_acc, val_s_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_hazard_acc'].append(val_h_acc)
        history['val_severity_acc'].append(val_s_acc)

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_DIR / 'best_model.pth')
            with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            improved = " *"
        else:
            patience_counter += 1
            improved = ""

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"h_acc={val_h_acc:.3f} | s_acc={val_s_acc:.3f}{improved}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    elapsed = time.time() - t_start
    print(f"Training completed in {elapsed:.1f}s | Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth', map_location=device))
    test_loss, test_h_acc, test_s_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f} | hazard_acc={test_h_acc:.3f} | severity_acc={test_s_acc:.3f}")

    # Save final artifacts
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': {
            'n_features': n_features,
            'seq_length': args.seq_length,
            'pred_horizon': args.pred_horizon,
            'n_params': n_params,
        },
        'history': history,
    }, MODEL_DIR / 'model_checkpoint.pt')

    print(f"Models saved to {MODEL_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()
