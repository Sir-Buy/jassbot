"""Live training monitor for V6. Reads CSV and auto-refreshes.

Usage:
    python training_monitor.py
    python training_monitor.py --csv checkpoints_v6/training.csv --interval 5
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np


def find_csv():
    """Find most recent training.csv in checkpoint dirs."""
    base = os.path.dirname(os.path.abspath(__file__))
    for d in sorted(
        [x for x in os.listdir(base) if x.startswith("checkpoints_v")],
        reverse=True,
    ):
        p = os.path.join(base, d, "training.csv")
        if os.path.exists(p):
            return p
    return None


def read_csv(path):
    d = {k: [] for k in [
        "epoch", "dev", "med", "perf", "ploss", "vloss",
        "ent", "kl", "gps", "gpu", "lr", "ent_coeff",
    ]}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d["epoch"].append(int(row["epoch"]))
                d["dev"].append(float(row["avg_dev"]))
                d["med"].append(int(row["median_dev"]))
                d["perf"].append(float(row["perfect_pct"]))
                d["ploss"].append(float(row["policy_loss"]))
                d["vloss"].append(float(row["value_loss"]))
                d["ent"].append(float(row["entropy"]))
                d["kl"].append(float(row["kl"]))
                d["gps"].append(float(row["games_per_sec"]))
                d["gpu"].append(int(row["gpu_util"]))
                d["lr"].append(float(row["lr"]))
                d["ent_coeff"].append(float(row["ent_coeff"]))
    except Exception:
        pass
    return {k: np.array(v) for k, v in d.items()}


def roll(arr, w):
    if len(arr) < w:
        return arr, np.arange(len(arr))
    r = np.convolve(arr, np.ones(w) / w, mode="valid")
    return r, np.arange(w - 1, len(arr))


def draw(frame, path, fig, axes):
    d = read_csv(path)
    n = len(d["epoch"])
    if n < 2:
        return

    e, dev = d["epoch"], d["dev"]
    for ax in axes:
        ax.clear()

    # ── Top: Full deviation history ──
    ax = axes[0]
    W = max(20, n // 20)
    ax.plot(e, dev, color="#e53935", alpha=0.12, linewidth=0.5)
    rv, ri = roll(dev, W)
    ax.plot(e[ri], rv, color="#e53935", linewidth=3, label=f"Rolling {W}")
    bi = np.argmin(dev)
    ax.plot(e[bi], dev[bi], "v", color="#4CAF50", ms=12, zorder=5)
    ax.annotate(f" {dev[bi]:.1f}", (e[bi], dev[bi]), fontsize=10,
                fontweight="bold", color="#4CAF50")
    ax.axhline(5.6, color="#2196F3", ls="--", alpha=0.5, lw=1.5)
    ax.text(e[0] + 1, 5.9, "PIMC baseline (5.6)", fontsize=8, color="#2196F3")
    ax.axhline(12.5, color="#FF9800", ls=":", alpha=0.4, lw=1)
    ax.text(e[0] + 1, 12.8, "V5 converged (12.5)", fontsize=8, color="#FF9800", alpha=0.7)
    ax.axhline(8, color="#4CAF50", ls="--", alpha=0.4, lw=1)
    ax.text(e[0] + 1, 8.3, "Strong Human (8)", fontsize=8, color="#4CAF50", alpha=0.7)
    ax.set_ylabel("Avg Dev (per round)", fontsize=11)
    games_m = e[-1] * 4096 * 4 / 1e6
    ax.set_title(
        f"Epoch {e[-1]} | Dev {dev[-1]:.1f} | Best {dev[bi]:.1f} (E{e[bi]}) | ~{games_m:.0f}M games",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(max(0, dev[bi] - 2), max(dev[0] + 2, dev[-1] + 5))
    ax.grid(True, alpha=0.15)

    # ── Mid-left: Zoomed recent ──
    ax = axes[1]
    z = min(300, n)
    ez, dz = e[-z:], dev[-z:]
    ax.plot(ez, dz, color="#e53935", alpha=0.25, lw=0.8)
    W2 = max(10, z // 10)
    rv2, ri2 = roll(dz, W2)
    ax.plot(ez[ri2], rv2, color="#e53935", lw=2.5, label="Dev")
    if z >= 20:
        coef = np.polyfit(np.arange(z, dtype=float), dz, 1)
        trend_line = np.poly1d(coef)(np.arange(z))
        c = "#4CAF50" if coef[0] < -0.002 else ("#FF9800" if coef[0] < 0.002 else "#e53935")
        ax.plot(ez, trend_line, color=c, lw=1.5, ls="--", alpha=0.7)
        lbl = "improving" if coef[0] < -0.002 else ("flat" if coef[0] < 0.002 else "worse")
        ax.set_title(f"Last {z} -- {lbl} ({coef[0]:+.4f}/ep)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg Dev")
    ax.set_xlabel("Epoch")
    pad = max(0.5, (dz.max() - dz.min()) * 0.3)
    ax.set_ylim(dz.min() - pad, dz.max() + pad)
    ax.grid(True, alpha=0.15)

    # ── Mid-right: Entropy + KL ──
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(e, d["ent"], color="#00897B", lw=1.5, label="Entropy")
    ax.plot(e, d["kl"], color="#7B1FA2", lw=1, alpha=0.5, label="KL")
    ax2.plot(e, d["ploss"], color="#E65100", lw=1.5, ls="--", label="Policy Loss")
    ax.axhline(0.015, color="#7B1FA2", ls=":", alpha=0.3)
    ax.set_ylabel("Entropy / KL", fontsize=9)
    ax2.set_ylabel("Policy Loss", color="#E65100", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.set_title(
        f"ent={d['ent'][-1]:.3f}  kl={d['kl'][-1]:.4f}  pl={d['ploss'][-1]:.4f}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.15)

    # ── Bottom: Stats panel ──
    ax = axes[3]
    ax.axis("off")
    avg_gps = d["gps"].mean()
    wall_per_ep = 2.7  # approximate from profiling
    elapsed_min = n * wall_per_ep / 60
    remaining_ep = max(0, 5000 - e[-1])
    eta_hr = remaining_ep * wall_per_ep / 3600

    if n >= 50:
        imp = dev[-50] - dev[-1]
        trend = f"{imp:+.2f} last 50ep"
    else:
        trend = "warming up"

    left = (
        f"  Epoch:   {e[-1]:,} / 5,000\n"
        f"  Games:   ~{games_m:.0f}M\n"
        f"  Time:    ~{elapsed_min:.0f} min\n"
        f"  ETA:     ~{eta_hr:.1f} hr\n"
        f"  Speed:   {avg_gps:.0f} g/s"
    )
    right = (
        f"  Dev:     {dev[-1]:.1f} (best {dev[bi]:.1f})\n"
        f"  Median:  {d['med'][-1]}\n"
        f"  Perfect: {d['perf'][-1]:.1f}%\n"
        f"  GPU:     {d['gpu'][-1]}%\n"
        f"  Trend:   {trend}\n"
        f"  LR:      {d['lr'][-1]:.1e}"
    )
    ax.text(0.02, 0.9, left, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8fc", ec="#ddd"))
    ax.text(0.52, 0.9, right, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8fc", ec="#ddd"))

    if dev[-1] < 5.6:
        status, sc = "BEATING PIMC!", "#2196F3"
    elif dev[-1] < 8:
        status, sc = "SUPERHUMAN RANGE", "#4CAF50"
    elif dev[-1] < 12.5:
        status, sc = "BEATING V5", "#FF9800"
    else:
        status, sc = "TRAINING", "#e53935"
    ax.text(0.5, 0.05, status, transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="center", color=sc)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Jass V6 RL Training — 4-Round Rank Reward",
                 fontsize=14, fontweight="bold", y=0.99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    path = args.csv or find_csv()
    if not path:
        print("No CSV found. Pass --csv <path>")
        return
    print(f"Watching: {path} (refresh {args.interval}s)")

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.3, 1, 0.6], hspace=0.35, wspace=0.3)
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]
    ani = FuncAnimation(fig, draw, fargs=(path, fig, axes),
                        interval=args.interval * 1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
