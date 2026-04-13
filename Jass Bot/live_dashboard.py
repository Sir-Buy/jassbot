"""Live dashboard for Jass Bot performance monitoring.

Auto-refreshes every 10 seconds, reads Export Games folder.
Shows rolling average deviation, per-round results, and comparison vs Swisslos bots.

Usage:
    python live_dashboard.py
    python live_dashboard.py --dir "Jass Data/Export Games"
"""

import json
import glob
import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DEFAULT_EXPORT = SCRIPT_DIR / "data" / "export_games"

SUIT_ID = {'H': 0, 'D': 1, 'S': 2, 'C': 3}
SUIT_NAME = {0: 'Herz', 1: 'Ecke', 2: 'Schaufel', 3: 'Kreuz'}
SUIT_SYM = {'H': 'H', 'D': 'D', 'S': 'S', 'C': 'C'}


def load_rounds(export_dir):
    """Load all rounds from export directory."""
    files = sorted(
        glob.glob(str(export_dir / "jass_data_*.json")) +
        glob.glob(str(export_dir / "round_*.json"))
    )
    rounds = []
    for fpath in files:
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Handle both formats
        if 'matches' in data:
            matches = data['matches']
        elif 'match_id' in data:
            matches = [data]
        else:
            continue

        for m in matches:
            players = m.get('players', [])
            # Find our seat
            our_seat = None
            for j, p in enumerate(players):
                name = p.get('name', '') if isinstance(p, dict) else str(p)
                uid = p.get('uid', 'x') if isinstance(p, dict) else 'x'
                if 'Gast' in name or uid is None:
                    our_seat = j
                    break

            for r in m.get('rounds', []):
                trump = r.get('trump', r.get('trump_id', ''))
                if isinstance(trump, str) and trump in SUIT_ID:
                    trump_code = trump
                elif isinstance(trump, int):
                    trump_code = ['H', 'D', 'S', 'C'][trump] if 0 <= trump <= 3 else '?'
                else:
                    trump_code = '?'

                results = r.get('results', {})
                res_list = results.get('results', []) if isinstance(results, dict) else []

                our_res = None
                opp_devs = []
                for res in res_list:
                    pos = res.get('pos')
                    declared = res.get('callP', 0)
                    scored = res.get('points', 0)
                    dev = abs(declared - scored)
                    if pos == our_seat:
                        our_res = {
                            'declared': declared,
                            'scored': scored,
                            'deviation': dev,
                            'trump': trump_code,
                        }
                    else:
                        opp_devs.append(dev)

                if our_res:
                    our_res['opp_avg_dev'] = sum(opp_devs) / len(opp_devs) if opp_devs else 0
                    rounds.append(our_res)

    return rounds


def update_dashboard(frame, export_dir, fig, axes):
    """Called every refresh interval."""
    rounds = load_rounds(export_dir)
    if not rounds:
        return

    n = len(rounds)
    devs = [r['deviation'] for r in rounds]
    opp_devs = [r['opp_avg_dev'] for r in rounds]
    declared = [r['declared'] for r in rounds]
    scored = [r['scored'] for r in rounds]
    trumps = [r['trump'] for r in rounds]

    # Rolling averages
    window = min(10, n)
    rolling_dev = []
    rolling_opp = []
    for i in range(n):
        start = max(0, i - window + 1)
        rolling_dev.append(np.mean(devs[start:i+1]))
        rolling_opp.append(np.mean(opp_devs[start:i+1]))

    x = list(range(1, n + 1))

    # Colors by trump
    trump_colors = {'H': '#e53935', 'D': '#e53935', 'S': '#333333', 'C': '#333333'}
    bar_colors = [trump_colors.get(t, '#888888') for t in trumps]

    # ── Plot 1: Per-round deviation bars ──
    ax1 = axes[0]
    ax1.clear()
    ax1.bar(x, devs, color=bar_colors, alpha=0.7, width=0.8, edgecolor='none')
    ax1.axhline(y=np.mean(devs), color='#e53935', linewidth=1.5, linestyle='--', label=f'Avg: {np.mean(devs):.1f}')
    if n >= 3:
        ax1.axhline(y=np.mean(opp_devs), color='#1565C0', linewidth=1.5, linestyle=':', label=f'Bots avg: {np.mean(opp_devs):.1f}')
    ax1.set_ylabel('Deviation', fontsize=10)
    ax1.set_title(f'Per-Round Deviation  ({n} rounds)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0.5, max(n + 0.5, 10))
    ax1.set_ylim(0, max(max(devs) + 5, 30))

    # Mark perfect rounds
    for i, d in enumerate(devs):
        if d == 0:
            ax1.annotate('*', (i + 1, 1), fontsize=14, ha='center', color='green', fontweight='bold')

    # ── Plot 2: Rolling average ──
    ax2 = axes[1]
    ax2.clear()
    ax2.plot(x, rolling_dev, color='#e53935', linewidth=2, label=f'Our bot (rolling {window})')
    ax2.plot(x, rolling_opp, color='#1565C0', linewidth=2, linestyle='--', label=f'Swisslos bots (rolling {window})')
    ax2.fill_between(x, rolling_dev, rolling_opp, alpha=0.1,
                      color='green' if rolling_dev[-1] < rolling_opp[-1] else 'red')
    ax2.set_ylabel('Avg Deviation', fontsize=10)
    ax2.set_title('Rolling Average - Us vs Swisslos Bots', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0.5, max(n + 0.5, 10))
    ax2.set_ylim(0, max(max(rolling_dev + rolling_opp) + 5, 25))
    ax2.axhline(y=5.8, color='#4CAF50', linewidth=1, linestyle=':', alpha=0.5)
    ax2.text(1, 6.2, 'Arena baseline (5.8)', fontsize=7, color='#4CAF50', alpha=0.7)

    # ── Plot 3: Declared vs Scored scatter ──
    ax3 = axes[2]
    ax3.clear()
    ax3.scatter(declared, scored, c=bar_colors, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    # Perfect line
    lim = max(max(declared + scored) + 10, 100)
    ax3.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Declared', fontsize=10)
    ax3.set_ylabel('Scored', fontsize=10)
    ax3.set_title('Declaration Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlim(-5, lim)
    ax3.set_ylim(-5, lim)
    ax3.set_aspect('equal', adjustable='box')

    # ── Plot 4: Stats panel ──
    ax4 = axes[3]
    ax4.clear()
    ax4.axis('off')

    perfect = sum(1 for d in devs if d == 0)
    good = sum(1 for d in devs if d <= 5)
    bad = sum(1 for d in devs if d > 10)

    stats_text = (
        f"Rounds:  {n}\n"
        f"Avg Dev:  {np.mean(devs):.1f}\n"
        f"Median:  {int(np.median(devs))}\n"
        f"Best:  {min(devs)}\n"
        f"Worst:  {max(devs)}\n"
        f"\n"
        f"Perfect (0):  {perfect}  ({100*perfect/n:.0f}%)\n"
        f"Good (<=5):  {good}  ({100*good/n:.0f}%)\n"
        f"Bad (>10):  {bad}  ({100*bad/n:.0f}%)\n"
        f"\n"
        f"Swisslos Bots:  {np.mean(opp_devs):.1f}\n"
    )

    winning = np.mean(devs) < np.mean(opp_devs)
    status = "BEATING BOTS" if winning else "BEHIND BOTS"
    status_color = '#4CAF50' if winning else '#e53935'

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5fa', edgecolor='#ddd'))

    ax4.text(0.5, 0.08, status, transform=ax4.transAxes,
             fontsize=14, fontweight='bold', ha='center', color=status_color)

    # Last 5 rounds ticker
    last5 = rounds[-5:]
    ticker = "Last 5: " + " ".join(
        f"{r['trump']}{r['deviation']}" for r in last5
    )
    ax4.text(0.5, 0.02, ticker, transform=ax4.transAxes,
             fontsize=8, ha='center', color='#888')

    fig.suptitle('Jass Bot Live Dashboard', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def main():
    parser = argparse.ArgumentParser(description='Jass Bot Live Dashboard')
    parser.add_argument('--dir', default=str(DEFAULT_EXPORT), help='Export Games directory')
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds')
    args = parser.parse_args()

    export_dir = Path(args.dir)
    if not export_dir.exists():
        print(f"Directory not found: {export_dir}")
        sys.exit(1)

    # Setup figure
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    axes = [
        fig.add_subplot(gs[0, :]),    # top: full width bar chart
        fig.add_subplot(gs[1, 0]),    # bottom-left: rolling avg
        fig.add_subplot(gs[1, 1]),    # bottom-right: scatter (will be split)
    ]
    # Split bottom-right into scatter + stats
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.05)
    axes[2] = fig.add_subplot(gs_inner[0, 0])
    axes.append(fig.add_subplot(gs_inner[0, 1]))

    # Initial draw
    update_dashboard(0, export_dir, fig, axes)

    # Auto-refresh
    ani = FuncAnimation(fig, update_dashboard, fargs=(export_dir, fig, axes),
                        interval=args.interval * 1000, cache_frame_data=False)

    plt.show()


if __name__ == '__main__':
    main()
