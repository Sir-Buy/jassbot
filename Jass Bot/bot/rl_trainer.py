"""RL Self-Play trainer for Differenzler Jass.

4 copies of a neural network play each other. Each player only sees
their own information set. REINFORCE with baseline trains the policy.

Usage:
    python -m bot.rl_trainer --rounds 50000 --eval-every 2000
"""

from __future__ import annotations

import os
import sys
import random
import time
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.cards import (
    CARD_SUIT, CARD_VALUE, CARD_STRENGTH, CARD_POINTS,
    NUM_CARDS, NUM_VALUES, points as card_points, strength as card_strength,
)
from engine.rules import legal_moves

FEATURE_SIZE = 266


# ═══════════════════════════════════════════════════════════════════
#  Network (same architecture as existing DifferenzlerNet)
# ═══════════════════════════════════════════════════════════════════

class PolicyValueNet(nn.Module):
    """Policy + value network for Differenzler card play."""

    def __init__(self, input_size: int = FEATURE_SIZE):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, 36)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, legal_mask=None):
        h = self.trunk(x)
        logits = self.policy_head(h)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), -1e9)
        value = self.value_head(h).squeeze(-1)
        return logits, value


# ═══════════════════════════════════════════════════════════════════
#  State encoding (same as existing generate_training_data.py)
# ═══════════════════════════════════════════════════════════════════

def encode_state(
    hand: list[int],
    played_cards: set[int],
    trick_cards: list[tuple[int, int]],
    trump: int,
    legal: list[int],
    trick_number: int,
    my_points: int,
    opp_points: list[int],
    my_target: int,
    voids: dict[int, set[int]],
) -> np.ndarray:
    f = np.zeros(FEATURE_SIZE, dtype=np.float32)
    off = 0

    for c in hand:
        f[off + c] = 1.0
    off += 36

    for c in played_cards:
        f[off + c] = 1.0
    off += 36

    for _, c in trick_cards:
        f[off + c] = 1.0
    off += 36

    for c in range(NUM_CARDS):
        if CARD_SUIT[c] == trump:
            f[off + c] = 1.0
    off += 36

    for c in legal:
        f[off + c] = 1.0
    off += 36

    if trick_cards:
        lead_suit = CARD_SUIT[trick_cards[0][1]]
        best_c = trick_cards[0][1]
        for _, c in trick_cards[1:]:
            cs, bs = CARD_SUIT[c], CARD_SUIT[best_c]
            if cs == trump and bs != trump:
                best_c = c
            elif cs == bs and card_strength(c, trump) > card_strength(best_c, trump):
                best_c = c
        f[off + best_c] = 1.0
    off += 36

    all_seen = played_cards | {c for _, c in trick_cards} | set(hand)
    for s in range(4):
        remaining = sum(1 for c in range(NUM_CARDS) if CARD_SUIT[c] == s and c not in all_seen)
        for slot in range(4):
            f[off + s * 4 + slot] = remaining / 9.0
    off += 16

    for oi, p in enumerate([1, 2, 3]):
        for s in range(4):
            if s in voids.get(p, set()):
                f[off + oi * 4 + s] = 1.0
    off += 12

    gap = my_target - my_points
    total_pts = my_points + sum(opp_points)
    n_on_table = len(trick_cards)
    trump_in_hand = sum(1 for c in hand if CARD_SUIT[c] == trump)
    puur_id = trump * NUM_VALUES + 5
    nell_id = trump * NUM_VALUES + 3
    trump_ace_id = trump * NUM_VALUES + 8

    scalars = [
        trick_number / 8.0,
        my_points / 157.0,
        opp_points[0] / 157.0,
        opp_points[1] / 157.0,
        opp_points[2] / 157.0,
        my_target / 157.0,
        gap / 157.0,
        abs(gap) / 157.0,
        1.0 if gap > 0 else 0.0,
        1.0 if gap < 0 else 0.0,
        1.0 if gap == 0 else 0.0,
        (9 - trick_number) / 9.0,
        len(hand) / 9.0,
        1.0 if n_on_table == 0 else 0.0,
        n_on_table / 3.0,
        sum(card_points(c, trump) for _, c in trick_cards) / 60.0 if trick_cards else 0.0,
        total_pts / 157.0,
        trump_in_hand / 9.0,
        1.0 if puur_id in hand else 0.0,
        1.0 if nell_id in hand else 0.0,
        1.0 if trump_ace_id in hand else 0.0,
        len(legal) / 9.0,
    ]
    for i, v in enumerate(scalars):
        f[off + i] = v

    return f


# ═══════════════════════════════════════════════════════════════════
#  Game simulation with neural players
# ═══════════════════════════════════════════════════════════════════

def trick_winner(trick: list[tuple[int, int]], lead_suit: int, trump: int) -> int:
    best_p, best_c = trick[0]
    for p, c in trick[1:]:
        cs, bs = CARD_SUIT[c], CARD_SUIT[best_c]
        if cs == trump and bs != trump:
            best_p, best_c = p, c
        elif cs == bs and card_strength(c, trump) > card_strength(best_c, trump):
            best_p, best_c = p, c
    return best_p


def deal(rng: random.Random) -> tuple[int, dict[int, list[int]]]:
    cards = list(range(36))
    rng.shuffle(cards)
    return rng.randint(0, 3), {
        0: sorted(cards[0:9]),
        1: sorted(cards[9:18]),
        2: sorted(cards[18:27]),
        3: sorted(cards[27:36]),
    }


def declare_from_hand(hand: list[int], trump: int) -> int:
    """Simple heuristic declaration — will be replaced by learned declaration."""
    trump_cards = [c for c in hand if CARD_SUIT[c] == trump]
    trump_pts = sum(card_points(c, trump) for c in trump_cards)
    base = int(trump_pts * 1.7)
    tc = len(trump_cards)
    if tc >= 5:
        base = max(base, 25 + tc * 5)
    elif tc >= 3:
        base += (tc - 2) * 3
    for c in trump_cards:
        if CARD_STRENGTH[c][1] == 8: base += 4
        elif CARD_STRENGTH[c][1] == 7: base += 2
    for c in hand:
        if CARD_SUIT[c] != trump and CARD_STRENGTH[c][0] == 8:
            base += 9 if sum(1 for x in hand if CARD_SUIT[x] == CARD_SUIT[c]) >= 2 else 5
    return max(0, min(157, base))


@torch.no_grad()
def play_game_batch(
    net: PolicyValueNet,
    device: torch.device,
    batch_size: int,
    rng: random.Random,
    temperature: float = 1.0,
) -> dict:
    """Play a batch of games with neural policy. Vectorized GPU inference.

    Key optimization: within each trick position, ALL games' decisions for
    that position are batched into a single GPU forward pass.
    """
    net.eval()

    # Deal all games
    games = []
    for _ in range(batch_size):
        trump, hands = deal(rng)
        decls = {p: declare_from_hand(hands[p], trump) for p in range(4)}
        games.append({
            "trump": trump,
            "hands": {p: list(h) for p, h in hands.items()},
            "decls": decls,
            "points": [0, 0, 0, 0],
            "played": set(),
            "voids": {0: set(), 1: set(), 2: set(), 3: set()},
            "leader": 0,
            "trick": [],
            "lead_suit": None,
        })

    # Collect trajectories: flat list of (game_idx, player, state, action, log_prob, value)
    all_traj_entries = []

    for trick_num in range(9):
        for seat_offset in range(4):
            # Collect states for ALL games where this seat needs to act
            batch_states = []
            batch_masks = []
            batch_indices = []  # (game_idx, player, legal_cards)

            for gi, g in enumerate(games):
                p = (g["leader"] + seat_offset) % 4
                hand = g["hands"][p]
                if not hand:
                    continue

                legal = legal_moves(hand, g["lead_suit"], g["trump"],
                                    g["trick"] if g["trick"] else None)
                if not legal:
                    legal = list(hand)

                opp_pts = []
                for oi in range(1, 4):
                    opp_pts.append(g["points"][(p + oi) % 4])

                state = encode_state(
                    hand, g["played"], g["trick"], g["trump"], legal,
                    trick_num, g["points"][p], opp_pts, g["decls"][p], g["voids"],
                )

                mask = np.zeros(36, dtype=np.float32)
                for c in legal:
                    mask[c] = 1.0

                batch_states.append(state)
                batch_masks.append(mask)
                batch_indices.append((gi, p, legal))

            if not batch_states:
                continue

            # SINGLE batched GPU forward pass for all games at this position
            states_t = torch.tensor(np.array(batch_states), device=device)
            masks_t = torch.tensor(np.array(batch_masks), device=device)
            logits, values = net(states_t, masks_t)
            logits = logits / max(temperature, 0.01)

            # Sample actions
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            # Apply actions to each game
            actions_cpu = actions.cpu().numpy()
            log_probs_cpu = log_probs.cpu().numpy()
            values_cpu = values.cpu().numpy()

            for idx, (gi, p, legal) in enumerate(batch_indices):
                g = games[gi]
                card = int(actions_cpu[idx])
                if card not in legal:
                    card = legal[0]

                all_traj_entries.append({
                    "game_idx": gi,
                    "player": p,
                    "state": batch_states[idx],
                    "action": card,
                    "log_prob": float(log_probs_cpu[idx]),
                    "value": float(values_cpu[idx]),
                })

                g["hands"][p].remove(card)
                g["trick"].append((p, card))
                if g["lead_suit"] is None and g["trick"]:
                    g["lead_suit"] = CARD_SUIT[g["trick"][0][1]]

                # Detect voids
                if len(g["trick"]) > 1 and CARD_SUIT[card] != CARD_SUIT[g["trick"][0][1]]:
                    g["voids"][p].add(CARD_SUIT[g["trick"][0][1]])

        # Resolve all tricks
        for g in games:
            if len(g["trick"]) == 4:
                winner = trick_winner(g["trick"], g["lead_suit"], g["trump"])
                pts = sum(card_points(c, g["trump"]) for _, c in g["trick"])
                if trick_num == 8:
                    pts += 5
                g["points"][winner] += pts
                g["leader"] = winner
                for _, c in g["trick"]:
                    g["played"].add(c)
            g["trick"] = []
            g["lead_suit"] = None

    # Build trajectory structure and compute rewards
    trajs = {p: [[] for _ in range(batch_size)] for p in range(4)}
    for entry in all_traj_entries:
        trajs[entry["player"]][entry["game_idx"]].append(entry)

    rewards = {p: [] for p in range(4)}
    deviations = {p: [] for p in range(4)}
    for gi, g in enumerate(games):
        for p in range(4):
            dev = abs(g["decls"][p] - g["points"][p])
            rewards[p].append(-dev / 157.0)
            deviations[p].append(dev)

    return {
        "trajs": trajs,
        "rewards": rewards,
        "deviations": deviations,
        "games": games,
    }


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_on_batch(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_result: dict,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
) -> dict:
    """REINFORCE with baseline on one batch of games."""
    net.train()

    all_states = []
    all_actions = []
    all_log_probs = []
    all_advantages = []
    all_returns = []

    trajs = batch_result["trajs"]
    rewards = batch_result["rewards"]
    batch_size = len(rewards[0])

    for p in range(4):
        for game_idx in range(batch_size):
            game_reward = rewards[p][game_idx]
            steps = trajs[p][game_idx]
            for step in steps:
                all_states.append(step["state"])
                all_actions.append(step["action"])
                all_log_probs.append(step["log_prob"])
                all_advantages.append(game_reward - step["value"])
                all_returns.append(game_reward)

    if not all_states:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0}

    states_t = torch.tensor(np.array(all_states), device=device)
    actions_t = torch.tensor(all_actions, dtype=torch.long, device=device)
    advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)

    # Normalize advantages
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # Forward pass
    legal_mask = torch.zeros(len(states_t), 36, device=device)
    for i, s in enumerate(all_states):
        for c in range(36):
            if s[4 * 36 + c] > 0.5:  # legal_mask is at offset 4*36
                legal_mask[i, c] = 1.0

    logits, values = net(states_t, legal_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Policy loss (REINFORCE)
    policy_loss = -(action_log_probs * advantages_t.detach()).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns_t)

    # Entropy bonus
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Total loss
    loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": loss.item(),
    }


# ═══════════════════════════════════════════════════════════════════
#  Evaluation against heuristic opponents
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_vs_heuristic(
    net: PolicyValueNet,
    device: torch.device,
    num_rounds: int = 200,
    rng: random.Random | None = None,
) -> dict:
    """Evaluate neural player 0 vs 3 heuristic opponents."""
    from bot.heuristics import (
        pick_opponent_target_aware, estimate_hand_score,
    )

    net.eval()
    r = rng or random.Random(999)
    deviations = []

    for _ in range(num_rounds):
        trump, hands = deal(r)
        decls = {p: declare_from_hand(hands[p], trump) for p in range(4)}
        points = [0, 0, 0, 0]
        played = set()
        voids = {0: set(), 1: set(), 2: set(), 3: set()}
        leader = 0

        for trick_num in range(9):
            trick = []
            lead_suit = None

            for i in range(4):
                p = (leader + i) % 4
                hand = hands[p]
                if not hand:
                    continue

                legal = legal_moves(hand, lead_suit, trump,
                                    trick if trick else None)
                if not legal:
                    legal = list(hand)

                if p == 0:
                    # Neural player
                    opp_pts = [points[1], points[2], points[3]]
                    state = encode_state(
                        hand, played, trick, trump, legal,
                        trick_num, points[0], opp_pts, decls[0], voids,
                    )
                    state_t = torch.tensor(state, device=device).unsqueeze(0)
                    lm = torch.zeros(1, 36, device=device)
                    for c in legal:
                        lm[0, c] = 1.0
                    logits, _ = net(state_t, lm)
                    card = legal[logits.squeeze(0)[legal].argmax().item()]
                else:
                    # Heuristic opponent
                    card = pick_opponent_target_aware(
                        hand, trick, lead_suit, trump,
                        decls[p], points[p], trick_num,
                    )

                hand.remove(card)
                trick.append((p, card))
                if lead_suit is None:
                    lead_suit = CARD_SUIT[trick[0][1]]

                if len(trick) > 1 and CARD_SUIT[card] != CARD_SUIT[trick[0][1]]:
                    voids[p].add(CARD_SUIT[trick[0][1]])

            winner = trick_winner(trick, lead_suit, trump)
            pts = sum(card_points(c, trump) for _, c in trick)
            if trick_num == 8:
                pts += 5
            points[winner] += pts
            leader = winner

            for _, c in trick:
                played.add(c)

        dev = abs(decls[0] - points[0])
        deviations.append(dev)

    avg = sum(deviations) / len(deviations)
    perfect = sum(1 for d in deviations if d == 0)
    return {
        "avg_dev": avg,
        "perfect": perfect,
        "n": num_rounds,
        "deviations": deviations,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RL Self-Play Trainer")
    parser.add_argument("--rounds", type=int, default=100000, help="Total self-play rounds")
    parser.add_argument("--batch", type=int, default=256, help="Games per training batch")
    parser.add_argument("--eval-every", type=int, default=5000, help="Evaluate every N rounds")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0, help="Exploration temperature")
    parser.add_argument("--temp-decay", type=float, default=0.995, help="Temperature decay per batch")
    parser.add_argument("--entropy", type=float, default=0.02, help="Entropy bonus coefficient")
    parser.add_argument("--output", default="data/rl_policy.pt")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net = PolicyValueNet().to(device)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in net.parameters())
    print(f"Network: {param_count:,} parameters")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = random.Random(42)

    total_rounds = 0
    best_avg_dev = float("inf")
    temperature = args.temperature

    # Initial evaluation
    print("\n=== Initial evaluation (random policy) ===")
    eval_result = evaluate_vs_heuristic(net, device, 200, random.Random(999))
    print(f"  Avg dev: {eval_result['avg_dev']:.1f}, "
          f"Perfect: {eval_result['perfect']}/{eval_result['n']}")

    t0 = time.time()
    batch_num = 0

    while total_rounds < args.rounds:
        batch_num += 1

        # Self-play
        batch_result = play_game_batch(
            net, device, args.batch, rng, temperature=temperature,
        )

        # Train
        train_stats = train_on_batch(
            net, optimizer, device, batch_result,
            entropy_coeff=args.entropy,
        )

        total_rounds += args.batch
        temperature = max(0.1, temperature * args.temp_decay)

        # Logging
        all_devs = []
        for p in range(4):
            all_devs.extend(batch_result["deviations"][p])
        sp_avg = sum(all_devs) / len(all_devs) if all_devs else 0

        if batch_num % 10 == 0:
            elapsed = time.time() - t0
            rps = total_rounds / elapsed
            print(f"  Batch {batch_num:>5d} | rounds={total_rounds:>7d} | "
                  f"sp_dev={sp_avg:.1f} | loss={train_stats['total_loss']:.4f} | "
                  f"entropy={train_stats['entropy']:.3f} | temp={temperature:.3f} | "
                  f"{rps:.0f} r/s")

        # Evaluate
        if total_rounds % args.eval_every < args.batch:
            print(f"\n=== Evaluation at {total_rounds} rounds ===")
            eval_result = evaluate_vs_heuristic(net, device, 200, random.Random(999))
            avg_dev = eval_result["avg_dev"]
            perfect = eval_result["perfect"]
            print(f"  vs Heuristic: avg_dev={avg_dev:.1f}, "
                  f"perfect={perfect}/{eval_result['n']}")

            if avg_dev < best_avg_dev:
                best_avg_dev = avg_dev
                torch.save({
                    "model_state_dict": net.state_dict(),
                    "avg_dev": avg_dev,
                    "rounds_trained": total_rounds,
                }, args.output)
                print(f"  NEW BEST: {avg_dev:.1f} — saved to {args.output}")
            print()

    # Final evaluation
    print(f"\n=== Final evaluation ({total_rounds} rounds trained) ===")
    eval_result = evaluate_vs_heuristic(net, device, 500, random.Random(999))
    print(f"  vs Heuristic (500 rounds): avg_dev={eval_result['avg_dev']:.1f}, "
          f"perfect={eval_result['perfect']}/{eval_result['n']}")

    torch.save({
        "model_state_dict": net.state_dict(),
        "avg_dev": eval_result["avg_dev"],
        "rounds_trained": total_rounds,
    }, args.output)
    print(f"  Saved to {args.output}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_rounds/total_time:.0f} rounds/sec)")


if __name__ == "__main__":
    main()
