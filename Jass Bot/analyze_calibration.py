"""Analyze collected Swisslos data to validate and calibrate declaration model.

Compares old vs new estimate_hand_score(), analyzes declaration accuracy,
and finds optimal calibration parameters from real game data.
"""

from __future__ import annotations

import json
import sys
import os
import statistics

sys.path.insert(0, os.path.dirname(__file__))

from engine.cards import CARD_SUIT, CARD_STRENGTH, CARD_POINTS, points, strength


def estimate_hand_score_OLD(hand: list[int], trump_suit: int) -> int:
    """Original (broken) estimate — for comparison."""
    total_pts = sum(points(c, trump_suit) for c in hand)
    trump_cards = [c for c in hand if CARD_SUIT[c] == trump_suit]
    trump_count = len(trump_cards)
    base = total_pts // 2
    trump_bonus = trump_count * 5
    for c in trump_cards:
        trump_str = CARD_STRENGTH[c][1]
        if trump_str == 8:
            trump_bonus += 8
        elif trump_str == 7:
            trump_bonus += 5
    for c in hand:
        if CARD_SUIT[c] != trump_suit:
            non_trump_str = CARD_STRENGTH[c][0]
            if non_trump_str == 8:
                base += 5
    return max(0, min(157, base + trump_bonus))


from bot.heuristics import estimate_hand_score as estimate_hand_score_NEW


def load_training_data():
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'Jass Data', 'Processed', 'training_rounds.json'
    )
    if not os.path.exists(data_path):
        # Try bot-specific
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'Jass Data', 'Processed', 'training_rounds_bot.json'
        )
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('rounds', [])


def analyze():
    rounds = load_training_data()
    if not rounds:
        print("No training data found!")
        return

    print(f"Analyzing {len(rounds)} rounds from Swisslos data\n")
    print("=" * 75)

    # Collect all player hands with their declarations and actual scores
    old_errors = []
    new_errors = []
    decl_errors = []  # actual player declaration errors
    actual_scores = []
    old_estimates = []
    new_estimates = []

    for rnd in rounds:
        trump = rnd['trump']
        hands = rnd.get('hands', {})
        declarations = rnd.get('declarations', {})
        results = rnd.get('results', {})

        for pid_str, hand in hands.items():
            pid = int(pid_str) if isinstance(pid_str, str) else pid_str
            res = results.get(str(pid), results.get(pid, {}))
            if not res:
                continue

            actual = res.get('scored', None)
            declared = res.get('declared', None)
            if actual is None or declared is None:
                continue

            old_est = estimate_hand_score_OLD(hand, trump)
            new_est = estimate_hand_score_NEW(hand, trump)

            old_errors.append(abs(old_est - actual))
            new_errors.append(abs(new_est - actual))
            decl_errors.append(abs(declared - actual))
            actual_scores.append(actual)
            old_estimates.append(old_est)
            new_estimates.append(new_est)

    n = len(old_errors)
    print(f"\nTotal player-rounds analyzed: {n}")
    print(f"Actual scores: mean={statistics.mean(actual_scores):.1f}, "
          f"median={statistics.median(actual_scores):.0f}, "
          f"std={statistics.stdev(actual_scores):.1f}")

    print(f"\n{'Metric':<35} {'Old Estimate':>14} {'New Estimate':>14} {'Player Decl':>14}")
    print("-" * 80)

    for label, errors in [
        ("Mean absolute error", [old_errors, new_errors, decl_errors]),
    ]:
        print(f"{label:<35} {statistics.mean(errors[0]):>14.1f} "
              f"{statistics.mean(errors[1]):>14.1f} {statistics.mean(errors[2]):>14.1f}")

    for label, errors in [
        ("Median absolute error", [old_errors, new_errors, decl_errors]),
    ]:
        print(f"{label:<35} {statistics.median(errors[0]):>14.1f} "
              f"{statistics.median(errors[1]):>14.1f} {statistics.median(errors[2]):>14.1f}")

    # Bias analysis
    old_bias = [old_estimates[i] - actual_scores[i] for i in range(n)]
    new_bias = [new_estimates[i] - actual_scores[i] for i in range(n)]

    print(f"\n{'Bias (estimate - actual):':<35}")
    print(f"  Old: mean={statistics.mean(old_bias):+.1f}, median={statistics.median(old_bias):+.0f}")
    print(f"  New: mean={statistics.mean(new_bias):+.1f}, median={statistics.median(new_bias):+.0f}")

    # Distribution of errors
    print(f"\n{'Error distribution:':<35}")
    for threshold in [5, 10, 15, 20, 30]:
        old_pct = sum(1 for e in old_errors if e <= threshold) / n * 100
        new_pct = sum(1 for e in new_errors if e <= threshold) / n * 100
        decl_pct = sum(1 for e in decl_errors if e <= threshold) / n * 100
        print(f"  Within {threshold:>2} pts: old={old_pct:5.1f}%  new={new_pct:5.1f}%  "
              f"player={decl_pct:5.1f}%")

    # Scatter analysis: show worst cases
    print(f"\n{'Worst new estimate errors (>25):':<35}")
    cases = [(new_errors[i], actual_scores[i], new_estimates[i], old_estimates[i])
             for i in range(n)]
    cases.sort(reverse=True)
    for err, actual, new_est, old_est in cases[:10]:
        if err > 25:
            print(f"  Actual={actual:>3}, new_est={new_est:>3} (err={err:>2}), "
                  f"old_est={old_est:>3} (err={abs(old_est-actual):>2})")

    # Test specific expert example: Puur + Nell + 1 side Ace
    print(f"\n{'Expert validation:'}")
    # Herz trump, Puur=5, Nell=3, rest low trumps, Ecke Ace=17
    test_hand = [5, 3, 0, 1, 2, 17, 10, 19, 28]  # HU, H9, H6, H7, H8, EA, D7, S7, K7
    test_trump = 0  # herz
    old_v = estimate_hand_score_OLD(test_hand, test_trump)
    new_v = estimate_hand_score_NEW(test_hand, test_trump)
    print(f"  Puur+Nell+3 low trump + 1 Ace: old={old_v}, new={new_v}, expert≈79")


if __name__ == '__main__':
    analyze()
