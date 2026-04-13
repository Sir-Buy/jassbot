"""Head-to-head 4-round tournament: PIMC vs V7 vs V6 vs Heuristic."""

import sys, time, numpy as np, torch
sys.path.insert(0, '.')

from bot.strategy import StrategyEngine
from bot.heuristics import estimate_hand_score
from bot.declaration_net import NeuralDeclarationModel
from bot.train_v7_league import PlayNet
from env.vec_env import VecJassEnv
from ctypes import c_int

device = torch.device('cuda')

print('Loading engines...')

v7 = PlayNet(204, 512).to(device)
v7.load_state_dict(torch.load('checkpoints_v7/best.pt', map_location=device, weights_only=False)['play_net'])
v7.eval()
print('  V7 PlayNet (9.75 dev)')

v6 = PlayNet(204, 512).to(device)
v6.load_state_dict(torch.load('checkpoints_v6/best.pt', map_location=device, weights_only=False)['play_net'])
v6.eval()
print('  V6 PlayNet (10.1 dev)')

neural_decl = NeuralDeclarationModel()
print(f'  Neural decl: {"yes" if neural_decl.available else "no"}')

pimc = StrategyEngine(num_worlds=200, ansage_worlds=1000,
                      use_target_aware_opponents=True, endgame_depth=4)
print('  PIMC 200w + endgame depth 4')

print()
print('=' * 70)
print('HEAD-TO-HEAD: 4-ROUND MATCHES')
print('Seat 0: PIMC 200w + neural decl')
print('Seat 1: V7 RL + expert decl')
print('Seat 2: V6 RL + expert decl')
print('Seat 3: Heuristic + expert decl')
print('=' * 70)

env = VecJassEnv(n_envs=1, seed=0)
N_MATCHES = 100
N_ROUNDS = 4

names = ['PIMC-200w', 'V7-RL', 'V6-RL', 'Heuristic']
match_wins = [0, 0, 0, 0]
round_devs_all = [[], [], [], []]
match_ranks = [[], [], [], []]

t_start = time.perf_counter()

for match in range(N_MATCHES):
    total_devs = [0, 0, 0, 0]

    for rnd in range(N_ROUNDS):
        env._seed_counter += 1
        env.lib.rl_game_init(env._ptr_at(0), env._seed_counter)
        decl_obs = env.reset()
        expert_decls = env.expert_declare()[0]

        # Get hand/trump for seat 0
        hand0_bin = decl_obs[0, 0, :36]
        hand0 = list(np.where(hand0_bin > 0.5)[0])
        trump = int(np.argmax(decl_obs[0, 0, 36:40]))

        declarations = expert_decls.copy()
        if neural_decl.available:
            declarations[0] = neural_decl.declare(hand0, trump)

        env.set_declarations(declarations.reshape(1, 4))

        # PIMC state
        pimc.reset()
        pimc_hand = list(hand0)
        pimc_played = []
        pimc_points = 0
        pimc_trick_num = 0
        pimc_target = int(declarations[0])

        for step in range(36):
            phases = env.get_phases()
            if (phases == 2).all():
                break

            obs_np, legal_np, cur_players = env.get_play_obs()
            cur_p = cur_players[0]
            legal_cards = list(np.where(legal_np[0] > 0.5)[0])

            if cur_p == 0:
                # PIMC
                if len(legal_cards) == 1:
                    action = legal_cards[0]
                else:
                    action = pimc.zug(
                        hand=pimc_hand, trump=trump, target=pimc_target,
                        current_points=pimc_points, table_cards=[],
                        leader=0, played_cards=pimc_played,
                        trick_number=pimc_trick_num,
                    )
                    if action is None or action not in legal_cards:
                        action = legal_cards[0]
                if action in pimc_hand:
                    pimc_hand.remove(action)
                pimc_played.append(action)
                env.lib.rl_step(env._ptr_at(0), c_int(int(action)))

            elif cur_p == 1:
                # V7
                obs_t = torch.as_tensor(obs_np, device=device)
                legal_t = torch.as_tensor(legal_np, device=device)
                with torch.no_grad():
                    logits, _ = v7(obs_t, legal_t)
                    action = int(logits[0].argmax())
                if action not in legal_cards:
                    action = legal_cards[0]
                pimc_played.append(action)
                env.lib.rl_step(env._ptr_at(0), c_int(int(action)))

            elif cur_p == 2:
                # V6
                obs_t = torch.as_tensor(obs_np, device=device)
                legal_t = torch.as_tensor(legal_np, device=device)
                with torch.no_grad():
                    logits, _ = v6(obs_t, legal_t)
                    action = int(logits[0].argmax())
                if action not in legal_cards:
                    action = legal_cards[0]
                pimc_played.append(action)
                env.lib.rl_step(env._ptr_at(0), c_int(int(action)))

            else:
                # Heuristic
                card = env.lib.rl_pick_heuristic(env._ptr_at(0))
                pimc_played.append(card)
                env.lib.rl_step(env._ptr_at(0), c_int(int(card)))

            # Track PIMC points after each trick completes
            pts_now = env.get_points()[0]
            pimc_points = int(pts_now[0])
            new_trick = pimc_points != int(pts_now[0]) if step > 0 else False
            # Simple trick counting: every 4 steps
            pimc_trick_num = step // 4

        # Round results
        points = env.get_points()[0]
        for p in range(4):
            dev = abs(int(declarations[p]) - int(points[p]))
            total_devs[p] += dev
            round_devs_all[p].append(dev)

    # Rank by total deviation
    ranked = sorted(range(4), key=lambda p: total_devs[p])
    winner = ranked[0]
    match_wins[winner] += 1
    for p in range(4):
        rank = ranked.index(p) + 1
        match_ranks[p].append(rank)

    if (match + 1) % 10 == 0:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (match + 1) * (N_MATCHES - match - 1)
        w = " | ".join(f"{names[i]}={match_wins[i]}" for i in range(4))
        print(f'  Match {match+1}/{N_MATCHES} ({elapsed:.0f}s, ETA {eta:.0f}s) | Wins: {w}')

elapsed = time.perf_counter() - t_start
print()
print('=' * 70)
print(f'RESULTS ({N_MATCHES} matches x {N_ROUNDS} rounds, {elapsed:.0f}s)')
print('=' * 70)
print()
fmt = f'{"Engine":<15} {"Wins":>6} {"Win%":>6} {"AvgRank":>8} {"AvgDev":>7} {"MedDev":>7} {"Perf%":>7}'
print(fmt)
print('-' * 70)
for p in range(4):
    devs = np.array(round_devs_all[p])
    ranks = np.array(match_ranks[p])
    win_pct = 100 * match_wins[p] / N_MATCHES
    avg_rank = ranks.mean()
    avg_dev = devs.mean()
    med_dev = np.median(devs)
    perf = 100 * (devs == 0).sum() / len(devs)
    print(f'{names[p]:<15} {match_wins[p]:>6} {win_pct:>5.1f}% {avg_rank:>8.2f} {avg_dev:>7.1f} {med_dev:>7.0f} {perf:>6.1f}%')

print()
print('HEAD-TO-HEAD (who finishes ahead in match ranking):')
for i in range(4):
    for j in range(i + 1, 4):
        i_ahead = sum(1 for m in range(N_MATCHES) if match_ranks[i][m] < match_ranks[j][m])
        j_ahead = sum(1 for m in range(N_MATCHES) if match_ranks[j][m] < match_ranks[i][m])
        ties = N_MATCHES - i_ahead - j_ahead
        print(f'  {names[i]:>12} vs {names[j]:<12}: {i_ahead:>3}-{j_ahead:<3} (ties: {ties})')
