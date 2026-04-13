# Jass Differenzler Bot

## What This Project Is

A Jass Differenzler bot targeting superhuman play on the Swisslos online platform. Differenzler is a Swiss card game where 4 players each get 9 cards from a 36-card deck, a trump suit is chosen randomly, each player secretly predicts how many points they'll score, then plays 9 tricks following standard Jass rules. The goal is to **minimize |predicted - actual points|**, NOT to maximize points. Total points per round = 157 (including 5 for last trick). The player with the lowest cumulative deviation across rounds wins.

---

## Current Best: 5.7 avg deviation

**PIMC 500w + C engine + neural declaration (MAE 2.7) + belief tracking + endgame depth 4**

| Metric | Value |
|--------|-------|
| Avg deviation | 5.7 (500 rounds) |
| Perfect rounds (0 dev) | 88/500 (17.6%) |
| Time per round | 123ms (parallel, 2× faster than sim-based) |
| C engine speed | 196K worlds/sec |
| Declaration time | <1ms (neural) vs ~200ms (simulation) |
| Declaration MAE | 2.7 pts (neural net, 800K training samples) |

### Performance Evolution

| Phase | What | Avg Dev | Speed |
|-------|------|---------|-------|
| 1 | Clean engine, correct rules, PIMC 25w, arena | 9.1 | ~5s/round |
| 2 | Median declaration, target-aware opponents | 7.4 | ~4s/round |
| 3 | Belief tracking, endgame solver (max^n depth 3) | 6.8 | ~3.5s/round |
| 4 | C engine 500w, bug fixes, calibration, endgame depth 4 | 5.8 | 36ms/round |
| Neural POC | 66% policy accuracy, but Python rollout too slow | 6.0 at 20w | 3s/round |
| 5a (live) | Swisslos deployment vs training bots (old calibration) | 13.5 live | ~12ms/move |
| 5b | Expert-calibrated estimate_hand_score + median declaration | 5.6 | 238ms/round |
| 5c | Neural declaration net (800K samples, MAE 2.7) | **5.7** | **123ms/round** |

### Neural Declaration Model (`bot/declaration_net.py`)
- **Architecture**: 72→256→256→128→1, BatchNorm + Dropout, 119K params
- **Training**: 800K samples from C engine simulation (30 worlds/hand), L1 loss (Bayes-optimal for median), OneCycleLR, early stopping. Trains in 53s on RTX 4070.
- **Input features** (72): hand binary (36) + trump one-hot (4) + trump card positions (9) + suit lengths (4) + suit points (4) + has_puur/nell/ace (3) + trump_count + void_count + total_points + trump_points + side_aces + high_trump_count + protected_aces + long_suits (8)
- **Performance**: MAE=2.70, bias=-0.21, within 5pts=83.9%, within 10pts=98.0%
- **Data generation**: `data/generate_declaration_data.py` — 200K deals in 119s using C engine
- **Model file**: `data/declaration_model.pt`

**Note:** Neural declaration is slightly better avg dev than simulation-based (5.7 vs 5.8) and 2× faster. Simulation-based has more perfect rounds (20.8% vs 17.6%) because it computes per-hand median exactly. Both available; server uses neural when model exists, falls back to simulation.

---

## Architecture

### Core Engine (`engine/`)
- **cards.py**: Card IDs 0-35 (`card_id = suit * 9 + value_index`). Suits: herz=0, ecke=1, schaufel=2, kreuz=3. Values: 6=0 through A=8. Precomputed lookup tables for points, strength, suit.
- **rules.py**: Full Jass legal move rules including undertrump restriction and Puur privilege. Trick winner, scoring with last-trick +5 bonus.
- **game_state.py**: Immutable-style GameState with hands, points, voids, declarations. Cheap copy for simulation.
- **game.py**: Round runner with Player protocol (`declare`, `play`, `notify_trick`). Validates moves, auto-detects voids.
- **fast.py**: Numpy vectorized operations (legacy, mostly superseded by C engine).

### C Engine (`engine_c/`)
- **jass_engine.c** (~980 lines): Bitmask hands (uint64_t), precomputed lookup tables. Functions:
  - `trick_winner`, `trick_points_sum` — core operations
  - `pick_card` — gap-based heuristic with last-trick +5 awareness
  - `pick_card_smart` — advanced heuristic (exists but unused — made things worse)
  - `sim_one_game`, `sim_remaining` — full game simulation with heuristic play
  - `simulate_games_batch` — N worlds in one call
  - `evaluate_moves_batch` — evaluate candidate cards across worlds (used by PIMC)
  - `rollout_game` — single mid-game rollout (used by ISMCTS)
  - `eg_solve`, `solve_endgame` — max^n endgame solver with transposition table
  - `eval_declaration`, `find_best_declaration`, `simulate_for_distribution` — declaration support
- **fast_engine.py**: ctypes wrapper. `FastEngine` class with `simulate_batch`, `solve_endgame`, `evaluate_moves`, `rollout`, `find_declaration`, `simulate_for_distribution`.
- **jass_engine.dll**: Compiled with `gcc -O3 -march=native -shared`. Not in git.
- **__init__.py**: Auto-adds `C:/msys64/ucrt64/bin` to DLL search path on Windows.

### Bot (`bot/`)
- **strategy.py**: PIMC engine. `StrategyEngine` with `ansage()` (declaration sweep) and `zug()` (move selection). C-accelerated path via `_c_engine.evaluate_moves()`. Passes actual opponent points to C engine. Python fallback with tree search.
- **declaration.py**: `DeclarationModel` — simulates N worlds, picks calibrated percentile (default pct=50, offset=0 = median). Uses C engine `simulate_for_distribution`. Python fallback available.
- **belief_tracker.py**: `BeliefTracker` — Bayesian posterior over opponent declarations. Gaussian prior, sigmoid likelihood updates on observed play (shedding/grabbing classification). Provides `sample_declaration`, `get_expected_declaration`, `get_declaration_range`.
- **endgame.py**: `EndgameSolver` — max^n exact solver for last N tricks. Uses C engine when available, Python fallback. Depth 4 default.
- **heuristics.py**: `pick_opponent_greedy`, `pick_opponent_target_aware`, `pick_sim_target_aware`, `estimate_hand_score`. Used for opponent modeling in simulation. `estimate_hand_score` uses expert Jass formula: trump face × 1.7 + trump count bonus + Puur/Nell extra + side aces (protected vs singleton) + void suit trumping bonus. Validated against 372 real game observations (MAE=12.4, bias=-1.0).
- **world_sampler.py**: `sample_worlds`, `sample_worlds_with_declarations`. Constrained generation respecting void knowledge. Integrates with belief tracker for opponent declaration sampling.
- **ismcts.py**: SO-ISMCTS implementation. `ISMCTSEngine` with `search()`. Tree walk with UCB1, compatibility filtering, C engine rollouts. **Tested but doesn't improve over PIMC for this game** — iterations spread too thin.
- **neural_net.py**: `DifferenzlerNet` — 256→256→128 trunk, policy head (→36 logits), value head (→1). 187K params. Feature size: 266.
- **neural_player.py**: `DirectNeuralPlayer` (pure network), `NeuralPIMCPlayer` (PIMC with neural rollouts via `NeuralRolloutEvaluator`).
- **neural_rollout.py**: `NeuralRolloutEvaluator` — batched neural play for player 0, gap heuristic for opponents, endgame solver at cutoff. Python loop is 80x slower than C.
- **train.py**: GPU training with mixed precision, OneCycleLR, early stopping. Trains in ~3.5s on RTX 4070.

### Arena (`arena/`)
- **arena.py**: `run_arena` with serial and parallel (multiprocessing) modes. `ArenaStats` tracks deviations, points, perfect rounds. Workers support PIMC and ISMCTS player configs.
- **players.py**: `RandomPlayer`, `HeuristicPlayer`, `PIMCPlayer`, `ISMCTSPlayer`. All implement the Player protocol.

### Data (`data/`)
- **generate_training_data.py**: Parallel PIMC self-play data generator. 5K rounds → 45K samples in 77s on 16 cores. Feature vector: 266 floats.
- **model_poc.pt**: Trained POC model (66% policy accuracy, 92% top-3, 4.5 value MAE).
- **poc_features.npy**, **poc_labels_card.npy**, **poc_labels_deviation.npy**, **poc_labels_utility.npy**: Training data arrays.

### Tests (`tests/`)
- **test_rules.py**: Card points, strength rankings, legal moves (undertrump, Puur privilege), trick winner, scoring. 30 tests.
- **test_engine.py**: Deal, round totals 157, deviations. 9 tests.
- **test_declaration.py**: Distribution, calibrated percentile, high/low hands. 8 tests.
- **test_belief_tracker.py**: Prior, updates, queries. 10 tests.
- **test_endgame.py**: 1/2/3-trick positions, activation threshold. 7 tests.
- **test_opponents.py**: Target-aware play, hand estimation, round totals. 9 tests.
- **test_integration.py**: Full rounds with beliefs + endgame. 4 tests.
- **test_c_engine.py**: C/Python cross-validation (100K tricks, batch sim, endgame, speed). 23 tests.
- **test_ismcts.py**: Node UCB, compatibility, time budget, player integration. 14 tests.
- **test_neural.py**: Encoding, model forward, save/load, player integration. 11 tests.
- **Total: 122 tests, all passing.**

### Server (`server.py`)
Flask API server that bridges the Tampermonkey extension to the PIMC engine.
- **Endpoints**: `/new-round` (declaration), `/play` (card selection), `/notify-trick` (belief updates), `/save` (write game data to disk), `/end-round`, `/status`
- Accepts Swisslos card codes (H6, DJ, SA) and converts to/from card IDs
- Maintains round state (beliefs, voids, target) between calls
- Auto-saves completed games to `../Jass Data/Export Games/`
- **Run**: `python server.py --port 5000 --worlds 200`
- **Requires**: `pip install flask flask-cors`

### Jass Data (`../Jass Data/`)
Data collection and analysis pipeline (sibling folder to Jass Bot).
- **swisslos_jass_collector.user.js**: Tampermonkey v7.1 — floating overlay with WATCH and BOT PLAY modes
- **process_collected_data.py**: Merges exports, validates, separates human vs bot games, outputs training-ready JSON
- **replay_analysis.py**: Replays real human games with bot in each seat, compares declaration + card play
- **Export Games/**: Raw JSON files auto-saved by server (one per completed round)
- **Processed/**: Clean output (training_rounds.json, training_rounds_human.json, training_rounds_bot.json)

### Other
- **main.py**: CLI entry point. `--gui`, `--arena N`, `--benchmark`, `--compare N`.
- **gui/assistant.py**: Pygame GUI for manual play assistance.
- **requirements.txt**: Dependencies (`pygame>=2.5, pytest>=7.0, flask, flask-cors`).
- **.gitignore**: Excludes caches, DLLs, env files.

---

## Building the C Engine

Requires GCC via MSYS2 UCRT64 (`C:/msys64/ucrt64/bin`):

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
gcc -O3 -march=native -shared -o engine_c/jass_engine.dll engine_c/jass_engine.c
```

`engine_c/__init__.py` auto-adds the UCRT64 DLL directory. Python falls back gracefully if DLL is missing.

---

## Key Rules Reference (Swisslos Differenzler)

### Card Points
| Card | Non-Trump | Trump |
|------|-----------|-------|
| 6    | 0         | 0     |
| 7    | 0         | 0     |
| 8    | 0         | 0     |
| 9    | 0         | 14    |
| 10   | 10        | 10    |
| Under (U/J) | 2  | 20    |
| Ober (O/Q)  | 3  | 3     |
| König (K)   | 4  | 4     |
| Ass (A)     | 11 | 11    |

**Total per round**: 152 card points + 5 last trick bonus = **157**

### Strength Rankings
- **Non-trump**: 6 < 7 < 8 < 9 < 10 < U < O < K < A (values 0-8)
- **Trump**: 6 < 7 < 8 < 10 < O < K < A < 9 (Nell) < U (Puur) (values 0-8)

### Card ID Layout
`card_id = suit_index * 9 + value_index` (0-35)
- Suits: herz=0, ecke=1, schaufel=2, kreuz=3
- Values: 6=0, 7=1, 8=2, 9=3, 10=4, U=5, O=6, K=7, A=8
- Example: Herz Under (Puur when herz is trump) = 0*9+5 = 5

### Legal Move Rules
1. Must follow led suit if possible
2. If trump is led: must play trump AND must overtrump if you can (Puur privilege: may keep Puur if it's the only card that can overtrump)
3. Cannot undertrump (play lower trump than table's highest) unless it's your only option
4. If you can't follow suit: free to play anything

---

## What Was Tried and Failed

### Smart Rollout Policy (failed)
Upgraded C `pick_card` to ~150 lines with trick-winning awareness, trump management, void creation, endgame awareness. **Made things worse** (5.8 → 9.1 avg dev). Root cause: changing the rollout policy distorted the score distribution used for declarations. The smarter heuristic made everyone (including opponents in simulation) play differently, breaking the consistency between declaration simulation and play evaluation. Reverted. The code exists as `pick_card_smart` in jass_engine.c but is unused.

### ISMCTS (failed)
Full SO-ISMCTS implementation with UCB1, compatibility filtering, C engine rollouts. At 1-5 second time budgets, **worse than PIMC** (6.3-8.0 vs 5.8). Root cause: 5000 iterations spread across 9 legal moves × 4 players = too thin per-branch signal. PIMC concentrates 500 independent world evaluations per candidate card. The tree overhead added cost without adding signal. Code exists in `bot/ismcts.py` and `ISMCTSPlayer` in players.py.

### Neutral Declaration Target (failed)
Tried using fixed target=78 (midpoint) or two-pass iteration to break the circular dependency between `est_target` and score distribution. Both made things worse. The `estimate_hand_score` heuristic is actually well-calibrated for the gap-based rollout. The "circular dependency" is mild and self-correcting through calibration (pct=45, off=-1).

### More PIMC Worlds (diminishing returns)
500w → 2000w gives only 5.9 → 5.8 (0.1 improvement). The heuristic rollout quality has converged — more samples of the same policy gives a more precise estimate of the wrong answer. The rollout quality is the bottleneck, not sample count.

### Neural Python Rollouts (too slow)
Neural network plays 66% correctly and 92% top-3. But Python simulation loop is 80x slower than C. NeuralPIMC 20w Python (6.0 avg dev, 3s/round) can't compete with PIMC 500w C (5.8 avg dev, 36ms/round). The C engine's quantity advantage outweighs the neural network's quality advantage. **The fix is ONNX export to C** — neural-quality rollouts at C speed.

---

## Current Bottleneck

The C engine's `pick_card` heuristic is a ~40-line gap-based function. It plays the same way whether it has Puur+Nell or garbage. The neural network (66% accuracy, 92% top-3) is much better but can't run inside the C simulation loop. Embedding the neural network in C via ONNX Runtime would give 500+ worlds with neural-quality rollouts in <100ms — the best of both worlds. This is the critical next step.

---

## Development Guidelines

- **Python 3.12+**, type hints everywhere
- **PyTorch** for neural networks, CUDA on RTX 4070
- Use `pytest` for testing — all 122 tests must pass
- Keep the engine **fast**: integer card IDs internally, C for hot paths
- When in doubt about a Jass rule, the Swisslos implementation is authoritative
- `main.py` CLI: `--gui`, `--arena N`, `--benchmark`, `--compare N`
