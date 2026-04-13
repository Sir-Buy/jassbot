# Jass Differenzler Bot — Roadmap to Superhuman

## The Goal

Build a Differenzler bot that consistently beats the strongest human players on Swisslos. "Superhuman" means:
- Average deviation < 3.5 over 1000+ rounds against real opponents
- Perfect prediction (deviation = 0) in 30%+ of rounds
- Adapts to real opponent behavior in real time

---

## Completed Phases

### Phase 1: Clean Architecture & Correct Engine (DONE)
- Modular Python codebase: engine, bot, arena, gui, tests
- Correct Jass rules: undertrump, Puur privilege, last-trick bonus
- Integer card IDs (0-35), copyable GameState
- Full round runner with Player protocol
- **Result: 9.1 avg dev**

### Phase 2: Improved Declaration & Opponents (DONE)
- Distribution-based declaration (median picker, Bayes-optimal)
- Target-aware opponent heuristics (gap-based play)
- World sampling with opponent declaration inference
- **Result: 7.4 avg dev**

### Phase 3: Belief Tracking & Endgame (DONE)
- Bayesian belief tracking over opponent declarations
- Max^n endgame solver (exact for last 3 tricks)
- Python optimization (flat tuples, inline hashing)
- **Result: 6.8 avg dev**

### Phase 4: C Engine, Optimization, Neural POC (DONE)
Compiled C engine (GCC -O3, 196K worlds/sec, 50-100x speedup):
- Bitmask hands (uint64_t), batch simulation, endgame solver
- `evaluate_moves_batch` for PIMC move evaluation
- `rollout_game` for single mid-game rollouts
- `pick_card` with last-trick +5 bonus awareness

Bug fixes:
- Opponent points now passed to C engine (was zeros before)
- Declaration calibration: percentile=45, offset=-1

Things tried that didn't work:
- Smart rollout policy in C (distorted score distributions, reverted)
- SO-ISMCTS (iterations too thin for 4-player game, worse than PIMC)
- Neutral declaration target (broke calibration, reverted)
- More worlds beyond 500 (diminishing returns, rollout quality is bottleneck)

Neural network POC:
- 187K-param net: 256→256→128 trunk, policy head (36), value head (1)
- Trained on 45K samples from 5K PIMC self-play rounds
- 66% policy accuracy, 92% top-3, 4.5 value MAE — all POC criteria passed
- Python neural rollouts work but 80x slower than C — needs ONNX export

**Result: 5.8 avg dev, 36ms/round, 122 tests passing**

---

## Phase 5: Real Game Data Collection & Live Play (ACTIVE)

### Status (2026-04-07)

**Data collected: 93 valid rounds (all bot training)**

Full pipeline operational:
- Tampermonkey userscript v7.1 with floating overlay dashboard
- Two modes: WATCH (observe human games) and BOT PLAY (auto-play vs Swisslos training bots)
- Flask server (`server.py`) bridges extension ↔ PIMC engine
- Auto-save to disk on every completed round
- Data processor separates human vs bot games

### Live Play Results (vs Swisslos Training Bots) — Before Fix

| Metric | Our PIMC Bot | Swisslos Bots |
|--------|:---:|:---:|
| Avg deviation | 13.5 | 12.6 |
| Win rate (rank 1) | 25% | — |
| Perfect rounds | 8% | 8% |

**Root cause identified and fixed:** `estimate_hand_score()` was systematically underestimating hand values (MAE=15.6, bias=-2.3). Rewritten with expert Jass formulas based on "trump×2 + aces" rule enhanced with trump count, void suit, and card protection analysis. New metrics: MAE=12.4, bias=-1.0.

Server declaration calibration also fixed: was percentile=40/offset=-3 (pushing declarations down), now percentile=50/offset=0 (median, Bayes-optimal).

### Arena Results (After Declaration Fix)

| Metric | Before | After |
|--------|--------|-------|
| Avg deviation | 5.8 | **5.6** |
| Perfect rounds | 41/200 (20.5%) | **51/200 (25.5%)** |

**Needs re-testing on Swisslos** to measure live improvement. Expected to close the 13.5→5.6 gap significantly.

### Replay Analysis (Bot on Real Human Hands)

Ran bot on 7 real human rounds (all 4 seats). Result:
- Human avg deviation: 10.4
- Bot avg deviation: 20.7 (WORSE) — **this was with the old broken estimate**
- Root cause: declaration wildly off on real hands (e.g., human declared 88, scored 90 → dev 2; bot would declare 31 → dev 59)
- Card play agreement: 54% (reasonable — many positions have multiple good options)

**Should re-run replay analysis with fixed declaration model.**

### Swisslos Protocol (Fully Reverse-Engineered)

- **Game URL**: `https://www.swisslos.ch/de/jass/differenzler/spielen.html`
- **Server**: `wss://gs.swisslos.ch/sock/?g=jassen&EIO=4&transport=websocket`
- **Engine.IO v4**: messages prefixed with `4` (message type), `2`/`3` = ping/pong
- **Card format**: `D10`, `HK`, `SA`, `CJ` (suit letter + value)

#### Key Messages

| ID | Direction | Format | Description |
|----|-----------|--------|-------------|
| 9022 | S→C | `{cards, trump, currR, maxR, players, ...}` | Round init (includes dealt hand) |
| 9029 | C→S | `{mt_id: -1}` | Join training game |
| 1 | C→S | `{points: "45", seqNr: N}` | Submit declaration |
| 5 | S→C | `{pCards: ["H9", ...]}` | Playable cards (your turn) |
| 6 | C→S | `{card: "H9", seqNr: N}` | Play a card |
| 7 | S→C | `{pos: N, card: "H9"}` | Card played by any player |
| 8 | S→C | `{pos: N, points: P}` | Trick winner |
| 10 | S→C | `{results: [{pos, callP, points, total, rank}]}` | Round result |
| 17 | S→C | same as 10 | Final round result |
| 9033 | C→S | `{t_id: N, mt_id: N}` | Join table as visitor |
| 9021 | S→C | `{t_id, mt_id, state, p}` | Table update (Differenzler) |
| 9035 | S→C | `{id[], title[], stake[], matchPot[]}` | Room list |

**Note:** Server `total` field in results is CUMULATIVE deviation across the match, not per-round.

### Architecture

```
Swisslos (browser)
  ↕ Tampermonkey userscript v7.1 (JS)
  ↕ HTTP fetch to localhost:5000
  ↕ Flask server (server.py)
  ↕ PIMC StrategyEngine + C engine
```

### Data Pipeline

```
Export Games/*.json → process_collected_data.py → Processed/
  ├── training_rounds.json       (all rounds)
  ├── training_rounds_human.json (human games only)
  ├── training_rounds_bot.json   (bot training only)
  ├── cleaned_games.json         (all matches)
  ├── cleaned_games_human.json
  └── cleaned_games_bot.json
```

### Next Steps for Data Collection
- Collect 500+ human game rounds (for declaration recalibration)
- Run bot overnight against training bots (for card play data volume)
- Build converter: training_rounds.json → numpy 266-feature arrays (for Phase 7)

---

## Phase 6: ONNX Neural Rollouts in C (NEXT)

The critical engineering step that will break through the 5.8 plateau.

### What
Export the trained PyTorch model to ONNX format. Embed ONNX Runtime inference in the C engine, replacing `pick_card` with neural network inference during rollouts.

### Why
- Current PIMC: 500 worlds × heuristic rollout = 5.8 avg dev, 36ms
- Neural rollout in Python: 20 worlds × neural play = 6.0 avg dev, 3000ms (80x slower)
- ONNX in C: 500 worlds × neural play = **estimated 4.5-5.0 avg dev**, ~100ms

The neural network plays 66% correctly vs the heuristic's much lower quality. At equal world counts, neural rollouts produce better score distributions and more accurate move evaluations. The Python overhead is the only thing preventing this improvement.

### How
1. Export `DifferenzlerNet` to ONNX: `torch.onnx.export(model, dummy_input, "model.onnx")`
2. Add ONNX Runtime C API to `jass_engine.c`
3. Replace `pick_card` calls for player 0 with ONNX inference
4. Batch player 0 decisions across worlds (one ONNX call per trick position)
5. Keep `pick_card` for opponent decisions (fast, no neural overhead)

### Target
- < 5.0 avg dev with 500 neural-guided worlds
- < 100ms per move (vs 36ms with heuristic — acceptable slowdown for better quality)

---

## Phase 7: Scaled Training with Real Data

### What
- Combine self-play data (100K rounds) with real Swisslos data (10K+ games)
- Train larger model or fine-tune existing one
- Real human data teaches patterns that self-play misses: opponent tendencies, common mistakes, meta-game

### Training Pipeline
1. Generate 100K PIMC self-play rounds (with ONNX-speed neural rollouts)
2. Add Swisslos collected data as additional training signal
3. Train with per-card PIMC utilities as soft labels (not just chosen card)
4. Self-play iteration: play → train → replace rollout policy → repeat 2-3 times

### Target
- < 4.5 avg dev against heuristic opponents
- Network policy accuracy > 75%

---

## Phase 8: Swisslos Deployment

### What
Full automation: Tampermonkey userscript reads game state, sends it to local Python/C bot, bot returns move, userscript plays it.

### Architecture
```
Swisslos (browser)
  ↕ Tampermonkey userscript (JS)
  ↕ WebSocket / HTTP to localhost
  ↕ Local Python server
  ↕ C engine + ONNX neural rollouts
```

### Components
1. **Tampermonkey userscript**: Intercepts WebSocket messages, extracts game state, displays bot recommendation or auto-plays
2. **Local server**: Flask/FastAPI on localhost, receives game state JSON, returns card to play
3. **Bot engine**: PIMC with ONNX neural rollouts, belief tracking, endgame solver

### Safety
- Randomize play timing (don't play instantly every time)
- Configurable auto-play vs suggestion-only mode
- Handle disconnects and reconnects gracefully

### Target
- Working end-to-end against real Swisslos opponents
- < 5.0 avg dev in live play (real opponents are harder than heuristic)

---

## Phase 9: Live Learning & Superhuman Play

### What
- Train on bot's own real games + observed opponent behavior
- Opponent modeling from real player patterns
- Meta-game: adjust risk based on cumulative standings
- Continuous improvement loop: play → collect data → retrain → deploy

### Opponent Modeling
- Cluster opponent play styles (aggressive, passive, erratic)
- Infer opponent declaration from play patterns (neural, not just sigmoid heuristic)
- Adapt strategy mid-game per opponent type

### Meta-Game
- Track cumulative standings across rounds
- If behind overall: higher-variance declarations (take risks)
- If leading: play safe, minimize max possible deviation

### Target
- < 3.5 avg dev against strong human opponents
- Genuine superhuman play: finds lines humans miss, exploits tendencies humans can't track

---

## Performance Trajectory

| Phase | Avg Dev | Perfect % | Speed | Key Unlock |
|-------|---------|-----------|-------|------------|
| 1 (engine) | 9.1 | ~8% | ~5s | Correct rules |
| 2 (declaration) | 7.4 | ~12% | ~4s | Median picker |
| 3 (beliefs) | 6.8 | ~15% | ~3.5s | Bayesian tracking |
| 4 (C engine) | 5.8 | 20% | 36ms | 196K worlds/sec |
| 5a (calibration) | 5.6 | 25.5% | 238ms | Expert hand estimation |
| 5b (neural decl) | **5.7** | **17.6%** | **123ms** | Neural declaration (MAE 2.7) |
| 6 (ONNX) | < 5.0 | ~25% | ~100ms | Neural C rollouts |
| 7 (scaled) | < 4.5 | ~28% | ~100ms | 100K training data |
| 8 (deployed) | < 5.0 live | ~20% live | ~200ms | Real opponents |
| 9 (superhuman) | < 3.5 | ~35% | ~200ms | Live learning |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Engine | C (via ctypes), Python fallback |
| Search | PIMC (500 worlds, C-accelerated) |
| Learning | PyTorch → ONNX Runtime |
| Training | GPU (RTX 4070), mixed precision |
| Data | Swisslos scraping + self-play |
| Deployment | Tampermonkey + localhost server |
| Platform | Windows 11, MSYS2 UCRT64 GCC |
| Testing | pytest, 122 tests |

---

## Key Principles (Validated by Experience)

1. **Speed enables everything.** The C engine's 196K worlds/sec is the foundation. Every improvement multiplies on top of it.

2. **Consistency matters more than intelligence.** A smarter rollout policy that distorts the score distribution is WORSE than a dumb one that's self-consistent. Declaration accuracy depends on simulation matching reality.

3. **The declaration is half the game.** A perfect card player with a bad declaration still loses. Declaration calibration (pct=45, off=-1) is as important as search improvements.

4. **PIMC beats ISMCTS for this game.** The 4-player imperfect-information structure with 9 legal moves makes ISMCTS iterations too thin. PIMC's independent world evaluations are more statistically robust.

5. **The rollout quality is the ceiling.** More worlds don't help once the heuristic has converged. The path to < 5.0 is replacing the heuristic with neural play, not adding more samples.

6. **Real data > self-play.** Self-play against heuristic opponents teaches a limited game. Real Swisslos data will expose patterns and strategies that don't exist in self-play.

7. **Deploy early, iterate live.** At 5.8 avg dev the bot already beats most human players. Real-game data is more valuable than theoretical improvements.
