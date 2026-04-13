# Jass Differenzler RL Bot — Claude Code Handoff

## Mission

Build a Differenzler Jass bot that **beats human players** using reinforcement learning with self-play. Target: average deviation < 5.0 against real opponents (humans average 7-8, current PIMC bot gets 5.7 in self-play but 14.9 against real Swisslos opponents).

## Why RL (and why everything else failed)

The project has been through multiple iterations:
- **PIMC + heuristic rollouts**: 5.7 avg dev in self-play, but 14.9 live because the heuristic opponent model doesn't match real play
- **ISMCTS**: Tried, didn't improve over PIMC for this game
- **Neural network (supervised on self-play data)**: 66% accuracy but trained on fake opponent behavior
- **Smart rollouts, belief tracking, endgame solver**: All marginal improvements on a fundamentally flawed opponent model

**The core problem**: Every approach so far simulates opponents with heuristics that don't match reality. RL with self-play sidesteps this entirely — the opponents ARE the learning agent, so the policy naturally adapts to strong, realistic play.

## Game Rules (Complete)

### Basics
- 4 players, 36 Swiss cards (4 suits × 9 values: 6,7,8,9,10,U,O,K,A)
- 9 cards per player, 9 tricks per round
- Trump determined randomly each round
- **Objective**: Each player declares (predicts) their score before playing. Penalty = |declared - actual|. Lowest cumulative penalty wins.
- Total points per round: 152 card points + 5 last trick bonus = 157

### Card Points
**Non-trump**: 6=0, 7=0, 8=0, 9=0, 10=10, U(Under/Bube)=2, O(Ober/Dame)=3, K(König)=4, A(Ass)=11
**Trump**: 6=0, 7=0, 8=0, 9(Nell)=14, 10=10, U(Puur)=20, O=3, K=4, A=11

### Strength Rankings
**Non-trump** (low→high): 6 < 7 < 8 < 9 < 10 < U < O < K < A
**Trump** (low→high): 6 < 7 < 8 < 10 < O < K < A < 9(Nell) < U(Puur)

### Trick Rules (suit-following with restrictions)
1. Must follow led suit if possible
2. Trump may always be played (even if you have the led suit — this is "einstechen")
3. If trump is led: must play trump. Exception: if your only trump is Puur (trump Under), you may play any card
4. **Undertrump restriction**: If a trump has already been played in the trick and you can't follow the led suit, you may only play a trump that is HIGHER than the current highest trump on the table. If you can't beat it with trump, you may play any card.
5. If you can't follow suit and no trump has been played: play anything

### Round Flow
1. Cards dealt (9 each)
2. Trump revealed
3. Each player declares (simultaneously/hidden)
4. 9 tricks played (counter-clockwise)
5. Penalty = |declared - actual score|

## Architecture: RL Self-Play

### Overview

```
┌─────────────────────────────────────────────────────┐
│                  TRAINING LOOP                       │
│                                                      │
│  C Engine (fast game sim)                            │
│       ↕ ctypes                                       │
│  Python Vectorized Env (256+ parallel games)         │
│       ↕                                              │
│  PPO Agent (PyTorch)                                 │
│    ├── Shared backbone                               │
│    ├── Declaration head → predict score 0-157        │
│    ├── Play policy head → card probabilities         │
│    └── Value head → expected deviation               │
│                                                      │
│  Self-play: 4 copies of same network play each other │
│  Reward at round end: -|declared - actual_score|     │
└─────────────────────────────────────────────────────┘
```

### Component 1: C Game Engine (SPEED CRITICAL)

This is the bottleneck. Must be blazing fast. The existing project has a C engine doing 196K worlds/sec for PIMC — we need similar or better for RL rollouts.

**Data representation:**
```c
typedef uint64_t hand_t;  // 36 cards as bitmask, fits in uint64_t

// Card ID 0-35: suit = id / 9, value = id % 9
// Value encoding: 0=6, 1=7, 2=8, 3=9, 4=10, 5=Under, 6=Ober, 7=König, 8=Ass

// Precomputed lookup tables
int CARD_POINTS[36][2];     // [card_id][is_trump]
int CARD_STRENGTH[36][2];   // [card_id][is_trump] → ranking for comparison
hand_t SUIT_MASKS[4];       // bitmask for each suit
```

**Core functions needed:**
```c
void init_tables(void);
hand_t legal_moves(hand_t hand, int lead_suit, int trump, int highest_trump_played);
int trick_winner(int cards[4], int lead_suit, int trump);  // returns player index 0-3
int trick_points(int cards[4], int trump);  // sum of points
```

**Vectorized game simulation for RL:**
```c
// Run N games in parallel, return observations and rewards
// This is what the Python env calls each step
typedef struct {
    hand_t hands[4];           // 4 players' hands
    int trump;                 // 0-3
    int declarations[4];       // declared scores
    int points[4];             // accumulated points
    int trick_cards[4];        // cards played in current trick (-1 if not yet played)
    int trick_count;           // which trick (0-8)
    int current_player;        // whose turn
    int leader;                // who leads this trick
    int cards_in_trick;        // 0-3 cards played so far
    int phase;                 // 0=declare, 1=play
} GameState;

void game_init(GameState* g);                    // deal cards, set trump
void game_set_declaration(GameState* g, int player, int value);
int game_step(GameState* g, int card_id);        // play a card, returns 0 if ok
void game_get_observation(GameState* g, int player, float* obs);  // fill observation vector
int game_is_done(GameState* g);                  // round finished?
float game_get_reward(GameState* g, int player); // -|decl - actual|
```

**Python ctypes wrapper:**
```python
# Vectorized environment running 256+ games simultaneously
class VecJassEnv:
    def __init__(self, n_envs=256):
        self.lib = ctypes.CDLL("./jass_engine.so")
        self.states = (GameState * n_envs)()
        # ...
    
    def reset(self) -> observations:
        # Reset all games, return initial observations
    
    def step(self, actions) -> (observations, rewards, dones, infos):
        # Execute actions across all parallel games
        # During declaration phase: actions are 0-157
        # During play phase: actions are card indices 0-35 (masked to legal)
```

### Component 2: Neural Network (PyTorch)

**Input features (~300 floats):**

Card features (36 × 6 = 216):
- `my_hand[36]`: binary — cards I hold
- `played_all[36]`: binary — all cards played in previous tricks
- `current_trick[36×3]`: binary — cards on table by position (up to 3 visible)
- `legal_mask[36]`: binary — legal moves

Context (36 + ~50):
- `is_trump[36]`: binary — which cards are trump suit
- `void_knowledge[3×4]`: binary — known voids (opponent × suit)
- Scalars: trick_number/8, my_points/157, gap/157, position_in_trick/3, am_leader, etc.

**For declaration phase:**
- `my_hand[36]` + `is_trump[36]` + hand statistics (trump count, suit lengths, high cards)
- ~100 features total

**Architecture:**
```python
class JassNet(nn.Module):
    def __init__(self, play_input_size=300, decl_input_size=100):
        # Play network
        self.play_trunk = nn.Sequential(
            nn.Linear(play_input_size, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.play_policy = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 36))
        self.play_value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        
        # Declaration network (separate — different input)
        self.decl_trunk = nn.Sequential(
            nn.Linear(decl_input_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.decl_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # Output: predicted score (regression), used as declaration
    
    def forward_play(self, obs, legal_mask):
        x = self.play_trunk(obs)
        logits = self.play_policy(x)
        logits[~legal_mask] = -1e9  # mask illegal
        value = self.play_value(x)
        return logits, value
    
    def forward_declare(self, hand_features):
        x = self.decl_trunk(hand_features)
        return self.decl_head(x)  # predicted score
```

### Component 3: PPO Training Loop

```python
# Pseudocode for the training loop
env = VecJassEnv(n_envs=256)
agent = JassNet()
optimizer = Adam(agent.parameters(), lr=3e-4)

for epoch in range(10000):
    # Collect rollouts
    obs = env.reset()
    
    # Declaration phase
    for player in range(4):
        decl_features = extract_decl_features(obs, player)
        predicted_score = agent.forward_declare(decl_features)
        declarations[player] = round(predicted_score.item())
        env.set_declarations(declarations)
    
    # Play phase — 9 tricks × 4 plays = 36 steps per game
    trajectory = []
    while not done:
        logits, value = agent.forward_play(obs, legal_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, done, info = env.step(action)
        trajectory.append((obs, action, log_prob, value, reward, done))
        obs = next_obs
    
    # PPO update
    # Compute advantages using GAE
    # Update policy and value heads
    # Update declaration head with MSE loss on (predicted_score - actual_score)
```

### Component 4: Deployment

After training, export to ONNX for fast inference:
```python
# Export
torch.onnx.export(agent, dummy_input, "jass_agent.onnx")

# Or just use the PyTorch model directly — inference is ~0.1ms per decision
# which is fast enough for real-time play
```

For playing against humans on Swisslos:
- Tampermonkey userscript reads game state from DOM
- Sends state to local Python server
- Server runs neural net inference
- Returns card to play / declaration value
- Userscript clicks the card

## Key Design Decisions

### Why PPO over DQN?
- DQN struggles with the sparse reward (only at round end, 36 actions away)
- PPO handles sparse rewards better with GAE advantage estimation  
- PPO naturally handles the continuous declaration action (regression head)
- PPO is more stable for self-play (important: all 4 players share weights)

### Why self-play?
- No need for a separate opponent model (the whole problem with PIMC)
- Naturally produces strong, realistic opponents
- Can add population-based training later (league of past checkpoints)

### Why C engine?
- RL needs millions of games. Python game logic would be 100-1000x slower.
- Target: 50,000+ games/second with vectorized C engine
- With 256 parallel envs: ~200 games/second per env × 256 = ~50K games/sec
- 10 million training games in ~3 minutes

### Declaration training approach
- **Option A (simpler)**: Separate supervised head. During self-play, record (hand, trump) → actual_score pairs. Train declaration head to predict actual_score. Use prediction as declaration.
- **Option B (joint)**: Include declaration in the RL reward. Agent's declaration feeds into its own reward. This lets it learn conservative declarations it can reliably hit.
- **Start with A, add B later.**

## Performance Targets

| Metric | Current (PIMC) | Target (RL) | Human |
|--------|---------------|-------------|-------|
| Avg deviation (self-play) | 5.7 | < 4.0 | — |
| Avg deviation (vs humans) | 14.9 | < 5.0 | 7-8 |
| Perfect rounds (0 dev) | 17.6% | > 25% | ~15% |
| Catastrophic (>20 dev) | 28% live | < 5% | ~5% |
| Decision time | 36ms | < 1ms | seconds |

## File Structure

```
jass-rl/
├── engine_c/
│   ├── jass_engine.c          # Game logic, vectorized sim
│   ├── jass_engine.h          # Header with GameState struct
│   ├── Makefile               # Build .so/.dll
│   └── test_engine.c          # C unit tests
├── env/
│   ├── jass_env.py            # Single game environment
│   ├── vec_env.py             # Vectorized wrapper (256+ parallel)
│   └── features.py            # Observation encoding
├── agent/
│   ├── network.py             # JassNet (policy + value + declaration)
│   ├── ppo.py                 # PPO algorithm
│   └── utils.py               # GAE, reward normalization
├── training/
│   ├── train.py               # Main training loop
│   ├── self_play.py           # Self-play game generation
│   ├── league.py              # Population-based training (later)
│   └── config.py              # Hyperparameters
├── evaluation/
│   ├── arena.py               # Benchmark against baselines
│   ├── analyze.py             # Training curve analysis
│   └── export.py              # ONNX export
├── deploy/
│   ├── server.py              # Local inference server
│   └── swisslos_bridge.js     # Tampermonkey userscript
├── tests/
│   ├── test_rules.py          # Verify C engine matches known game rules
│   ├── test_env.py            # Env step/reset correctness
│   ├── test_features.py       # Observation encoding
│   └── test_training.py       # Smoke test: loss decreases
├── CLAUDE.md                  # Project documentation for Claude Code
└── requirements.txt           # torch, numpy, etc.
```

## Build & Run

### Prerequisites
- GCC (for C engine)
- Python 3.10+
- PyTorch (with CUDA if GPU available)
- numpy

### Steps
```bash
# 1. Build C engine
cd engine_c && make && cd ..

# 2. Install Python deps
pip install torch numpy matplotlib

# 3. Run tests (ALWAYS do this first)
python -m pytest tests/ -v

# 4. Train
python training/train.py --n-envs 256 --n-epochs 10000 --lr 3e-4

# 5. Evaluate
python evaluation/arena.py --model checkpoints/latest.pt --n-games 1000

# 6. Export for deployment
python evaluation/export.py --model checkpoints/best.pt --output jass_agent.onnx
```

## Critical Implementation Notes

### Legal move computation MUST be correct
The undertrump restriction and Puur privilege are tricky. Get this wrong and the bot learns illegal strategies. Write exhaustive tests:
- Hand has trump but below table's trump → can play non-trump
- Hand has only Puur → can play anything when trump led
- Must overtrump if possible when trumping into a non-trump trick (NO — this is only when trump is LED)
- Actually: undertrump restriction only applies when you CAN'T follow the led suit and there's already a trump on the table

### Reward shaping considerations
- Base reward: `-|declared - actual|` at round end
- Consider intermediate shaping: small reward for being on-track mid-round (gap decreasing)
- Or: reward = `-|gap|` given at end, with value function estimating expected final gap
- **Start with pure sparse reward. Add shaping only if convergence is too slow.**

### Declaration as part of the episode
- The declaration is the FIRST action of each episode
- After all 4 players declare, play begins
- The declaration head should be trained end-to-end: bad declarations → bad rewards
- The agent should learn that a good declaration is one it can reliably achieve, not just the expected value

### Self-play details
- All 4 players use the same weights (parameter sharing)
- Each player gets its own observation (only sees own hand + public info)
- This is important: player 0 can NOT see players 1-3's hands
- Observations are from the perspective of the current player

### Training stability
- Use gradient clipping (max_grad_norm=0.5)
- Learning rate warmup, then cosine decay
- Entropy bonus (0.01) to prevent premature convergence
- Evaluate every 100 epochs against a frozen checkpoint
- Save best model by evaluation performance, not training loss

## What Already Exists (from previous work)

The previous project (`jass-differenzler/`) has:
- A working Python game engine (fully tested, 31 rule tests passing)
- A C engine (196K worlds/sec) with ctypes wrapper
- Neural network architecture (trained supervised, 66% play accuracy)
- Arena system for benchmarking
- Feature encoding (`encode_hand_rich`, ~300 features)
- Swisslos data collection userscript
- ~120 rounds of real Swisslos game data

**Reuse what makes sense** (especially the tested game rules and C engine skeleton), but the RL training pipeline should be built fresh — the old code was designed around PIMC, not RL.

## Success Criteria

1. **Convergence**: Training loss/deviation decreases over epochs
2. **Self-play strength**: < 4.0 avg dev in 1000-game self-play evaluation
3. **Generalization**: When deployed against Swisslos bots, avg dev < 7.0 (competitive with humans)
4. **Speed**: Full training run completes in < 24 hours on a single machine with GPU
5. **Correctness**: All game rule tests pass, no illegal moves generated

## Getting Started

1. **Build and test the C engine first.** This is the foundation. If game logic is wrong, everything trained on it will be wrong.
2. **Build the vectorized env.** Get 256 games running in parallel with random play. Verify speed (target: >10K games/sec).
3. **Implement basic PPO with random initialization.** Verify the agent learns SOMETHING (deviation should decrease from ~40 to ~20 within first 1000 epochs).
4. **Scale up training.** Run overnight. Monitor for convergence.
5. **Evaluate and iterate.**
