# Pokémon Battle Simulator RL

A reinforcement learning framework for competitive Pokémon battles, using **Stable-Baselines3 (PPO)** and **Gymnasium** to train agents on a local **Pokémon Showdown** server.

---

## 1. Project Overview
The goal of this project is to build a high-performance, strategic agent for Pokémon battles. Unlike simplified simulations, this project interacts with a full Pokémon Showdown Node.js server, ensuring absolute fidelity to the game's mechanics, including complex status effects, weather, and RNG.

### Key Architecture
- **Simulator**: [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (Active Node.js instance)
- **Interface**: `poke-env` for protocol handling and asynchronous battle management.
- **Environment**: Custom Gymnasium wrapper (`PokemonShowdownEnv`) with robust **Queue-based synchronization**.
- **Model**: Stable-Baselines3 PPO (Proximal Policy Optimization) with an MLP policy.

---

## 2. Current Capabilities
- **Stable Gymnasium Wrapper**: Resolved legacy turn-synchronization issues. `step()` now correctly waits for server confirmation before proceeding.
- **Unique Session Management**: Uses UUID-based naming (`RLPlayer_xxxx`) for every run to prevent session collisions on the local server.
- **Stability verified**: Successfully runs for hundreds of episodes without freezing or desync.
- **Human vs Agent Mode**: A dedicated mode allows users to challenge the trained agent directly from a web browser.
- **Advanced Reward Shaping**: Rewards are calculated based on HP differentials and faint counts to provide dense signals for training.

---

## 3. Training Pipeline
The pipeline is managed via `train_ppo.py`.

- **Observation Space (23 features)**: Includes normalized turn counts, team HP fractions, type effectiveness multipliers, weather/terrain flags, and status conditions.
- **Reward System**:
  - `(Δ Damage Dealt - Δ Damage Taken) + 0.5 * (Δ Faints)`
  - `+/- 2.0` terminal reward for victory/defeat.
  - `-0.02` per-turn penalty to discourage passive play.
- **Execution**: The script initializes a local `RandomPlayer` opponent and launches asynchronous battle threads monitored by the SB3 training loop.

---

## 4. Human Battle Mode
Interact with your trained agent in a real battle.

1.  **Start Showdown**: Ensure your local server is running at `localhost:8000`.
2.  **Launch Agent**:
    ```bash
    python battle_vs_human.py
    ```
3.  **Challenge**: the script will generate a unique agent name. Challenge this name on your browser-based Showdown client (Format: **Gen 8 Anything Goes**).

---

## 5. Current Limitations (Research Focus)
- **Static Strategy**: The agent relies on current-state observation and lacks long-term planning (e.g., preserving a core sweeper for late-game).
- **Missing State Data**: Stat boost stages and entry hazards (Stealth Rock, Spikes) are not yet in the observation vector.
- **Simple Switching**: The agent often prefers staying in over strategic pivots unless forced to switch.

---

## 6. Future Roadmap
- [ ] **Observation Expansion**: Add stat stages, weather duration, and hazard tracking.
- [ ] **Complex Rewards**: Reward "positioning" (e.g., getting a boost) vs just raw damage.
- [ ] **Opponent Scaling**: Move from Random opponents to heuristic-based or self-play regimes.
- [ ] **Architecture Search**: Compare PPO against DQN and QR-DQN implementations.
- [ ] **Web Deployment**: Allow the agent to play on public Smogon ladders.

---

## 7. Project Structure
- `pokemon_env.py`: Core Gymnasium logic and Queue-based sync engine.
- `train_ppo.py`: PPO training and hyperparameter configuration.
- `battle_vs_human.py`: Inference mode for human challenges.
- `ash_team.txt` / `leon_team.txt`: Team configurations in Poke-Paste format.

---

## 8. Research Direction
This project explores the intersection of high-cardinality action spaces and multi-modal state representation in a stochastic, non-deterministic game environment. Contribution focus areas include reward shaping, state embedding optimization, and curriculum learning.
