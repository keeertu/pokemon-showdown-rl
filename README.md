# Pokémon Battle Simulator RL

This project implements a reinforcement learning environment for competitive Pokémon battles, specifically targeting Gen 9 Random Battles. It leverages the Pokémon Showdown simulator and the `poke-env` library to provide a standardized interface for RL agents. Battles are executed in a real Pokémon Showdown simulator instance, not a mock or simplified environment, enabling interaction with the full game logic and stochasticity of competitive play.

**Intended Audience**: This project is intended for developers or researchers familiar with reinforcement learning and asynchronous systems.

## Gen 9 Random Battles Constraints
- **No team building**: The agent must adapt to randomly generated teams (Random Battles format).
- **Partial observability**: The agent only sees the information available to a player in a real match (e.g., opponent's held items and full team are hidden until revealed).
- **Long-horizon credit assignment**: Decisions made in early turns can have critical impacts many turns later.

## Current State

- **Environment wrapper**: ✅ Functional
- **Async RL bridge (RLPlayer)**: ✅ Functional
- **Training loop**: 🚧 Work in progress
- **Learning agent (DQN/PPO/etc.)**: ❌ Not implemented yet

## Tech Stack

- **Python**: Core logic and RL environment.
- **poke-env**: Python interface for Pokémon Showdown.
- **Gymnasium**: Standard API for reinforcement learning environments.
- **PyTorch**: Deep learning framework for agent implementation.
- **Pokémon Showdown (Node.js)**: The underlying battle simulator.
- **asyncio**: Asynchronous I/O for handling battle events.

## Project Structure

- `python/`: Contains the RL logic, environment wrappers, and agent definitions.
    - `pokemon_env.py`: Defines the `PokemonShowdownEnv` (Gymnasium wrapper) and `RLPlayer` (async bridge).
    - `test_showdown.py`: Integration test for bot communication.
- `showdown/`: Contains a local instance of the Pokémon Showdown server.
    - `pokemon-showdown/`: The Node.js server source code.
- `report/`: Documentation and project write-ups.

## How to Run (Current Capabilities)

> Note: These steps verify environment setup and simulator communication only. End-to-end RL training is not yet implemented.

### 1. Dependency Setup
Ensure you have Python 3.10+ and Node.js 16+ installed.

**Python Setup:**
```bash
pip install gymnasium poke-env numpy torch
```

**Node.js Setup:**
```bash
cd showdown/pokemon-showdown
npm install
```

### 2. Start the Showdown Server
The simulator must be running locally for the Python environment to connect.
```bash
cd showdown/pokemon-showdown
node pokemon-showdown start
```

### 3. Run Environment Test
Test the Gymnasium wrapper with a random agent:
```bash
python python/pokemon_env.py
```

### 4. Run Client Test
Verify the communication bridge:
```bash
python python/test_showdown.py
```

## Known Limitations
- **Sparse or noisy reward signals**: Win/loss signals only arrive at the end of long episodes; intermediate HP rewards can be misleading.
- **High variance between battles**: Random team generation leads to significant variance in difficulty and matchups.
- **Training not yet reproducible or stabilized**: As the training loop is WIP, hyperparameter sensitivity and convergence are not yet established.

## What Is NOT Implemented Yet

- **No trained RL agent**: Currently relies on random or manual action selection.
- **No stable reward shaping**: Reward signals are currently basic (HP difference).
- **No long-horizon training pipeline**: Infrastructure for multi-epoch training and evaluation is pending.

## Future Roadmap

1. **Heuristic Agent**: Implement a rule-based baseline using type-advantage logic.
2. **Reward Shaping**: Refine reward functions to include status effects and positioning.
3. **RL Agent**: Implement DQN or PPO using PyTorch.
4. **Self-play**: Establish a pipeline for agents to train against themselves.

## Project Philosophy

- **Research-oriented**: Focused on exploring RL application in complex state spaces.
- **Iterative development**: Prioritizing functional building blocks over a finished product.
- **Transparency over polish**: Openly documenting limitations and work-in-progress state.
