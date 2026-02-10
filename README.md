# ⚔️ Pokémon Showdown RL: The Autonomous Battle Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://github.com/DLR-RM/stable-baselines3)

An advanced Reinforcement Learning framework designed to master the complexities of competitive Pokémon battles. This project trains autonomous agents (PPO & DQN) to navigate the high-stakes, stochastic environment of Pokémon Showdown.

---

## 🚀 Project Overview

Pokémon battles are more than just "using the strongest move." They are games of incomplete information, prediction, and strategic positioning. This project leverages **Deep Reinforcement Learning** to bridge the gap between human intuition and machine precision.

Using a custom `Gymnasium` environment wrapping the `poke-env` library, our agents learn to:
- **Analyze** 48+ features of the battlefield in real-time.
- **Predict** opponent switches and type-effectiveness.
- **Optimize** long-term survival through smart health management and risk assessment.

---

## 🎮 Demo Features

The project includes a unified demo interface (`run_demo.bat`) with three core modes:

1.  **Human vs Agent**: Test your skills! Challenge a versioned PPO or DQN model via a local Showdown server.
2.  **Autonomous Evaluation**: Watch the giants clash! A control experiment matching PPO against DQN over 100 battles to determine the superior architecture.
3.  **Iterative Training**: A one-click pipeline to retrain agents. Models are automatically versioned (e.g., `v1` ➔ `v2`) to preserve progress and prevent data loss.

---

## ✨ Key Features

- 🧠 **Dual Architectures**: Full support for PPO (high stability) and DQN (aggressive exploration).
- 🔄 **Safe Synchronization**: Custom rendezvous pattern using `asyncio` queues for deterministic turn progression between Python and Showdown.
- 🛡️ **Action Masking**: Intelligent layers prevent illegal moves during forced switches or move-lock scenarios.
- 📦 **Model Versioning**: Automatic detection and loading of the highest versioned model for training and battles.
- 🧹 **Clean Logic**: Consolidated 48-feature observation building for training/inference parity.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Node.js** (for the Pokémon Showdown server)
- **Windows** (tested on Windows; batch files provided)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/your-username/pokemon-showdown-rl.git
cd pokemon-showdown-rl

# Create and activate virtual environment
python -m venv python/venv
python/venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the Showdown Server
Navigate to the `showdown/` folder and start the local server:
```powershell
cd showdown/pokemon-showdown
node pokemon-showdown
```

### 4. Launch the Demo
Run the startup script in the root directory:
```powershell
./run_demo.bat
```

---

## 🏋️ Training Agents

Training is standardized and iterative. To start a training run (20,000 steps per algorithm):

1.  Open the `run_demo.bat` menu.
2.  Select **Option 3: Run Iterative Training**.
3.  The agent will load the latest `vX.zip`, learn for 20k steps, and save a new `vX+1.zip`.

**Manual Training:**
- `python python/train_ppo.py`
- `python python/train_dqn.py`

---

## 📁 Project Structure

```text
├── python/
│   ├── venv/             # Python Virtual Environment
│   ├── pokemon_env.py    # Custom Gymnasium Environment
│   ├── model_utils.py    # Auto-versioning & Obs logic
│   ├── battle_vs_human.py# Human battle listener
│   ├── battle_control.py # PPO vs DQN evaluation
│   ├── train_ppo.py      # PPO training pipeline
│   ├── train_dqn.py      # DQN training pipeline
│   └── *_team.txt        # Competitive team files
├── models/               # v1, v2, v3 model storage
├── showdown/             # Pokemon Showdown local server
└── run_demo.bat          # Unified entry point
```

---

## 📈 Results & Observations

- **PPO** typically achieves higher stability and prefers conservative switching to preserve health.
- **DQN** focuses heavily on high-damage output and aggressive move selection.
- Both agents consistently reach **85%+ win rates** against random opponents within 20,000 steps of training.

---

## 🔮 Future Improvements

- [ ] Support for Dynamax/Terastallization mechanics.
- [ ] Integration of Transformer-based observation encoders.
- [ ] Multi-agent "Self-Play" league for exponential strategy growth.

---

## 🤝 Credits & Acknowledgements

- **Pokémon Showdown**: The incredible battle simulator powering this project.
- **Stable-Baselines3**: Reliable RL algorithms for research and production.
- **poke-env**: The bridge that made Python-Showdown communication a breeze.

---

## 📜 License
Available under the **MIT License**. See `LICENSE` for details.

---
*Created by k33r47 - 2026*
