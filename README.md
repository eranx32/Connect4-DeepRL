# 🔴 Connect 4 Pro - Dueling DQN AI 🟡

> A sophisticated Reinforcement Learning agent that mastered the game of Connect 4 using **Dueling Deep Q-Networks (Dueling DQN)** and **Curriculum Learning**.

<div align="center">
  <img src="assets/preview.gif" alt="Connect 4 Gameplay" width="300" />
  <br>
  <em>(Add a GIF here of your GUI in action!)</em>
</div>

## 🚀 Overview

This project implements an advanced AI agent capable of playing Connect 4 at a high level. Unlike simple tabular Q-Learning, this agent uses a **Deep Convolutional Neural Network (CNN)** to "see" the board and understand spatial patterns.

The training process involved hours of simulation where the agent learned from scratch, evolving from random moves to strategic blocking and trap-setting.

## ✨ Key Features

* 🧠 **Dueling DQN Architecture:** The network splits into two streams — one estimating the **Value** of the state and another estimating the **Advantage** of each action. This stabilizes learning significantly compared to standard DQN.
* 🎓 **Curriculum Learning:** The agent didn't just play against itself. It trained against a **Dynamic Rival** that started as a random player and slowly evolved into a smart heuristic master, forcing the agent to adapt to increasingly difficult strategies.
* 🛡️ **Hybrid Decision Making:** The agent combines the Neural Network's long-term strategy with a hard-coded "Safety Net" to detect immediate checkmates (winning moves or forced blocks).
* 🎨 **Polished GUI:** A fully interactive game interface built with `Pygame`, featuring chip gravity, bounce animations, and mouse-hover effects.
* ⚡ **Optimized Math:** Uses `scipy.signal.convolve2d` for extremely fast win-checking using 2D kernels instead of slow loops.

## 🛠️ Neural Network Architecture

The brain of the agent (`Connect4Model.py`) is designed to capture the spatial nature of the game:

1.  **Input:** A `(2, 6, 7)` Tensor representing the board (Channel 0: Agent pieces, Channel 1: Rival pieces).
2.  **Conv Layers:** 3 layers of Convolutional Filters (64 -> 128 -> 128) to detect lines and patterns.
3.  **Dueling Split:**
    * **Value Stream:** Outputs a single scalar (How good is this board state?).
    * **Advantage Stream:** Outputs 7 values (How good is each column relative to others?).
4.  **Aggregation:** Combines streams to produce final Q-Values.

## 📂 Project Structure

```bash
├── agent.py            # The RL Agent (DQN logic, Experience Replay, Epsilon Greedy)
├── connect4Model.py    # PyTorch Neural Network (Dueling DQN implementation)
├── connect4Env.py      # Game Environment & Logic (Fast convolution-based win check)
├── main.py             # Main training loop with Curriculum Learning
├── gui.py              # Interactive Game Interface (Pygame)
├── assets/             # Images for the GUI
└── models/             # Saved trained models (.pth)
```

## ⚙️ Installation & Usage

### Prerequisites
Make sure you have Python installed, then install the dependencies:

```bash
pip install torch numpy pygame scipy tqdm
```
### 1. Train the Agent
To start the training process (this may take hours for high proficiency):

```bash

python main.py
```
The script saves the best model to the models/ directory automatically.

### 2. Play Against the AI

Launch the GUI to challenge your trained model:

```Bash

python gui.py
```

## 📈 Training Strategy (Curriculum)
To prevent the agent from getting stuck in local optima, I implemented a dynamic training loop:

1. **Phase 1 (Exploration):** High Epsilon, Rival is 100% random. Agent learns basic rules.

2. **Phase 2 (Adaptation):** Epsilon decays, Rival becomes smarter (mixes random moves with heuristic blocks).

3. **Phase 3 (Mastery):** Low Epsilon, Rival plays near-perfect heuristic game. Agent must learn advanced tactics to win.


