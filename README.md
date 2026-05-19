# REINFORCE on CartPole-v1

A clean PyTorch implementation of the **REINFORCE** (Monte-Carlo Policy Gradient) algorithm applied to the classic **CartPole-v1** control task, following the treatment in:

> Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., **Chapter 13.3**

---

## Table of Contents

- [Background](#background)
- [Algorithm](#algorithm)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Known Limitations & Next Steps](#known-limitations--next-steps)
- [References](#references)

---

## Background

**CartPole-v1** is a classic benchmark from OpenAI Gymnasium. The agent controls a cart on a frictionless track and must keep a pole balanced upright by pushing left or right.

| Property | Value |
|---|---|
| Observation space | 4 continuous values (cart position, cart velocity, pole angle, pole angular velocity) |
| Action space | Discrete {0 = push left, 1 = push right} |
| Reward | +1 for every timestep the pole stays upright |
| Episode end | Pole angle > ±12°, cart position > ±2.4, or 500 steps reached |
| Solved threshold | Mean reward ≥ 475 over 50 consecutive episodes |

---

## Algorithm

REINFORCE is the simplest member of the **policy gradient** family. It directly optimises the policy parameters θ by ascending the gradient of the expected return.

### Objective

```
J(θ) = E_π [ Σ_t γ^t R_{t+1} ]
```

### Policy Gradient Theorem (S&B eq. 13.5)

```
∇J(θ) = E_π [ Σ_t γ^t G_t ∇ ln π(A_t | S_t, θ) ]
```

### Parameter Update (S&B eq. 13.8)

```
θ ← θ + α · Σ_t [ γ^t · G_t · ∇ ln π(A_t | S_t, θ) ]
```

where `G_t` is the **discounted return** from timestep `t`:

```
G_t = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ... + γ^{T-t-1}·R_T
```

computed efficiently via the backward recurrence:

```
G_T = 0
G_t = R_{t+1} + γ · G_{t+1}
```

### Implementation as Gradient Descent

PyTorch minimises loss, so the update is implemented as descent on the **negative** objective:

```
L(θ) = − Σ_t [ γ^t · Ĝ_t · log π(A_t | S_t, θ) ]
```

where `Ĝ_t` are the **normalised** returns (zero mean, unit variance per episode) for variance reduction.

### Episode Flow

```
┌─────────────────────────────────────────────────────┐
│  For each episode:                                   │
│                                                      │
│  1. Reset environment  →  S_0                        │
│                                                      │
│  2. Roll out trajectory:                             │
│       A_t ~ π(·|S_t; θ)                             │
│       collect (S_t, A_t, R_{t+1}, log π(A_t|S_t))  │
│       until terminal                                 │
│                                                      │
│  3. Compute G_t for all t  (backward recurrence)    │
│                                                      │
│  4. Normalise returns  →  Ĝ_t                       │
│                                                      │
│  5. Compute loss  L = −Σ_t γ^t · Ĝ_t · log π_t    │
│                                                      │
│  6. Backprop  +  Adam step                          │
└─────────────────────────────────────────────────────┘
```

---

## Architecture

The policy is a **2-layer MLP** that maps observations to a categorical action distribution:

```
Input (4,)
    │
    ▼
Linear(4 → 128) + ReLU
    │
    ▼
Linear(128 → 128) + ReLU
    │
    ▼
Linear(128 → 2)         ← logits
    │
    ▼
Categorical(logits)     ← π(a | s; θ)
```

Using `logits` (unnormalised log-probabilities) instead of explicit `softmax` is numerically more stable and is the standard PyTorch practice.

---

## Project Structure

```
CartPole_MLP/
├── reinforce_cartpole.py   # Full implementation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Module breakdown (`reinforce_cartpole.py`)

| Symbol | Role |
|---|---|
| `PolicyMLP` | 2-layer MLP; `forward` returns a `Categorical` distribution |
| `compute_returns` | Backward recurrence to compute `G_t` for each timestep |
| `select_action` | Samples from `π(·\|s)`, returns `(action, log_prob)` |
| `reinforce_update` | Computes and backpropagates the policy-gradient loss |
| `train` | Main training loop (up to 2 000 episodes) |
| `evaluate` | Runs the trained policy deterministically on 10 test episodes |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/FawzyKhalil/CartPole_MLP.git
cd CartPole_MLP

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, Gymnasium ≥ 1.0, NumPy ≥ 1.24

---

## Usage

```bash
python reinforce_cartpole.py
```

Sample output:

```
=== REINFORCE on CartPole-v1 ===

Episode    50 | Reward:   14.0 | Mean(last 50):   33.9
Episode   100 | Reward:   39.0 | Mean(last 50):   42.9
Episode   150 | Reward:  131.0 | Mean(last 50):   81.3
...
Episode  1100 | Reward:  248.0 | Mean(last 50):  216.8
...

--- Evaluation (greedy, no exploration) ---
  Eval episode 1: 73.0
  ...
Mean eval reward over 10 episodes: 66.6
```

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `HIDDEN_SIZE` | 128 | Neurons per hidden layer |
| `LEARNING_RATE` | 1e-3 | Adam optimizer step size |
| `GAMMA` | 0.99 | Discount factor γ |
| `MAX_EPISODES` | 2 000 | Training budget |
| `SOLVE_SCORE` | 475.0 | Mean reward over 50 episodes to declare solved |
| `SEED` | 42 | Global random seed |

---

## Results

Vanilla REINFORCE exhibits **high variance** — a well-known property of the algorithm (S&B §13.1). Training curves typically show:

- Rapid early improvement followed by sharp drops (catastrophic forgetting)
- Multiple "restarts" before sustained high performance
- High sensitivity to random seed and learning rate

This is **expected and correct behaviour** for the baseline algorithm without a value-function baseline.

---

## Known Limitations & Next Steps

Vanilla REINFORCE is intentionally minimal. The following extensions from S&B are natural next steps:

| Extension | S&B Section | Benefit |
|---|---|---|
| **REINFORCE with baseline** (subtract V(s)) | §13.4 | Reduces variance without introducing bias |
| **Actor-Critic** (bootstrap with 1-step TD) | §13.5 | Lower variance, online updates |
| **Entropy regularisation** | §13.7 | Prevents premature convergence |
| **Gradient clipping** | — | Stabilises training |
| **Multiple seeds + plotting** | — | Reliable performance measurement |

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Online](http://incompleteideas.net/book/the-book-2nd.html)
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229–256.
- Gymnasium documentation: [gymnasium.farama.org](https://gymnasium.farama.org)
- PyTorch documentation: [pytorch.org/docs](https://pytorch.org/docs)
