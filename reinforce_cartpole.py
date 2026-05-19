"""
REINFORCE (Monte-Carlo Policy Gradient) on CartPole-v1
Reference: Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed., Ch. 13.3

Algorithm (episodic, no baseline):
  For each episode:
    1. Generate trajectory {S_0,A_0,R_1, ..., S_{T-1},A_{T-1},R_T} following pi_theta
    2. For each t in the episode:
         G_t = sum_{k=t+1}^{T} gamma^{k-t-1} * R_k   (return from time t)
         theta <- theta + alpha * gamma^t * G_t * grad(ln pi(A_t|S_t, theta))
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
HIDDEN_SIZE   = 128      # neurons in each hidden layer
LEARNING_RATE = 1e-3
GAMMA         = 0.99     # discount factor
MAX_EPISODES  = 2_000
SOLVE_SCORE   = 475.0    # mean over 50 consecutive episodes considered "solved"
PRINT_EVERY   = 50


# ── 2-Layer MLP Policy ────────────────────────────────────────────────────────
class PolicyMLP(nn.Module):
    """
    Parameterises pi(a|s; theta).

    Architecture:
        obs (4,) -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax -> action probs
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """
    Compute discounted returns G_t for every timestep t in the episode.
    G_t = R_{t+1} + gamma*R_{t+2} + ... + gamma^{T-t-1}*R_T
    """
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def select_action(policy: PolicyMLP, obs: np.ndarray) -> tuple[int, torch.Tensor]:
    """Sample action from pi(·|obs) and return (action, log_prob)."""
    state = torch.from_numpy(obs).float().unsqueeze(0)  # (1, obs_dim)
    dist  = policy(state)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


# ── REINFORCE update (S&B eq. 13.8) ──────────────────────────────────────────
def reinforce_update(
    optimizer:  optim.Optimizer,
    log_probs:  list[torch.Tensor],
    returns:    torch.Tensor,
    gamma:      float,
) -> float:
    """
    theta <- theta + alpha * sum_t [ gamma^t * G_t * grad ln pi(A_t|S_t) ]

    Gradient ascent is implemented as gradient *descent* on the negative objective:
        loss = -sum_t [ gamma^t * G_t * log pi(A_t|S_t) ]
    """
    T = len(log_probs)
    gamma_t = torch.tensor([gamma ** t for t in range(T)], dtype=torch.float32)

    # Normalise returns for variance reduction (optional but widely used)
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

    policy_loss = -(gamma_t * returns_norm * torch.stack(log_probs)).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train() -> PolicyMLP:
    env = gym.make("CartPole-v1")
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]   # 4
    act_dim = env.action_space.n               # 2

    policy    = PolicyMLP(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    episode_rewards: list[float] = []

    for episode in range(1, MAX_EPISODES + 1):

        # ── Collect one full episode trajectory ──────────────────────────────
        obs, _    = env.reset(seed=SEED + episode)
        log_probs: list[torch.Tensor] = []
        rewards:   list[float]        = []

        done = False
        while not done:
            action, log_prob = select_action(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(float(reward))

        # ── Compute returns and update policy ─────────────────────────────────
        returns = compute_returns(rewards, GAMMA)
        reinforce_update(optimizer, log_probs, returns, GAMMA)

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        # ── Logging ───────────────────────────────────────────────────────────
        if episode % PRINT_EVERY == 0:
            mean_50 = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode:5d} | "
                f"Reward: {ep_reward:6.1f} | "
                f"Mean(last 50): {mean_50:6.1f}"
            )
            if mean_50 >= SOLVE_SCORE:
                print(f"\nSolved at episode {episode} "
                      f"(mean reward {mean_50:.1f} >= {SOLVE_SCORE}).")
                break

    env.close()
    return policy


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(policy: PolicyMLP, n_episodes: int = 10) -> None:
    env = gym.make("CartPole-v1", render_mode=None)
    policy.eval()
    total = 0.0
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=1000 + ep)
            done, ep_reward = False, 0.0
            while not done:
                action, _ = select_action(policy, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total += ep_reward
            print(f"  Eval episode {ep+1}: {ep_reward:.1f}")
    print(f"\nMean eval reward over {n_episodes} episodes: {total/n_episodes:.1f}")
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== REINFORCE on CartPole-v1 ===\n")
    trained_policy = train()
    print("\n--- Evaluation (greedy, no exploration) ---")
    evaluate(trained_policy)
