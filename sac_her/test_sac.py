import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from her_buffer import RunningMeanStd

CHECKPOINT = "checkpoints/YOUR_CHECKPOINT.pt"
ENV_ID = "FetchPickAndPlace-v4"
NUM_EPISODES = 5
SEED = 42

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

def load_actor(ckpt_path, device, env):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    action_low = env.action_space.low
    action_high = env.action_space.high
    action_dim = env.action_space.shape[0]

    obs_dim = ckpt.get("obs_dim", 25)
    goal_dim = ckpt.get("goal_dim", 3)
    input_dim = obs_dim + goal_dim
    hidden = ckpt.get("config", {}).get("hidden_dim", 256)
    
    actor = Actor(input_dim, action_dim, action_low, action_high, hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.to(device)
    actor.eval()

    step = ckpt.get("global_step", "?")
    ep = ckpt.get("episode_count", "?")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Training step: {step} | Episodes: {ep}")

    normalizer_state = ckpt.get("normalizer", None)
    return actor, normalizer_state

def build_obs_tensor(obs_dict, device, obs_rms=None, goal_rms=None):
    obs = obs_dict["observation"].astype(np.float32)
    goal = obs_dict["desired_goal"].astype(np.float32)
    
    if obs_rms is not None and goal_rms is not None:
        flat = np.concatenate([
            obs_rms.normalize(obs),
            goal_rms.normalize(goal),
        ], axis=-1)
    else:
        flat = np.concatenate([obs, goal]).astype(np.float32)
        
    return torch.as_tensor(flat, device=device).unsqueeze(0)

def run_episode(env, actor, device, obs_rms=None, goal_rms=None):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        obs_t = build_obs_tensor(obs, device, obs_rms, goal_rms)
        with torch.no_grad():
            _, _, action_mean = actor.get_action(obs_t)

        action = action_mean[0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    success = info.get("is_success", None)
    return total_reward, success, steps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.isfile(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    env = gym.make(ENV_ID, render_mode="human")
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    actor, normalizer_state = load_actor(CHECKPOINT, device, env)

    obs_rms, goal_rms = None, None
    if normalizer_state is not None:
        obs_rms = RunningMeanStd(shape=(normalizer_state["obs_rms"]["mean"].shape[0],))
        obs_rms.load_state_dict(normalizer_state["obs_rms"])
        goal_rms = RunningMeanStd(shape=(normalizer_state["goal_rms"]["mean"].shape[0],))
        goal_rms.load_state_dict(normalizer_state["goal_rms"])
        print("Normalizer loaded")
    else:
        print("No normalizer found in checkpoint")

    print(f"\nRunning {NUM_EPISODES} episodes on {ENV_ID}")
    
    returns = []
    successes = []

    for ep in range(NUM_EPISODES):
        ret, success, steps = run_episode(
            env, actor, device, obs_rms, goal_rms
        )
        returns.append(ret)
        if success is not None:
            successes.append(float(success))

        print(f"Episode {ep+1} | Return: {ret:.2f} | Steps: {steps} | Success: {success}")

    print(f"Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    if successes:
        print(f"Success rate: {np.mean(successes)*100:.1f}%")
        
    env.close()

if __name__ == "__main__":
    main()
