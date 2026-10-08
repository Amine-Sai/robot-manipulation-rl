import os
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from her_buffer import HERReplayBuffer

config = {
    "env_id": "FetchPickAndPlace-v4",
    "seed": 1,
    "cuda": True,
    "total_timesteps": 1_000_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    "gamma": 0.98,
    "tau": 0.005,
    "policy_lr": 3e-4,
    "q_lr": 1e-3,
    "policy_frequency": 2,
    "target_network_frequency": 1,
    "hidden_dim": 256,
    "grad_clip": 5.0,
    "autotune": True,
    "alpha": 0.2,
    "n_sampled_goal": 4,
    "max_episodes": 40_000,
    "n_batches": 50,
    "checkpoint_freq": 500,
    "checkpoint_dir": "checkpoints",
}

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_shape = env.observation_space.spaces["observation"].shape[0]
        goal_shape = env.observation_space.spaces["desired_goal"].shape[0]
        input_dim = obs_shape + goal_shape + env.action_space.shape[0]
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class Actor(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_shape = env.observation_space.spaces["observation"].shape[0]
        goal_shape = env.observation_space.spaces["desired_goal"].shape[0]
        input_dim = obs_shape + goal_shape
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_mean = nn.Linear(hidden_dim, env.action_space.shape[0])
        self.fc_logstd = nn.Linear(hidden_dim, env.action_space.shape[0])

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
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

def main():
    run_name = f"{config['env_id']}__{config['seed']}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() and config["cuda"] else "cpu")
    print(f"Device: {device}")

    env = make_env(config["env_id"], config["seed"])()
    
    obs_dim = env.observation_space.spaces["observation"].shape[0]
    goal_dim = env.observation_space.spaces["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    
    max_ep_steps = env.spec.max_episode_steps if env.spec.max_episode_steps else 50
    
    buffer = HERReplayBuffer(
        max_episodes=config["max_episodes"],
        max_episode_steps=max_ep_steps,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        n_sampled_goal=config["n_sampled_goal"],
    )
    
    compute_reward = env.unwrapped.compute_reward

    actor = Actor(env, hidden_dim=config["hidden_dim"]).to(device)
    qf1 = SoftQNetwork(env, hidden_dim=config["hidden_dim"]).to(device)
    qf2 = SoftQNetwork(env, hidden_dim=config["hidden_dim"]).to(device)
    qf1_target = SoftQNetwork(env, hidden_dim=config["hidden_dim"]).to(device)
    qf2_target = SoftQNetwork(env, hidden_dim=config["hidden_dim"]).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=config["q_lr"]
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config["policy_lr"])
    
    if config["autotune"]:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=config["q_lr"])
    else:
        alpha = config["alpha"]

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    global_step = 0
    episode_count = 0
    start_time = time.time()
    
    success_history = deque(maxlen=100)
    return_history = deque(maxlen=100)

    while global_step < config["total_timesteps"]:
        obs_dict, info = env.reset(seed=config["seed"] + episode_count)
        
        ep_obs = []
        ep_ag = []
        ep_g = []
        ep_actions = []
        ep_next_obs = []
        ep_next_ag = []
        
        ep_reward = 0.0
        
        for step_idx in range(max_ep_steps):
            global_step += 1
            
            obs_arr = obs_dict["observation"]
            ag_arr = obs_dict["achieved_goal"]
            g_arr = obs_dict["desired_goal"]
            
            if global_step < config["learning_starts"]:
                action = env.action_space.sample()
            else:
                norm_obs_goal = buffer.normalize_obs_goal(obs_arr, g_arr)
                t_obs = torch.as_tensor(norm_obs_goal, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _ = actor.get_action(t_obs)
                action = action_t[0].cpu().numpy()
            
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            ep_obs.append(obs_arr)
            ep_ag.append(ag_arr)
            ep_g.append(g_arr)
            ep_actions.append(action)
            ep_next_obs.append(next_obs_dict["observation"])
            ep_next_ag.append(next_obs_dict["achieved_goal"])
            
            obs_dict = next_obs_dict
            
            if terminated or truncated:
                break
                
        ep_obs_arr = np.array(ep_obs)
        ep_ag_arr = np.array(ep_ag)
        
        buffer.store_episode(
            ep_obs_arr, ep_ag_arr, np.array(ep_g),
            np.array(ep_actions), np.array(ep_next_obs), np.array(ep_next_ag),
        )
        buffer.update_normalizer(ep_obs_arr, ep_ag_arr)
        
        success = float(info.get("is_success", 0.0))
        success_history.append(success)
        return_history.append(ep_reward)
        episode_count += 1
        
        if global_step >= config["learning_starts"]:
            for _ in range(config["n_batches"]):
                data = buffer.sample(config["batch_size"], compute_reward, device)
                
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(data["next_observations"])
                    q1_next = qf1_target(data["next_observations"], next_actions)
                    q2_next = qf2_target(data["next_observations"], next_actions)
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                    target_q = data["rewards"].flatten() + (1 - data["dones"].flatten()) * config["gamma"] * min_q_next.view(-1)
                    
                    clip_range = 1.0 / (1.0 - config["gamma"])
                    target_q = torch.clamp(target_q, -clip_range, 0.0)
                    
                q1_pred = qf1(data["observations"], data["actions"]).view(-1)
                q2_pred = qf2(data["observations"], data["actions"]).view(-1)
                qf1_loss = F.mse_loss(q1_pred, target_q)
                qf2_loss = F.mse_loss(q2_pred, target_q)
                qf_loss = qf1_loss + qf2_loss
                
                q_optimizer.zero_grad()
                qf_loss.backward()
                if config["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(
                        list(qf1.parameters()) + list(qf2.parameters()),
                        config["grad_clip"],
                    )
                q_optimizer.step()
                
                if global_step % config["policy_frequency"] == 0:
                    pi, log_pi, _ = actor.get_action(data["observations"])
                    q1_pi = qf1(data["observations"], pi)
                    q2_pi = qf2(data["observations"], pi)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * log_pi - min_q_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if config["grad_clip"] > 0:
                        nn.utils.clip_grad_norm_(actor.parameters(), config["grad_clip"])
                    actor_optimizer.step()
                    
                    if config["autotune"]:
                        with torch.no_grad():
                            _, log_pi_a, _ = actor.get_action(data["observations"])
                        alpha_loss = (-log_alpha.exp() * (log_pi_a + target_entropy)).mean()
                        
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        
                if global_step % config["target_network_frequency"] == 0:
                    for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                        tp.data.copy_(config["tau"] * p.data + (1 - config["tau"]) * tp.data)
                    for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                        tp.data.copy_(config["tau"] * p.data + (1 - config["tau"]) * tp.data)
                        
        if episode_count % 10 == 0:
            mean_success = np.mean(success_history) if success_history else 0.0
            mean_return = np.mean(return_history) if return_history else 0.0
            sps = int(global_step / (time.time() - start_time))
            
            print(f"ep {episode_count} | step {global_step} | ret {mean_return:.1f} | success {mean_success:.1%} | alpha {alpha:.3f} | SPS {sps}")
            
            writer.add_scalar("charts/episodic_return", ep_reward, global_step)
            writer.add_scalar("charts/success_rate", mean_success, global_step)
            writer.add_scalar("charts/alpha", alpha, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/episodes", episode_count, global_step)
            writer.add_scalar("charts/buffer_episodes", buffer.n_episodes, global_step)
            
        if episode_count % config["checkpoint_freq"] == 0 and episode_count > 0:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"{run_name}_ep{episode_count}.pt")
            torch.save({
                "actor": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "qf1_target": qf1_target.state_dict(),
                "qf2_target": qf2_target.state_dict(),
                "q_opt": q_optimizer.state_dict(),
                "actor_opt": actor_optimizer.state_dict(),
                "log_alpha": log_alpha.detach().cpu() if config["autotune"] else torch.tensor(alpha),
                "a_opt": a_optimizer.state_dict() if config["autotune"] else None,
                "global_step": global_step,
                "episode_count": episode_count,
                "obs_dim": obs_dim,
                "goal_dim": goal_dim,
                "action_dim": action_dim,
                "normalizer": buffer.get_normalizer_state(),
                "config": config,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = os.path.join(config["checkpoint_dir"], f"{run_name}_final.pt")
    torch.save({
        "actor": actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "q_opt": q_optimizer.state_dict(),
        "actor_opt": actor_optimizer.state_dict(),
        "log_alpha": log_alpha.detach().cpu() if config["autotune"] else torch.tensor(alpha),
        "a_opt": a_optimizer.state_dict() if config["autotune"] else None,
        "global_step": global_step,
        "episode_count": episode_count,
        "obs_dim": obs_dim,
        "goal_dim": goal_dim,
        "action_dim": action_dim,
        "normalizer": buffer.get_normalizer_state(),
        "config": config,
    }, final_path)
    print(f"Training complete. {episode_count} episodes, {global_step} steps")
    print(f"Final success rate: {np.mean(success_history):.1%}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
