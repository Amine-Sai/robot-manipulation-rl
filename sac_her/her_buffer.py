import numpy as np
import torch

class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.clip = clip

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -self.clip, self.clip,
        ).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": self.count, "clip": self.clip}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]
        self.clip = d.get("clip", 5.0)

class HERReplayBuffer:
    def __init__(
        self,
        max_episodes,
        max_episode_steps,
        obs_dim,
        goal_dim,
        action_dim,
        n_sampled_goal=4,
    ):
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.n_sampled_goal = n_sampled_goal

        self.obs = np.zeros((max_episodes, max_episode_steps, obs_dim), dtype=np.float32)
        self.achieved_goal = np.zeros((max_episodes, max_episode_steps, goal_dim), dtype=np.float32)
        self.desired_goal = np.zeros((max_episodes, max_episode_steps, goal_dim), dtype=np.float32)
        self.actions = np.zeros((max_episodes, max_episode_steps, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_episodes, max_episode_steps, obs_dim), dtype=np.float32)
        self.next_achieved_goal = np.zeros((max_episodes, max_episode_steps, goal_dim), dtype=np.float32)

        self.episode_lengths = np.zeros(max_episodes, dtype=np.int32)

        self._episode_pos = 0   
        self._n_stored = 0      

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.goal_rms = RunningMeanStd(shape=(goal_dim,))

    @property
    def n_episodes(self):
        return min(self._n_stored, self.max_episodes)

    @property
    def n_transitions(self):
        if self.n_episodes == 0:
            return 0
        return int(self.episode_lengths[: self.n_episodes].sum())

    def store_episode(
        self,
        obs,
        achieved_goal,
        desired_goal,
        actions,
        next_obs,
        next_achieved_goal,
    ):
        T = len(obs)
        assert T <= self.max_episode_steps, (
            f"Episode length {T} exceeds max_episode_steps {self.max_episode_steps}"
        )

        idx = self._episode_pos
        self.obs[idx, :T] = obs
        self.achieved_goal[idx, :T] = achieved_goal
        self.desired_goal[idx, :T] = desired_goal
        self.actions[idx, :T] = actions
        self.next_obs[idx, :T] = next_obs
        self.next_achieved_goal[idx, :T] = next_achieved_goal
        self.episode_lengths[idx] = T

        self._episode_pos = (self._episode_pos + 1) % self.max_episodes
        self._n_stored += 1

    def update_normalizer(self, ep_obs, ep_ag):
        self.obs_rms.update(ep_obs)
        self.goal_rms.update(ep_ag)

    def normalize_obs_goal(
        self, obs, goal
    ):
        return np.concatenate([
            self.obs_rms.normalize(obs),
            self.goal_rms.normalize(goal),
        ], axis=-1)

    def get_normalizer_state(self):
        return {
            "obs_rms": self.obs_rms.state_dict(),
            "goal_rms": self.goal_rms.state_dict(),
        }

    def set_normalizer_state(self, state):
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.goal_rms.load_state_dict(state["goal_rms"])

    def sample(
        self,
        batch_size,
        compute_reward,
        device="cpu",
    ):
        n_ep = self.n_episodes
        assert n_ep > 0, "Buffer is empty — cannot sample."

        ep_indices = np.random.randint(0, n_ep, size=batch_size)
        ep_lens = self.episode_lengths[ep_indices]  
        t_indices = np.array([
            np.random.randint(0, l) for l in ep_lens
        ], dtype=np.int32)

        obs = self.obs[ep_indices, t_indices].copy()                     
        ag = self.achieved_goal[ep_indices, t_indices].copy()            
        g = self.desired_goal[ep_indices, t_indices].copy()              
        actions = self.actions[ep_indices, t_indices].copy()             
        next_obs = self.next_obs[ep_indices, t_indices].copy()           
        next_ag = self.next_achieved_goal[ep_indices, t_indices].copy()  

        # HER relabeling  —  "future" strategy
        her_prob = self.n_sampled_goal / (1.0 + self.n_sampled_goal)
        her_mask = np.random.uniform(size=batch_size) < her_prob
        can_relabel = t_indices < (ep_lens - 1)
        relabel = her_mask & can_relabel

        if relabel.any():
            future_t = np.array([
                np.random.randint(t + 1, l)
                for t, l, do in zip(t_indices, ep_lens, relabel)
                if do
            ], dtype=np.int32)

            relabel_ep = ep_indices[relabel]
            new_goals = self.achieved_goal[relabel_ep, future_t]  
            g[relabel] = new_goals

        rewards = compute_reward(next_ag, g, {})  

        full_obs = self.normalize_obs_goal(obs, g)             
        full_next_obs = self.normalize_obs_goal(next_obs, g)   

        dones = np.zeros(batch_size, dtype=np.float32)

        return {
            "observations": torch.as_tensor(full_obs, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device=device),
            "next_observations": torch.as_tensor(full_next_obs, dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
            "dones": torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1),
        }
