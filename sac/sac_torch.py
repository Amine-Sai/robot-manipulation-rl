import torch as T
import numpy as np
import torch.nn.functional as F
from sac.buffer import ReplayBuffer
from sac.networks import ActorNetwork, CriticNetwork


class Agent():
    def __init__(self, input_dims, n_actions, alpha=3e-4, beta=1e-4, tau=5e-3,
                 gamma=0.99, env=None, max_size=100_000, batch_size=128,
                 reward_scale=1, auto_entropy=True, alpha_init=0.2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='target_critic_2')

        self.update_network_parameters(tau=1)

        self.auto_entropy = auto_entropy
        if self.auto_entropy:
            self.target_entropy = -0.5 * n_actions
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=1e-4)
            self.alpha = max(self.log_alpha.exp().item(), 0.05)
        else:
            self.alpha = alpha_init

        self.q1 = T.tensor(0.0)
        self.q2 = T.tensor(0.0)
        self.log_probs = T.tensor(0.0)

    def choose_action(self, observation):
        state = T.tensor(
            np.asarray([observation], dtype=np.float32),
             device=self.actor.device,
        )
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def choose_action_deterministic(self, observation):
        state = T.tensor(
            np.asarray([observation], dtype=np.float32),
            device=self.actor.device,
        )
        mu, _ = self.actor.forward(state)
        scale = T.as_tensor(self.actor.max_action, dtype=mu.dtype, device=self.actor.device)
        action = T.tanh(mu) * scale
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        with T.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        print('saving models:')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('loading models:')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device).view(-1)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device).view(-1)

        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=True)
            next_log_probs = next_log_probs.view(-1)

            q1_target = self.target_critic_1.forward(state_, next_actions).view(-1)
            q2_target = self.target_critic_2.forward(state_, next_actions).view(-1)
            q_target = T.min(q1_target, q2_target) - self.alpha * next_log_probs
            q_target[done] = 0.0

            q_hat = self.scale * reward + self.gamma * q_target

        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = F.mse_loss(q1, q_hat)
        critic_2_loss = F.mse_loss(q2, q_hat)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        q1_new = self.critic_1.forward(state, new_actions).view(-1)
        q2_new = self.critic_2.forward(state, new_actions).view(-1)
        critic_value = T.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - critic_value).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.q1 = q1.detach()
        self.q2 = q2.detach()
        self.log_probs = log_probs.detach()

        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        self.update_network_parameters()