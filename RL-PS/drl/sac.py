import copy
import gymnasium as gym
import numpy as np
import torch
from drl.agent import DrlAgent, TargetNetMixin
from drl.networks import SACActorNet, DDPGCriticNet
from drl.shared_code.processing import batch_to_tensors
from drl.shared_code.memory import (ReplayMemory, PrioritizedReplayMemory)
from drl.shared_code.exploration import GaussianNoise


class Sac(DrlAgent, TargetNetMixin):
    def __init__(self, env, memory_size=500000,
                 gamma=0.99, batch_size=256, tau=0.001, start_train=300,
                 actor_fc_dims=(256, 256, 256), critic_fc_dims=(256, 256, 256),
                 actor_learning_rate=0.0001, critic_learning_rate=0.0005,
                 entropy_learning_rate=0.0001,
                 train_interval=1, train_steps=1,
                 optimizer='Adam', fixed_alpha=0.005, target_entropy=None,
                 grad_clip=None, layer_norm=True, *args, **kwargs):
        self.start_train = max(start_train, batch_size)
        actor_fc_dims = list(actor_fc_dims)
        critic_fc_dims = list(critic_fc_dims)

        assert isinstance(env.action_space, gym.spaces.Box)
        # Actor only outputs tanh action space [-1, 1] -> rescale
        env = gym.wrappers.RescaleAction(env, -1, 1)
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super().__init__(env, gamma=gamma, *args, **kwargs)

        self.tau = tau
        self.update_counter = 0
        self.batch_size = batch_size
        self.batch_idxs = np.arange(self.batch_size, dtype=np.int32)
        self.grad_clip = grad_clip
        self.train_interval = train_interval
        self.train_steps = train_steps


        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        self._init_networks(actor_fc_dims, actor_learning_rate,
                            critic_fc_dims, critic_learning_rate,
                            optimizer, layer_norm)

        self.device = self.actor.device
        self._init_memory(memory_size)

        self.fixed_alpha = fixed_alpha
        if self.fixed_alpha:
            self.alpha = torch.tensor(fixed_alpha)
        else:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=entropy_learning_rate)
            self.alpha = self.log_alpha.exp()

        self.training_metrics = {
            'episode_rewards': [],
            'batch_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'timesteps': []
        }

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, layer_norm):
        self.actor = SACActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
            optimizer=optimizer, output_activation='tanh', layer_norm=layer_norm)

        self.critic1 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic2_target = copy.deepcopy(self.critic2)

    def _init_memory(self, memory_size: int):
        self.memory = ReplayMemory(
            memory_size, self.n_obs, self.n_act, n_rewards=1)

    @torch.no_grad()
    def act(self, obs):
        """ Use actor to create actions with exploration. """
        if len(self.memory) < self.start_train:
            return self.env.action_space.sample()

        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        return np.clip(self.actor(obs, act_only=True).cpu().numpy(), -1, 1)

    @torch.no_grad()
    def test_act(self, obs, deterministic=True):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        # Return only the mean, deterministic action for testing
        return np.clip(
            self.actor(obs, act_only=True, deterministic=deterministic)
            .cpu().numpy(), -1, 1)

    def learn(self, obs, act, reward, next_obs, done,
              state=None, next_state=None, info=None, env_idx=0):
        self.remember(obs, act, reward, next_obs, done, info)

        if len(self.memory) < self.start_train:
            return

        if self.step % self.train_interval == 0:
            for i in range(self.train_steps):
                batch = self.memory.sample_random_batch(self.batch_size)
                batch = batch_to_tensors(batch, self.device, continuous=True)
                obss, acts, rewards, next_obss, dones = batch

                self.training_metrics['episode_rewards'].append(rewards.mean())
                self._learn(obss, acts, rewards, next_obss, dones)

    def remember(self, obs, action, reward, next_obs, done, info=None):
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)
        self._train_actor(obss, acts, rewards, next_obss, dones)

        if not self.fixed_alpha:
            self._update_alpha(obss)

        self._soft_target_update(self.critic1, self.critic1_target, self.tau)
        self._soft_target_update(self.critic2, self.critic2_target, self.tau)

        batch_avg_reward = rewards.mean().item()
        self.training_metrics['batch_rewards'].append(batch_avg_reward)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        targets = self._compute_targets(next_obss, dones, rewards)

        # Train critic1
        self.critic1.optimizer.zero_grad()
        q_values1 = self.critic1(obss, acts)
        critic1_loss = self.critic1.loss(targets, q_values1)
        critic1_loss.backward()
        self.training_metrics['critic_losses'].append(critic1_loss.item())
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic1.parameters(), self.grad_clip)
        self.critic1.optimizer.step()

        # Train critic2
        self.critic2.optimizer.zero_grad()
        q_values2 = self.critic2(obss, acts)
        critic2_loss = self.critic2.loss(targets, q_values2)
        critic2_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic2.parameters(), self.grad_clip)
        self.critic2.optimizer.step()

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        self.actor.optimizer.zero_grad()
        entropy, new_acts = self.actor.forward(obss)
        q_values = torch.minimum(self.critic1(obss, new_acts).sum(axis=1),
                                 self.critic2(obss, new_acts).sum(axis=1))

        actor_loss = -(q_values + self.alpha.detach() * entropy).mean()
        actor_loss.backward()
        self.training_metrics['actor_losses'].append(actor_loss.item())
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()

    def _update_alpha(self, obss):
        entropy, _ = self.actor.forward(obss)
        self.alpha_optimizer.zero_grad()

        alpha_loss = -((self.target_entropy - entropy.detach())
                       * self.log_alpha).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.training_metrics['alpha_losses'].append(alpha_loss.item())

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        next_entropy, next_acts = self.actor.forward(next_obss)
        target_values1 = self.critic1_target(next_obss, next_acts)
        target_values2 = self.critic2_target(next_obss, next_acts)
        target_values = torch.minimum(target_values1, target_values2)

        target_values += (self.alpha.item() * next_entropy).reshape(-1, 1)

        target_values[dones == 1.0] = 0.0

        return rewards + self.gamma * target_values

    def store_model(self):
        torch.save(self.actor.state_dict(), self.path + 'actor.pth')
        torch.save(self.critic1.state_dict(), self.path + 'critic1.pth')
        torch.save(self.critic2.state_dict(), self.path + 'critic2.pth')

    def load_model(self):
        actor_weight_dict = torch.load(
            self.path + 'actor.pth', map_location=torch.device(self.device))
        self.actor.load_state_dict(actor_weight_dict)

        critic1_weight_dict = torch.load(
            self.path + 'critic1.pth', map_location=torch.device(self.device))
        self.critic1.load_state_dict(critic1_weight_dict)
        self.critic1_target.load_state_dict(critic1_weight_dict)

        critic2_weight_dict = torch.load(
            self.path + 'critic2.pth', map_location=torch.device(self.device))
        self.critic2.load_state_dict(critic2_weight_dict)
        self.critic2_target.load_state_dict(critic2_weight_dict)

    def log_episode_reward(self, total_reward):
        self.training_metrics['episode_rewards'].append(total_reward)

    def plot_training_curve(self, window=100):
        import matplotlib.pyplot as plt
        import numpy as np

        actor_losses = np.array(self.training_metrics.get('actor_losses', []), dtype=float)
        critic_losses = np.array(self.training_metrics.get('critic_losses', []), dtype=float)
        alpha_losses = np.array(self.training_metrics.get('alpha_losses', []), dtype=float)

        if actor_losses.size == 0 and critic_losses.size == 0 and alpha_losses.size == 0:
            print("No training metrics to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        if actor_losses.size > 0:
            ax.plot(actor_losses, label='Actor Loss')
        if critic_losses.size > 0:
            ax.plot(critic_losses, label='Critic Loss')
        if alpha_losses.size > 0:
            ax.plot(alpha_losses, label='Alpha Loss')
        ax.set_title('SAC Training Losses')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')

        if (actor_losses.min(initial=np.inf) > 0 and
                critic_losses.min(initial=np.inf) > 0 and
                alpha_losses.min(initial=np.inf) > 0):
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        ax.legend()

        ax1 = axes[1]
        ax2 = ax1.twinx()
        if actor_losses.size > 0:
            ax1.plot(actor_losses, color='C0', label='Actor Loss')
        if critic_losses.size > 0:
            ax2.plot(critic_losses, color='C1', label='Critic Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Actor Loss', color='C0')
        ax2.set_ylabel('Critic Loss', color='C1')

        if actor_losses.min(initial=np.inf) > 0:
            ax1.set_yscale('log')
        else:
            ax1.set_yscale('linear')
        if critic_losses.min(initial=np.inf) > 0:
            ax2.set_yscale('log')
        else:
            ax2.set_yscale('linear')
        axes[1].set_title('Dual Axis Loss Comparison')

        plt.tight_layout()
        plt.show()

    def train(self, n_steps: int):
        """Standard training loop for multi-step environments."""
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(1, n_steps + 1):
            action = self.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.learn(obs, action, reward, next_obs, terminated)

            episode_reward += reward

            if terminated or truncated:
                self.log_episode_reward(episode_reward)
                episode_reward = 0.0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            if step % 1000 == 0:
                print(f"Step {step}/{n_steps}, Memory size: {len(self.memory)}")

        print(f"Training completed: {n_steps} steps")


class SacPer(Sac):
    """SAC with Prioritized Experience Replay."""

    def _init_memory(self, memory_size: int):
        self.memory = PrioritizedReplayMemory(
            memory_size, self.n_obs, self.n_act, n_rewards=1)

    def learn(self, obs, act, reward, next_obs, done,
              state=None, next_state=None, info=None, env_idx=0):
        self.remember(obs, act, reward, next_obs, done, info)

        if len(self.memory) < self.start_train:
            return

        if self.step % self.train_interval == 0:
            for i in range(self.train_steps):
                batch, indices, weights = self.memory.sample_random_batch(self.batch_size)
                weights = torch.tensor(weights).to(self.device)
                batch = batch_to_tensors(batch, self.device, continuous=True)
                obss, acts, rewards, next_obss, dones = batch

                self.training_metrics['episode_rewards'].append(rewards.mean())
                self._learn_per(obss, acts, rewards, next_obss, dones, weights, indices)

    def _learn_per(self, obss, acts, rewards, next_obss, dones, weights, indices):
        # Train critics with PER
        targets = self._compute_targets(next_obss, dones, rewards)

        # Train critic1 with PER
        self.critic1.optimizer.zero_grad()
        q_values1 = self.critic1(obss, acts)
        td_errors1 = targets - q_values1
        critic1_loss = (td_errors1.pow(2) * weights).mean()
        critic1_loss.backward()
        self.training_metrics['critic_losses'].append(critic1_loss.item())
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
        self.critic1.optimizer.step()

        # Train critic2 with PER
        self.critic2.optimizer.zero_grad()
        q_values2 = self.critic2(obss, acts)
        td_errors2 = targets - q_values2
        critic2_loss = (td_errors2.pow(2) * weights).mean()
        critic2_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
        self.critic2.optimizer.step()

        # Update priorities
        td_errors = torch.min(td_errors1.abs(), td_errors2.abs())
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Train actor and alpha (same as original)
        self._train_actor(obss, acts, rewards, next_obss, dones)

        if not self.fixed_alpha:
            self._update_alpha(obss)

        self._soft_target_update(self.critic1, self.critic1_target, self.tau)
        self._soft_target_update(self.critic2, self.critic2_target, self.tau)

        batch_avg_reward = rewards.mean().item()
        self.training_metrics['batch_rewards'].append(batch_avg_reward)


class SacExploration(Sac):
    """SAC with enhanced exploration."""

    def __init__(self, env, exploration_noise=0.3, min_exploration=0.01,
                 exploration_decay=0.9995, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.exploration_noise = exploration_noise
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay

    @torch.no_grad()
    def act(self, obs):
        """Use actor with exploration noise."""
        if len(self.memory) < self.start_train:
            return self.env.action_space.sample()

        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        action = self.actor(obs_tensor, act_only=True).cpu().numpy()

        # Add exploration noise
        if self.exploration_noise > self.min_exploration:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            self.exploration_noise *= self.exploration_decay

        return np.clip(action, -1, 1)