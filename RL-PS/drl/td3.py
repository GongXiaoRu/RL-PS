import copy
import gymnasium as gym
import numpy as np
import torch
from drl.agent import DrlAgent, TargetNetMixin
from drl.networks import DDPGActorNet, DDPGCriticNet
from drl.shared_code.processing import batch_to_tensors
from drl.shared_code.memory import (ReplayMemory, PrioritizedReplayMemory)
from drl.shared_code.exploration import GaussianNoise


class Td3(DrlAgent, TargetNetMixin):
    def __init__(self, env, memory_size=500000,
                 gamma=0.99, batch_size=256, tau=0.001, start_train=300,
                 actor_fc_dims=(256, 256, 256), critic_fc_dims=(256, 256, 256),
                 actor_learning_rate=0.0001, critic_learning_rate=0.0005,
                 train_interval=1, train_steps=1,
                 noise_std_dev=0.1, output_activation='tanh',
                 grad_clip=None, layer_norm=True,
                 update_delay=2, target_noise_std_dev=0.2, noise_clip=0.5,
                 *args, **kwargs):

        self.start_train = max(start_train, batch_size)
        actor_fc_dims = list(actor_fc_dims)
        critic_fc_dims = list(critic_fc_dims)

        assert isinstance(env.action_space, gym.spaces.Box)
        if output_activation == 'tanh':
            env = gym.wrappers.RescaleAction(env, -1, 1)
            self.min_range = -1
        elif output_activation == 'sigmoid':
            env = gym.wrappers.RescaleAction(env, 0, 1)
            self.min_range = 0
        env.action_space.seed(kwargs['seed'])

        super().__init__(env, gamma=gamma, *args, **kwargs)

        self.tau = tau
        self.update_counter = 0
        self.batch_size = batch_size
        self.batch_idxs = np.arange(self.batch_size, dtype=np.int32)
        self.grad_clip = grad_clip
        self.train_interval = train_interval
        self.train_steps = train_steps
        self.update_delay = update_delay
        self.target_noise_std_dev = target_noise_std_dev
        self.noise_clip = noise_clip

        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        self._init_networks(actor_fc_dims, actor_learning_rate,
                            critic_fc_dims, critic_learning_rate,
                            output_activation, **kwargs)

        self.device = self.actor.device
        self._init_memory(memory_size)
        self._init_noise(noise_std_dev)

        self.training_metrics = {
            'episode_rewards': [],
            'batch_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'timesteps': []
        }

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate,
                       output_activation, **kwargs):
        self.actor = DDPGActorNet(
            self.n_obs, actor_fc_dims, self.n_act,
            actor_learning_rate,
            output_activation=output_activation, **kwargs)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1 = DDPGCriticNet(
            self.n_obs, self.n_act,
            critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, **kwargs)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = DDPGCriticNet(
            self.n_obs, self.n_act,
            critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, **kwargs)
        self.critic2_target = copy.deepcopy(self.critic2)

    def _init_memory(self, memory_size: int):
        self.memory = ReplayMemory(
            memory_size, self.n_obs, self.n_act, n_rewards=1)

    def _init_noise(self, std_dev):
        self.noise = GaussianNoise((self.n_act,), std_dev)

    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if len(self.memory) < self.start_train:
            act = self.env.action_space.sample()
            return act
        action = self.test_act(obs)
        action += self.noise()
        return np.clip(action, self.min_range, 1)

    @torch.no_grad()
    def test_act(self, obs):
        return self.actor.forward(torch.tensor(obs, dtype=torch.float).to(self.device)).cpu().numpy()

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

        if self.update_counter % self.update_delay == 0:
            self._train_actor(obss, acts, rewards, next_obss, dones)
            self._soft_target_update(self.actor, self.actor_target, self.tau)
            self._soft_target_update(self.critic1, self.critic1_target, self.tau)
            self._soft_target_update(self.critic2, self.critic2_target, self.tau)

        self.update_counter += 1

        batch_avg_reward = rewards.mean().item()
        self.training_metrics['batch_rewards'].append(batch_avg_reward)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        # Train critic1
        self.critic1.optimizer.zero_grad()
        q_values1 = self.critic1(obss, acts)
        targets = self._compute_targets(next_obss, dones, rewards)
        critic1_loss = self.critic1.loss(targets, q_values1)
        critic1_loss.backward()
        self.training_metrics['critic_losses'].append(critic1_loss.item())
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
        self.critic1.optimizer.step()

        # Train critic2
        self.critic2.optimizer.zero_grad()
        q_values2 = self.critic2(obss, acts)
        critic2_loss = self.critic2.loss(targets, q_values2)
        critic2_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
        self.critic2.optimizer.step()

        return targets

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic1(obss, self.actor(obss)).sum(axis=1).mean()
        actor_loss.backward()
        self.training_metrics['actor_losses'].append(actor_loss.item())
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        next_acts = self.actor_target(next_obss)
        noise = (torch.randn_like(next_acts) * self.target_noise_std_dev).clamp(-self.noise_clip, self.noise_clip)
        next_acts = (next_acts + noise).clamp(self.min_range, 1)

        target_values1 = self.critic1_target(next_obss, next_acts)
        target_values2 = self.critic2_target(next_obss, next_acts)
        target_values = torch.minimum(target_values1, target_values2)
        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)

    def store_model(self):
        torch.save(self.actor.state_dict(), self.path + 'actor.pth')
        torch.save(self.critic1.state_dict(), self.path + 'critic1.pth')
        torch.save(self.critic2.state_dict(), self.path + 'critic2.pth')

    def load_model(self):
        actor_weight_dict = torch.load(
            self.path + 'actor.pth', map_location=torch.device(self.device))
        self.actor.load_state_dict(actor_weight_dict)
        self.actor_target.load_state_dict(actor_weight_dict)

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

        if actor_losses.size == 0 and critic_losses.size == 0:
            print("No training metrics to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        if actor_losses.size > 0:
            ax.plot(actor_losses, label='Actor Loss')
        if critic_losses.size > 0:
            ax.plot(critic_losses, label='Critic Loss')
        ax.set_title('Training Losses')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')

        if actor_losses.min(initial=np.inf) > 0 and critic_losses.min(initial=np.inf) > 0:
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