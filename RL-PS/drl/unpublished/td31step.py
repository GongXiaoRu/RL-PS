""" Special variants of TD3 for 1-step environment (done always True). """

import numpy as np
import torch
from drl.shared_code.exploration import GaussianNoise
from drl.td3 import Td3


class Td31Step(Td3):
    """TD3 variant for 1-step RL problems with simplifications."""

    def _learn(self, obss, acts, rewards, next_obss, dones):
        super()._learn(obss, acts, rewards, next_obss, dones)

    def _compute_targets(self, next_obss, dones, rewards):
        # In 1-step environments, targets are just the rewards
        return rewards

    def train(self, n_steps: int):
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(1, n_steps + 1):
            action = self.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.learn(obs, action, reward, next_obs, True)

            episode_reward += reward

            if terminated or truncated:
                self.log_episode_reward(episode_reward)
                self.training_metrics.setdefault('line_losses', []).append(-episode_reward)
                episode_reward = 0.0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        print(f"Train finished: recorded "  
              f"{len(self.training_metrics.get('episode_rewards', []))} episodes, "  
              f"{len(self.training_metrics.get('actor_losses', []))} loss-points.")

    def plot_line_loss(self):
        """Override: draw the true (positive) line loss per episode."""
        import matplotlib.pyplot as plt
        losses = self.training_metrics.get('line_losses', [])
        if not losses:
            print("No line losses to plot.")
            return
        plt.figure(figsize=(8,4))
        plt.plot(losses, label='Line Loss')
        plt.xlabel('Episode')
        plt.ylabel('Line Loss')
        plt.title('Line Loss per Episode')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Td31StepSpecial(Td31Step):
    """A variant that samples multiple actions per step and picks the best."""

    def __init__(self, env, n_samples=10, noise_std_dev=0.2, *args, **kwargs):
        self.n_samples = n_samples
        super().__init__(env, noise_std_dev=noise_std_dev, *args, **kwargs)

    @torch.no_grad()
    def act(self, obs):
        """Sample multiple noisy actions and pick the one with highest Q-value."""
        if len(self.memory) < self.start_train:
            return self.env.action_space.sample()

        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        base_action = self.actor(obs_tensor).cpu()
        # replicate and add noise
        samples = base_action.expand(self.n_samples, -1)
        samples = samples + self.noise()
        # evaluate with critic1 (or you could use both critics and take min/max)
        q_vals = self.critic1(
            obs_tensor.expand(self.n_samples, -1),
            samples.float().to(self.device)
        ).cpu()
        # choose best
        best_idx = q_vals.argmax()
        action = samples[best_idx].numpy()
        return np.clip(action, self.min_range, 1.0)

    def _init_noise(self, std_dev):
        # override to match sample shape
        self.noise = GaussianNoise((self.n_samples, self.n_act), std_dev)