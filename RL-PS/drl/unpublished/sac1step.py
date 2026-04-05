""" Special variants of SAC for 1-step environment (done always True). """

import numpy as np
import torch
from drl.sac import Sac


class Sac1Step(Sac):
    """SAC variant for 1-step RL problems with simplifications."""

    def __init__(self, env, memory_size=1000000,
                 gamma=0.999, batch_size=256, tau=0.0001, start_train=300,
                 actor_fc_dims=(256, 256, 256), critic_fc_dims=(256, 256, 256),
                 actor_learning_rate=0.0028, critic_learning_rate=0.0035,
                 entropy_learning_rate=0.0001,
                 train_interval=1, train_steps=1,
                 optimizer='Adam', fixed_alpha=0.005 , target_entropy=None,
                 grad_clip=None, layer_norm=True, exploration_noise=0.1,
                 *args, **kwargs):

        # 调用父类初始化
        super().__init__(
            env, memory_size=memory_size, gamma=gamma, batch_size=batch_size,
            tau=tau, start_train=start_train, actor_fc_dims=actor_fc_dims,
            critic_fc_dims=critic_fc_dims, actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate, entropy_learning_rate=entropy_learning_rate,
            train_interval=train_interval, train_steps=train_steps, optimizer=optimizer,
            fixed_alpha=fixed_alpha, target_entropy=target_entropy, grad_clip=grad_clip,
            layer_norm=layer_norm, *args, **kwargs
        )

        # 关键修复：确保动作维度正确设置为24小时
        self.n_act = 24

        # 添加探索噪声参数
        self.exploration_noise = exploration_noise
        self.min_exploration = 0.1
        self.exploration_decay = 0.9995
    @torch.no_grad()
    def act(self, obs):
        """Use actor to create actions with exploration noise."""
        if len(self.memory) < self.start_train:
            return self.env.action_space.sample()

        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        action = self.actor(obs_tensor, act_only=True).cpu().numpy()

        # 添加探索噪声（类似DDPG）
        if self.exploration_noise > self.min_exploration:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            self.exploration_noise *= self.exploration_decay

        return np.clip(action, -1, 1)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        """Simplified learning for 1-step environments."""
        self._train_critic(obss, acts, rewards, next_obss, dones)
        self._train_actor(obss, acts, rewards, next_obss, dones)

        if not self.fixed_alpha:
            self._update_alpha(obss)

        self._soft_target_update(self.critic1, self.critic1_target, self.tau)
        self._soft_target_update(self.critic2, self.critic2_target, self.tau)

        batch_avg_reward = rewards.mean().item()
        self.training_metrics['batch_rewards'].append(batch_avg_reward)

    def _compute_targets(self, next_obss, dones, rewards):
        """保持SAC理论正确性的目标值计算"""
        with torch.no_grad():
            next_entropy, next_acts = self.actor.forward(next_obss)
            target_values1 = self.critic1_target(next_obss, next_acts)
            target_values2 = self.critic2_target(next_obss, next_acts)
            target_values = torch.minimum(target_values1, target_values2)

            # 保持SAC的熵正则化
            target_values += (self.alpha.item() * next_entropy).reshape(-1, 1)
            target_values[dones == 1.0] = 0.0

            return rewards + self.gamma * target_values

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        """Train critic with debug information."""
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

        # Debug information
        if hasattr(self, '_train_step') and self._train_step % 100 == 0:
            with torch.no_grad():
                q1_mean = q_values1.mean().item()
                q2_mean = q_values2.mean().item()
                targets_mean = targets.mean().item()
                print(f"Step {self._train_step}: Q1={q1_mean:.3f}, Q2={q2_mean:.3f}, Targets={targets_mean:.3f}")
        if not hasattr(self, '_train_step'):
            self._train_step = 0
        self._train_step += 1

    def train(self, n_steps: int):
        """Training loop for 1-step environments."""
        obs, _ = self.env.reset()
        episode_reward = 0.0

        for step in range(1, n_steps + 1):
            action = self.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Store experience
            self.remember(obs, action, reward, next_obs, terminated)

            # Learn if enough samples
            if len(self.memory) >= self.start_train:
                self.learn(obs, action, reward, next_obs, terminated)

            episode_reward += reward

            if terminated or truncated:
                self.log_episode_reward(episode_reward)
                # Store line loss for plotting
                if 'total_loss_with_storage' in info:
                    self.training_metrics.setdefault('line_losses', []).append(info['total_loss_with_storage'])
                else:
                    self.training_metrics.setdefault('line_losses', []).append(-episode_reward)

                episode_reward = 0.0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            # Progress reporting
            if step % 100 == 0:
                mem_size = len(self.memory)
                train_status = "training" if mem_size >= self.start_train else "collecting samples"
                print(f"Step {step}/{n_steps}: Memory={mem_size}, {train_status}")

        print(f"Train finished: recorded "
              f"{len(self.training_metrics.get('episode_rewards', []))} episodes, "
              f"{len(self.training_metrics.get('actor_losses', []))} loss-points.")

    def plot_line_loss(self):
        """Plot the true line loss per episode."""
        import matplotlib.pyplot as plt
        losses = self.training_metrics.get('line_losses', [])
        if not losses:
            print("No line losses to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', alpha=0.7, label='Line Loss')

        # 添加移动平均线
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')

        plt.xlabel('Episode')
        plt.ylabel('Line Loss (kW)')
        plt.title('Line Loss per Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_training_curve(self, window=100):
        """Enhanced training curves with line loss."""
        import matplotlib.pyplot as plt
        import numpy as np

        # 原有的损失曲线
        actor_losses = np.array(self.training_metrics.get('actor_losses', []), dtype=float)
        critic_losses = np.array(self.training_metrics.get('critic_losses', []), dtype=float)
        alpha_losses = np.array(self.training_metrics.get('alpha_losses', []), dtype=float)
        line_losses = np.array(self.training_metrics.get('line_losses', []), dtype=float)

        if (actor_losses.size == 0 and critic_losses.size == 0 and
            alpha_losses.size == 0 and line_losses.size == 0):
            print("No training metrics to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 损失曲线
        ax = axes[0, 0]
        if actor_losses.size > 0:
            ax.plot(actor_losses, label='Actor Loss')
        if critic_losses.size > 0:
            ax.plot(critic_losses, label='Critic Loss')
        if alpha_losses.size > 0:
            ax.plot(alpha_losses, label='Alpha Loss')
        ax.set_title('SAC Training Losses')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 线损曲线
        ax1 = axes[0, 1]
        if line_losses.size > 0:
            ax1.plot(line_losses, 'g-', label='Line Loss')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Line Loss (kW)', color='g')
            ax1.tick_params(axis='y', labelcolor='g')
            ax1.set_title('Line Loss Progress')
            ax1.grid(True, alpha=0.3)

            # 添加第二个y轴显示奖励
            ax1_twin = ax1.twinx()
            episode_rewards = np.array(self.training_metrics.get('episode_rewards', []), dtype=float)
            if episode_rewards.size > 0:
                ax1_twin.plot(episode_rewards, 'orange', label='Episode Reward')
                ax1_twin.set_ylabel('Episode Reward', color='orange')
                ax1_twin.tick_params(axis='y', labelcolor='orange')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')

        # 双轴损失比较
        ax2 = axes[1, 0]
        ax2_twin = ax2.twinx()
        if actor_losses.size > 0:
            ax2.plot(actor_losses, color='C0', label='Actor Loss')
        if critic_losses.size > 0:
            ax2_twin.plot(critic_losses, color='C1', label='Critic Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Actor Loss', color='C0')
        ax2_twin.set_ylabel('Critic Loss', color='C1')
        ax2.set_title('Dual Axis Loss Comparison')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 探索噪声衰减
        ax3 = axes[1, 1]
        if hasattr(self, 'exploration_noise'):
            # 模拟噪声衰减曲线
            steps = min(1000, len(actor_losses) if actor_losses.size > 0 else 1000)
            noise_levels = [self.exploration_noise * (self.exploration_decay ** i) for i in range(steps)]
            ax3.plot(noise_levels, 'purple', label='Exploration Noise')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Noise Level')
            ax3.set_title('Exploration Noise Decay')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class Sac1StepSpecial(Sac1Step):
    """A variant that uses deterministic actions for better performance in 1-step environments."""

    def __init__(self, env, deterministic_ratio=0.8, *args, **kwargs):
        self.deterministic_ratio = deterministic_ratio
        super().__init__(env, *args, **kwargs)

    @torch.no_grad()
    def act(self, obs):
        """Use a mix of deterministic and stochastic actions."""
        if len(self.memory) < self.start_train:
            return self.env.action_space.sample()

        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)

        # Use deterministic actions with a certain probability for better exploitation
        if np.random.random() < self.deterministic_ratio:
            action = self.actor(obs_tensor, act_only=True, deterministic=True)
        else:
            action = self.actor(obs_tensor, act_only=True, deterministic=False)

        return np.clip(action.cpu().numpy(), -1, 1)