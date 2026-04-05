""" Idea: When we have multiple critics in TD3 and SAC, why do we need the 
target nets? Cant we simply use the other critic as the target?"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from drl.ddpg import Td3
from drl.sac import Sac


class TargetlessTd3(Td3):
    def __init__(self, env, use_actor_target=False, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        # Define if at least the actor target is still used (for ablation study)
        self.use_actor_target = use_actor_target

    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)

        if self.step % self.update_delay == 0:
            self._train_actor(obss, acts, rewards, next_obss, dones)
            if self.use_actor_target:
                self._soft_target_update(self.actor, self.actor_target, self.tau)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        for critic, target_critic in zip(
                (self.critic, self.critic2), (self.critic, self.critic2)):
            critic.optimizer.zero_grad()
            q_values = self.critic(obss, acts)
            targets = self._compute_targets(next_obss, dones, rewards, target_critic)
            critic_loss = critic.loss(targets, q_values)
            critic_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), self.grad_clip)
            critic.optimizer.step()

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards, target_critic):
        """ Add noise to target actions and use min q value from 2 critics. """
        if self.use_actor_target:
            next_acts = self.actor_target(next_obss)
        else:
            next_acts = self.actor(next_obss)
        noise = (
            torch.randn_like(next_acts) * self.target_noise_std_dev
        ).clamp(-self.noise_clip, self.noise_clip)
        next_acts = (next_acts + noise).clamp(self.min_range, 1)

        target_values = target_critic(next_obss, next_acts)
        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)

    def store_model(self):
        # Remove this to save space (not required for quick experiments)
        pass

    def load_model(self):
        pass


class TargetlessSac(Sac):
    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)

        self._train_actor(obss, acts, rewards, next_obss, dones)

        if not self.fixed_alpha:
            self._update_alpha(obss)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        for critic, target_critic in zip(
                (self.critic1, self.critic2), (self.critic2, self.critic1)):
            targets = self._compute_targets(next_obss, dones, rewards, target_critic)
            critic.optimizer.zero_grad()
            q_values = critic(obss, acts)
            critic_loss = critic.loss(targets, q_values)
            critic_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), self.grad_clip)
            critic.optimizer.step()

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards, target_critic):
        next_entropy, next_acts = self.actor.forward(next_obss)
        target_values = target_critic(next_obss, next_acts)
        target_values += self.alpha.item() * next_entropy
        target_values[dones == 1.0] = 0.0

        return rewards + self.gamma * target_values

    def store_model(self):
        pass

    def load_model(self):
        pass