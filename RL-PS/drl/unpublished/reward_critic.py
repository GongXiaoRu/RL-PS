""" Idea: Minor update to DQN/DDPG to predict two values R and Q_t+1 instead of 
only Q_t"""


import copy

import numpy as np
import torch

from drl.ddpg import Ddpg
from drl.networks import SACActorNet, DDPGCriticNet
from drl.networks import DDPGActorNet, DDPGCriticNet
from drl.shared_code.processing import batch_to_tensors


class RewardCriticDdpg(Ddpg):
    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate,
                       output_activation, **kwargs):
        self.actor = DDPGActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
            output_activation=output_activation, **kwargs)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=2, **kwargs)
        self.critic_target = copy.deepcopy(self.critic)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        self.critic.optimizer.zero_grad()
        reward_qs = self.critic(obss, acts)
        targets = self._compute_targets(next_obss, dones, rewards)
        critic_loss = self.critic.loss(targets, reward_qs)
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_clip)
        self.critic.optimizer.step()
        return targets

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        """ Do not sum reward and q_t+1, but return them separately """
        next_acts = self.actor_target(next_obss)
        target_values = self.critic_target(next_obss, next_acts).sum(axis=1, keepdim=True)
        target_values[dones == 1.0] = 0.0
        return torch.cat((rewards, target_values), axis=1)
