""" 
Implementation of Unrolling Safety Layer (USL): 
https://ojs.aaai.org/index.php/AAAI/article/view/26786 

"""

from collections.abc import Iterable
import copy

import numpy as np
import torch

from drl.sac import Sac
from drl.shared_code.memory import ReplayMemory
from drl.networks import DDPGCriticNet, SACActorNet


class UslSac(Sac):
    def __init__(self, env, n_penalties=1, penalty_factor=5, 
                 max_projection_iterations=20, projection_lr=0.01,
                 max_costs=0, extra_cost_critic=False,
                 use_correction_optimizer: str=None,
                 unconstrained_exploration=False,
                 exploration_decay=0.99999, 
                 *args, **kwargs):
        self.n_penalties = n_penalties
        self.penalty_factor = penalty_factor
        self.max_projection_iterations = max_projection_iterations
        self.projection_lr = projection_lr
        self.max_costs = (np.array(max_costs) 
                          if isinstance(max_costs, Iterable) 
                          else max_costs)
        
        # Original implementation: Use separate critic for cost prediction.
        # My approach: Use the same critic for both (n_outputs)
        self.extra_cost_critic = extra_cost_critic

        # Deviation from original implementation: Use optimizer for projection
        self.use_correction_optimizer = use_correction_optimizer 

        # Deviation from original implementation: Use unconstrained exploration
        # Create a second actor that is only trained for reward maximization
        self.unconstrained_exploration = unconstrained_exploration
        self.exploration_probability = 1.0
        self.exploration_decay = exploration_decay

        super().__init__(env, *args, **kwargs)

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, layer_norm):
        if self.extra_cost_critic:
            self.n_rewards = 1
            # Use separate critic for cost prediction
            self.c_critic1 = DDPGCriticNet(
                self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
                n_rewards=self.n_penalties, layer_norm=layer_norm)
            self.c_critic1_target = copy.deepcopy(self.c_critic1)

            self.c_critic2 = DDPGCriticNet(
                self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
                n_rewards=self.n_penalties, layer_norm=layer_norm)
            self.c_critic2_target = copy.deepcopy(self.c_critic2)
        else:
            self.n_rewards = 1 + self.n_penalties

        super()._init_networks(actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, layer_norm)
        
        if self.unconstrained_exploration:
            self.unconstrained_actor = SACActorNet(
                self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
                optimizer=optimizer, output_activation='tanh', layer_norm=layer_norm)

    def _init_memory(self, memory_size: int):
        self.memory = ReplayMemory(
            memory_size, self.n_obs, self.n_act, n_rewards=1+self.n_penalties)

    def remember(self, obs, act, reward, next_obs, done, info):
        # Costs as "rewards"
        costs = -info['cost']
        if isinstance(costs, Iterable):
            costs = np.concatenate(([reward], np.array(costs)))
        else:
            rewards = np.array([reward, costs])
        super().remember(obs, act, rewards, next_obs, done)

    @torch.no_grad()
    def act(self, obs):
        if self.memory.memory_counter < self.start_train:
            return self.env.action_space.sample()

        if self.unconstrained_exploration:
            self.exploration_probability *= self.exploration_decay
            if np.random.rand() < self.exploration_probability:
                obs = torch.tensor(obs, dtype=torch.float).to(self.device)
                return np.clip(self.unconstrained_actor(obs, act_only=True).cpu().numpy(), -1, 1)

        return super().act(obs)
        

    def test_act(self, obs, deterministic=True, correction=True):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        # Return only the mean, deterministic action for testing
        act = self.actor(obs, act_only=True, deterministic=deterministic)

        if not correction:
            return np.clip(act.cpu().detach().numpy(), -1, 1)

        act = act.reshape(1, -1) 
        obs = obs.reshape(1, -1)

        print('original act: ', np.clip(act.flatten().cpu().detach().numpy(), -1, 1))

        if self.extra_cost_critic:
            critic1 = self.c_critic1_target
            critic2 = self.c_critic2_target
        else:
            critic1 = self.critic1_target
            critic2 = self.critic2_target

        # Perform safety optimization (stage 2)
        if self.use_correction_optimizer:
            # Own implementation
            act = torch.autograd.Variable(act.detach(), requires_grad=True)
            optimizer_class = getattr(torch.optim, self.optimizer)
            optimizer = optimizer_class([act], lr=self.projection_lr)
            for _ in range(self.max_projection_iterations):
                optimizer.zero_grad()
                
                c_values = -torch.minimum(critic1(obs, act), critic2(obs, act))
                if not self.extra_cost_critic:
                    # Only look at costs here
                    c_values = c_values[:, 1:]

                c_values = torch.relu(c_values - self.max_costs)
                if c_values.sum() < 1e-9:
                    # No constraint violations
                    break
                c_values.sum().backward()
                optimizer.step()
        else:
            # Original implementation
            for _ in range(self.max_projection_iterations):
                act = torch.autograd.Variable(
                    torch.clamp(act.detach(), -1, 1), requires_grad=True)

                # Use both critics for worst-case cost prediction
                # Idea to use the targets here stolen from original implementation
                c_values = -torch.minimum(critic1(obs, act), critic2(obs, act))

                if not self.extra_cost_critic:  
                    # Only look at costs here
                    c_values = c_values[:, 1:]

                # Apply relu function to omit negative values
                c_values = torch.relu(c_values - self.max_costs).sum()

                if c_values < 1e-9:
                    # No constraint violations
                    break
                
                # Apply max normalized gradient, equation (5) in the paper
                c_values.backward()
                normalization = torch.norm(act.grad, np.inf).detach() + 1e-9
                act = act - self.projection_lr * act.grad / normalization 

        print('final act: ', np.clip(act.flatten().cpu().detach().numpy(), -1, 1))

        return np.clip(act.flatten().cpu().detach().numpy(), -1, 1)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        super()._learn(obss, acts, rewards, next_obss, dones)
        if self.extra_cost_critic:
            self._soft_target_update(self.c_critic1, self.c_critic1_target, self.tau)
            self._soft_target_update(self.c_critic2, self.c_critic2_target, self.tau)

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        self.actor.optimizer.zero_grad()
        entropy, acts = self.actor.forward(obss)

        if self.extra_cost_critic:
            c_values = torch.minimum(self.c_critic1(obss, acts),
                                     self.c_critic2(obss, acts))
            q_values = torch.minimum(self.critic1(obss, acts),
                                     self.critic2(obss, acts))
        else:
            cq_values = torch.minimum(self.critic1(obss, acts),
                                      self.critic2(obss, acts))
        
            q_values = cq_values[:, 0:1]
            c_values = cq_values[:, 1:]

        reward_loss = -(q_values + self.alpha.detach() * entropy).mean()

        # Compute penalties to perform stage 1 optimization
        violations = -c_values - self.max_costs
        violations[violations < 0] = 0.0
        cost_loss = self.penalty_factor * violations.sum(dim=1).mean()

        actor_loss = reward_loss + cost_loss
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()

        if self.unconstrained_exploration:
            self.unconstrained_actor.optimizer.zero_grad()
            entropy, acts = self.unconstrained_actor.forward(obss)
            if self.extra_cost_critic:
                q_values = torch.minimum(self.critic1(obss, acts),
                                         self.critic2(obss, acts))
            else:
                q_values = torch.minimum(self.critic1(obss, acts)[:, 0:1],
                                         self.critic2(obss, acts)[:, 0:1])
            actor_loss = -(q_values + self.alpha.detach() * entropy).mean()
            actor_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.unconstrained_actor.parameters(), self.grad_clip)
            self.unconstrained_actor.optimizer.step()

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        if self.extra_cost_critic:
            super()._train_critic(obss, acts, rewards[:,0:1], next_obss, dones)
            targets = rewards[:, 1:]
            for critic in (self.c_critic1, self.c_critic2):
                critic.optimizer.zero_grad()
                c_values = critic(obss, acts)
                critic_loss = critic.loss(targets, c_values)
                critic_loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), self.grad_clip)
                critic.optimizer.step()
        else:
            super()._train_critic(obss, acts, rewards, next_obss, dones)

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        target_values = super()._compute_targets(next_obss, dones, rewards)

        if not self.extra_cost_critic:
            # Costs are single-step only (no accumulation)
            target_values[:, 1:] = rewards[:, 1:]
        
        return target_values
