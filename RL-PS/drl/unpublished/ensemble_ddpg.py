""" Idea: Use DDPG with multiple actors and critics. Multiple critic provide an
improved gradient for the actors (see double q learning). Multiple actors
improve exploration and can be combined with standard ensemble learning
techniques. Further, the existence of multiple parallel models allows to use
multiple sets of hyperparameters, e.g. different learning rates.

"""

import copy

import gymnasium as gym
import numpy as np
import torch

from drl.ddpg import Ddpg
from drl.networks import SACActorNet, DDPGCriticNet
from drl.networks import DDPGActorNet, DDPGCriticNet
from drl.shared_code.processing import batch_to_tensors


"""
TODOs:
- Important: Currently actor/critic trained on different data!
- Minor: Somehow inherit from DDPG again to have same default hyperparams
- Add more variants for act() to test different strategies (how to evaluate which actor is best currently?)


Done:
- Computation time is a big problem, because increases with n of Neural Nets
-> Train exactly same number of NNs as DDPG to allow for better comparison?! -> set train_how_many=1

"""


class EnsembleDdpg(Ddpg):
    def __init__(self, env, memory_size=500000,
                 gamma=0.99, batch_size=256, tau=0.001, start_train=2000,
                 actor_fc_dims=(256, 256, 256), critic_fc_dims=(256, 256, 256),
                 actor_learning_rate=0.0001, critic_learning_rate=0.0005,
                 noise_std_dev=0.1, optimizer='Adam', activation='tanh',
                 grad_clip=None, n_critics=2, n_actors=4, train_how_many=1,
                 variant=5, test_variant=4, targetless=False,
                 *args, **kwargs):

        # TODO: Lots of HPs still missing!
        higher_n = max(n_critics, n_actors)
        # if isinstance(batch_size, int):
        #     batch_size = [batch_size] * higher_n
        if isinstance(actor_learning_rate, float):
            actor_learning_rate = [actor_learning_rate] * n_actors
        if isinstance(critic_learning_rate, float):
            critic_learning_rate = [critic_learning_rate] * n_critics
        if isinstance(optimizer, str):
            optimizer = [optimizer] * higher_n
        if isinstance(actor_fc_dims[0], int):
            actor_fc_dims = [list(actor_fc_dims)] * n_actors
        else:
            actor_fc_dims = [list(a) for a in actor_fc_dims]
        if isinstance(critic_fc_dims[0], int):
            critic_fc_dims = [list(critic_fc_dims)] * n_critics    
        else:
            critic_fc_dims = [list(c) for c in critic_fc_dims]  

        self.start_train = max(start_train, batch_size)

        assert isinstance(env.action_space, gym.spaces.Box)

        if activation == 'tanh':  # Multiple ones possible???
            # Actor only outputs tanh action space [-1, 1]
            # Also clips to action space
            env = gym.wrappers.RescaleAction(env, -1, 1)
            self.min_range = -1
        elif activation == 'sigmoid':
            env = gym.wrappers.RescaleAction(env, 0, 1)
            self.min_range = 0
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super(Ddpg, self).__init__(env, gamma, *args, **kwargs)

        self.n_critics = n_critics
        self.n_actors = n_actors
        self.variant = variant
        self.test_variant = test_variant
        self.targetless = targetless
        if targetless:
            assert n_actors > 1 and n_critics > 1 

        # Generally: Different hyperparams make sense only, if there is a method to test which actor is the best
        self.tau = tau  # Multiple ones possible??? -> would result in some targets to be slower. Effects unclear
        # Multiple ones possible??? Not really, because actor/critic get trained together
        self.batch_size = batch_size
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?
        self.grad_clip = grad_clip
        self.train_how_many = train_how_many

        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        self._init_networks(actor_fc_dims, actor_learning_rate,
                            critic_fc_dims, critic_learning_rate, optimizer, activation)

        # TODO: How to do for multiple nets?
        self.device = self.critics[0].device
        self._init_memory(memory_size)
        self._init_noise(noise_std_dev)

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, activation, **kwargs):
        self.actors = [DDPGActorNet(
            self.n_obs, actor_fc_dims[i], self.n_act, actor_learning_rate[i],
            optimizer=optimizer[i], output_activation=activation)
            for i in range(self.n_actors)]

        # TODO: variable optimizer for Critics as well
        self.critics = [DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate[i], critic_fc_dims[i],
            n_rewards=self.n_rewards, optimizer=optimizer[i])
            for i in range(self.n_critics)]
        
        # TODO: Use deep copy instead copy.deepcopy(self.actor)
        if not self.targetless:
            self.actor_targets = [copy.deepcopy(a) for a in self.actors]
            self.critic_targets = [copy.deepcopy(c) for c in self.critics]

    @torch.no_grad()
    def act(self, obs, test=False):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train:
            return self.env.action_space.sample()
        if test is True:
            return self.test_act(obs)

        action = act_variants(self, obs, self.variant)
        action += self.noise()
        return np.clip(action, self.min_range, 1)

    @torch.no_grad()
    def test_act(self, obs, stochastic_policy=False):
        return act_variants(self, obs, variant=self.test_variant)

    def learn(self, obs, act, reward, next_obs, done,
              state=None, next_state=None, info=None, env_idx=0):
        """ Same as normal DDPG """
        # TODO: Consider states as well!
        self.remember(obs, act, reward, next_obs, done)

        if len(self.memory) < self.start_train:
            return

        self._learn()

    def _learn(self):
        # Randomly assign actors to critics
        # The actors get trained with the same data their current critic was trained before
        actor_critic_pairs = zip(
            np.random.choice(self.n_actors, size=self.train_how_many),
            np.random.choice(self.n_critics, size=self.train_how_many))

        for actor_idx, critic_idx in actor_critic_pairs:
            batch = self.memory.sample_random_batch(self.batch_size)
            batch = batch_to_tensors(batch, self.device, continuous=True)
            self._train_critic(critic_idx, actor_idx, *batch)

            self._train_actor(critic_idx, actor_idx, *batch)
            if not self.targetless:
                self._soft_target_update(self.actors[actor_idx], self.actor_targets[actor_idx], self.tau)
                self._soft_target_update(self.critics[critic_idx], self.critic_targets[critic_idx], self.tau)

    def _train_critic(self, critic_idx, actor_idx, obss, acts, rewards, next_obss, dones):
        critic = self.critics[critic_idx]
        critic.optimizer.zero_grad()
        q_values = critic(obss, acts)
        targets = self._compute_targets(
            critic_idx, actor_idx, next_obss, dones, rewards)
        critic_loss = critic.loss(targets, q_values)
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        critic.optimizer.step()
        return targets

    def _train_actor(self, critic_idx, actor_idx, obss, acts, rewards, next_obss, dones):
        actor = self.actors[actor_idx]
        actor.optimizer.zero_grad()
        # TODO: Choose one randomly, or use all of them (mean), or use double DQN stuff?
        critic = self.critics[critic_idx]
        actor_loss = -critic(obss, actor(obss)).sum(axis=1).mean()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.grad_clip)
        actor.optimizer.step()

    @torch.no_grad()
    def _compute_targets(self, critic_idx, actor_idx, next_obss, dones, rewards):
        if self.targetless:
            # Simply use one of the other networks as target net
            idx = actor_idx + 1 if actor_idx < (self.n_actors - 1) else 0
            actor_target = self.actors[idx]
            idx = critic_idx + 1 if critic_idx < (self.n_critics - 1) else 0
            critic_target = self.critics[critic_idx]
        else:
            actor_target = self.actor_targets[actor_idx]
            critic_target = self.critic_targets[critic_idx]

        next_acts = actor_target(next_obss)
        target_values = critic_target(next_obss, next_acts)
        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)

    def store_model(self):
        pass

    def load_model(self):
        pass


class EnsembleSac(Ddpg):
    def __init__(self, env, memory_size=500000,
                 gamma=0.99, batch_size=256, tau=0.001, start_train=2000,
                 actor_fc_dims=(256, 256, 256), critic_fc_dims=(256, 256, 256),
                 actor_learning_rate=0.0002, critic_learning_rate=0.001,
                 optimizer='Adam', fixed_alpha=None,
                 grad_clip=None, layer_norm=True, 
                 n_critics=2, n_actors=4, train_how_many=1, variant=5, 
                 test_variant=4, targetless=False, min_of_all=False,
                 *args, **kwargs):
        # TODO: Train exploration for each actor individually?
        self.n_actors = n_actors
        self.n_critics = n_critics
        self.train_how_many = train_how_many
        self.variant = variant
        self.test_variant = test_variant
        self.targetless = targetless
        if targetless:
            assert n_actors > 1 and n_critics > 1
        self.min_of_all = min_of_all

        self.start_train = max(start_train, batch_size)
        actor_fc_dims = list(actor_fc_dims)
        critic_fc_dims = list(critic_fc_dims)

        # TODO: Discrete actions?!
        assert isinstance(env.action_space, gym.spaces.Box)
        # Actor only outputs tanh action space [-1, 1] -> rescale
        # Also clips to action space
        env = gym.wrappers.RescaleAction(env, -1, 1)
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super(Ddpg, self).__init__(env, gamma, *args, **kwargs)

        self.tau = tau
        self.update_counter = 0
        self.batch_size = batch_size  # Move to superclass?
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?
        self.grad_clip = grad_clip

        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        self._init_networks(list(actor_fc_dims), actor_learning_rate,
                            list(critic_fc_dims), critic_learning_rate,
                            optimizer, layer_norm)

        self.device = self.actors[0].device  # TODO: How to do for multiple nets?
        self._init_memory(memory_size)

        self.fixed_alpha = fixed_alpha
        if self.fixed_alpha:
            self.alpha = torch.tensor(fixed_alpha)
        else:
            # Heuristic from SAC paper, TODO: Maybe allow user to set this?!
            self.target_entropy = -np.prod(
                self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=actor_learning_rate)
            self.alpha = self.log_alpha.exp()

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, layer_norm):
        self.actors = [SACActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
            optimizer=optimizer, output_activation='tanh', layer_norm=layer_norm)
            for _ in range(self.n_actors)]
        # TODO: variable optimizer for Critics as well
        self.critics = [DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
            for _ in range(self.n_critics)]
        if not self.targetless:
            self.critic_targets = [copy.deepcopy(critic) for critic in self.critics]

    @torch.no_grad()
    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train:
            return self.env.action_space.sample()

        return act_variants(self, obs, variant=self.variant, act_only=True)

    @torch.no_grad()
    def test_act(self, obs, deterministic=True):
        # Return only the mean, deterministic action for testing
        return act_variants(
            self, obs, variant=self.test_variant, act_only=True, deterministic=deterministic)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        actor_critic_pairs = zip(
            np.random.choice(self.n_actors, size=self.train_how_many),
            np.random.choice(self.n_critics, size=self.train_how_many))

        for actor_idx, critic_idx in actor_critic_pairs:
            batch = self.memory.sample_random_batch(self.batch_size)
            batch = batch_to_tensors(batch, self.device, continuous=True)
            self._train_critic(actor_idx, critic_idx, *batch)
            self._train_actor(actor_idx, critic_idx, *batch)

        if not self.fixed_alpha:
            self._update_alpha(obss, actor_idx)

        if self.targetless:
            return

        for idx, critic in enumerate(self.critics):
            self._soft_target_update(
                critic, self.critic_targets[idx], self.tau)

    def _train_critic(self, actor_idx, critic_idx, obss, acts, rewards, next_obss, dones):
        targets = self._compute_targets(actor_idx, critic_idx, next_obss, dones, rewards)
        critic = self.critics[critic_idx]
        critic.optimizer.zero_grad()
        q_values = critic(obss, acts)
        critic_loss = critic.loss(targets, q_values)
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), self.grad_clip)
        critic.optimizer.step()

    def _train_actor(self, actor_idx, critic_idx, obss, acts, rewards, next_obss, dones):
        actor = self.actors[actor_idx]
        actor.optimizer.zero_grad()
        entropy, acts = actor.forward(obss)
        q_values = self.critics[critic_idx](obss, acts).sum(axis=1)

        actor_loss = (-q_values - self.alpha * entropy).mean()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), self.grad_clip)
        actor.optimizer.step()

    def _update_alpha(self, obss, actor_index):
        # TODO: Should this be done for every individual actor?!
        entropy, _ = self.actors[actor_index].forward(obss)
        self.alpha_optimizer.zero_grad()

        alpha_loss = -(self.log_alpha * (
            self.target_entropy - entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

    @torch.no_grad()
    def _compute_targets(self, actor_idx, critic_idx, next_obss, dones, rewards):
        next_entropy, next_acts = self.actors[actor_idx].forward(next_obss)
        if self.min_of_all:
            if self.targetless:
                target_values = torch.minimum(
                    *[critic(next_obss, next_acts) for critic in self.critics])
            else:
                target_values = torch.minimum(
                    *[critic(next_obss, next_acts) for critic in self.critic_targets])
        else:
            idx = critic_idx + 1 if critic_idx < (self.n_critics - 1) else 0
            if self.targetless:
                # Simply use other critics as target nets
                critic1_target = self.critics[idx]
                idx = idx + 1 if idx < (self.n_critics - 1) else 0
                critic2_target = self.critics[idx]
                # What happens if we use all of them?!
            else:
                critic1_target = self.critic_targets[critic_idx]
                critic2_target = self.critic_targets[idx]

            target_values1 = critic1_target(next_obss, next_acts)
            target_values2 = critic2_target(next_obss, next_acts)
            target_values = torch.minimum(target_values1, target_values2)

        target_values += self.alpha.item() * next_entropy
        target_values[dones == 1.0] = 0.0

        return rewards + self.gamma * target_values

    def store_model(self):
        pass

    def load_model(self):
        pass


class EnsembleTd3(EnsembleDdpg):
    def __init__(self, env, update_delay=2, target_noise_std_dev=0.2,
                 noise_clip=0.5, min_of_all=False, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.update_delay = update_delay
        self.target_noise_std_dev = target_noise_std_dev
        self.noise_clip = noise_clip
        self.min_of_all = min_of_all

    def _learn(self):
        # Randomly assign actors to critics
        # The actors get trained with the same data their current critic was trained before
        actor_critic_pairs = zip(
            np.random.choice(self.n_actors, size=self.train_how_many),
            np.random.choice(self.n_critics, size=self.train_how_many))

        for actor_idx, critic_idx in actor_critic_pairs:
            # TODO: In original TD3, two critics are trained each step!
            batch = self.memory.sample_random_batch(self.batch_size)
            batch = batch_to_tensors(batch, self.device, continuous=True)
            self._train_critic(critic_idx, actor_idx, *batch)

            if self.step % self.update_delay != 0:
                continue

            self._train_actor(critic_idx, actor_idx, *batch)

            if not self.targetless:
                self._soft_target_update(self.actors[actor_idx], self.actor_targets[actor_idx], self.tau)
                self._soft_target_update(self.critics[critic_idx], self.critic_targets[critic_idx], self.tau)

    @torch.no_grad()
    def _compute_targets(self, critic_idx, actor_idx, next_obss, dones, rewards):
        if self.targetless:
            # Simply use one of the other networks as target net
            idx = actor_idx + 1 if actor_idx < (self.n_actors - 1) else 0
            actor_target = self.actors[idx]
        else:
            actor_target = self.actor_targets[actor_idx]

        # Apply target act noise
        next_acts = actor_target(next_obss)
        noise = (
            torch.randn_like(next_acts) * self.target_noise_std_dev
        ).clamp(-self.noise_clip, self.noise_clip)
        next_acts = (next_acts + noise).clamp(self.min_range, 1)

        if self.min_of_all:
            if self.targetless:
                target_values = torch.minimum(*[critic(next_obss, next_acts) for critic in self.critics])
            else:
                target_values = torch.minimum(*[critic(next_obss, next_acts) for critic in self.critic_targets])
        else:
            if self.targetless:
                idx = critic_idx + 1 if critic_idx < (self.n_critics - 1) else 0
                critic_target1 = self.critics[idx]
                idx = idx + 1 if idx < (self.n_critics - 1) else 0
                critic_target2 = self.critics[idx]
            else:
                critic_target1 = self.critic_targets[critic_idx]
                # Use the next target net as second target
                idx = critic_idx + 1 if critic_idx < (self.n_critics - 1) else 0
                critic_target2 = self.critics[idx]

            target_values1 = critic_target1(next_obss, next_acts)
            target_values2 = critic_target2(next_obss, next_acts)
            target_values = torch.minimum(target_values1, target_values2)

        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)


def act_variants(self, obs, variant=None, **actor_kwargs):
    obs = torch.tensor(obs, dtype=torch.float).to(self.device)

    # 1. Use the one that has proven it's the best (how? -> Maybe store actor idx to buffer to know which one performed best in last n steps)
    # 2. Use critics to test, which actor proposes (expected) best action (similar to Q-learning! it's argmax! funny! )
    if variant == 2:
        acts = torch.vstack([actor(obs, **actor_kwargs) for actor in self.actors])
        expanded_obs = obs.expand(self.n_actors, -1)
        exp_q = [critic(expanded_obs, acts) for critic in self.critics]
        return acts[sum(exp_q).argmax(), :].cpu().numpy()
    # 3. Use randomly (all of them should be equally good, but randomness is sometimes helpful) -> does not really make sense for testing
    elif variant == 3:
        return np.random.choice(self.actors)(obs, **actor_kwargs).cpu().numpy()
    # 4. Use average of all so that errors can cancel each other out (works for testing as well)
    elif variant == 4:
        return np.mean([actor(obs, **actor_kwargs).cpu().numpy() for actor in self.actors], axis=0)
    # 5. Use randomly weighted average of all (combination of 3. and 4., also great for exploration) -> not good for testing but maybe for normal act()
    elif variant == 5:
        # TODO: Strange idea: Can this be done with a softmax layer or similar to make it differentiable and learnable?
        weights = np.random.rand(self.n_actors)
        weights /= weights.sum()
        return np.sum([actor(obs, **actor_kwargs).cpu().numpy() * weight for actor, weight in zip(self.actors, weights)], axis=0)
    # 6. Like 5. but explicit weights for every combination of actor/action -> n x m random weights (eg actor A decides action 1, but actor B decides action 2 -> even more exploration. Funny: Similar to recombination in GAs)
    # 7. Maybe learn some meta-actor in the end (or meanwhile), whcih learns to recombine all the actor's capabilities -> is this an n-armed bandit?!
    # 8. Simply use first actor (should not perform good; use as benchmark)
    elif variant == 8:
        return self.actors[0](torch.tensor(obs, dtype=torch.float).to(self.device)).cpu().numpy()
