from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy
import csv
import random
import time
from drl.util.evaluation import Eval

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
# import pettingzoo

from drl.util.seeding import seed_env
from drl.wrappers.obs_scaler import ObsScaler
from .util import evaluation


class DrlAgent(ABC):
    def __init__(self, env, gamma=0.99, n_envs=1, path='temp/', name='Unnamed Agent',
                 autoscale_obs=True, reward_scaling=1, schedule_hps: list=None,
                 *args, **kwargs):
        # Must be a gym env or a parallel env from pettingzoo (same API as gym)
        assert isinstance(env, gym.Env)
        self.env = env
        self.n_obs = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.n_act = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            assert len(self.env.action_space.shape) == 1
            self.n_act = self.env.action_space.shape[0]
        else:
            raise NotImplementedError(
                'Only discrete and continuous action spaces possible')
        self.gamma = gamma
        self.step = 0
        self.name = name
        self.path = path
        self.seed = kwargs['seed']
        self.reward_scaling = reward_scaling
        self.test_time = 0
        self.env_time = 0
        self.train_time = 0

        if autoscale_obs:
            self.env = ObsScaler(env, -1, 1)

        self.evaluator = Eval(agent_names=[name], path=path)
        self.schedule_hps = schedule_hps
        if isinstance(self.schedule_hps, Iterable):
            self.schedule_hps = list(self.schedule_hps)
            # Sort by step in ascending order so that first entry is next entry
            self.schedule_hps.sort(key=lambda x: x[0])

        self.n_envs = n_envs
        self.envs = [copy.deepcopy(self.env) for _ in range(self.n_envs)]
        [seed_env(env, random.randint(0, 100000)) for env in self.envs]
        self.start_time = time.time()
        self.objective_data = []

    def train(self, n_steps):
        train_until = self.step + n_steps

        dones = [True] * self.n_envs
        obss = [None] * self.n_envs
        states = [None] * self.n_envs
        total_rewards = [None] * self.n_envs
        while True:
            for idx, env in enumerate(self.envs):
                if dones[idx] is True:
                    # Environment is done -> reset & collect episode data
                    if total_rewards[idx] is not None:
                        self.evaluator.step(total_rewards[idx], self.step)
                    t = time.time()
                    obss[idx], info = self.envs[idx].reset()
                    self.env_time += time.time() - t
                    try:
                        states[idx] = self.envs[idx].state()
                    except TypeError:
                        states[idx] = self.envs[idx].state
                    except AttributeError:
                        states[idx] = None

                    total_rewards[idx] = 0
                    dones[idx] = False

                self.step += 1
                while self.schedule_hps and self.schedule_hps[0][0] == self.step:
                    # Apply hyperparameter change
                    step, hp_name, hp_value = self.schedule_hps.pop(0)
                    setattr(self, hp_name, hp_value)

                act = self.act(obss[idx])

                t = time.time()
                next_obs, reward, terminal, truncated, info = self.envs[idx].step(act)
                self.env_time += time.time() - t
                objective = info['objective']
                self.objective_data.append(objective)

                try:
                    next_state = self.envs[idx].state()
                except TypeError:
                    next_state = self.envs[idx].state
                except AttributeError:
                    next_state = None

                if np.isnan(reward):
                    print('Filter out nan value')
                else:
                    t = time.time()
                    self.learn(
                        obss[idx], act, reward * self.reward_scaling, next_obs,
                        terminal, states[idx], next_state, info, env_idx=idx)
                    self.train_time += time.time() - t

                obss[idx] = next_obs
                states[idx] = next_state
                dones[idx] = terminal or truncated
                total_rewards[idx] += reward

            if self.step >= train_until:
                self.evaluator.plot_reward()
                return

    def plot_line_loss(self):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(15, 10))

        # steps = self.training_metrics['timesteps']get_objective_data
        plt.plot(self.objective_data, label='LineLoss')
        plt.title('Line Losses')
        plt.xlabel('Training Step')
        plt.ylabel('Line Loss')
        # plt.yscale('log')
        plt.legend()
        plt.show()

    def test(self, test_episodes=None, test_steps=30):
        """ Evaluate the trained agent on a specified number of test
        steps/episodes. If both episodes and steps are specified, the first
        condition that is met ends the evaluation. Use `None` to set no limit,
        but one of them must always be defined to prevent endless loop.
        Testing is done on the original (not distributed) environment to ensure
        no interference with training environments. """
        assert (test_episodes is not None or test_steps is not None)
        self.store_model()

        print('Start testing: ')
        if test_episodes is None:
            test_episodes = np.inf
        if test_steps is None:
            test_steps = np.inf
        count_episodes = 0
        count_steps = 0
        t = time.time()

        total_rewards = []
        # Some reference reward, e.g. human-performance or optimal actions
        baseline_objs = []
        objs = []

        valids = []
        infos = []

        while True:
            count_episodes += 1
            obs, info = self.env.reset(options={'test': True})

            done = False
            if isinstance(self, MarlAgent):
                total_rewards.append({a_id: 0 for a_id in self.a_ids})
            else:
                total_rewards.append(0)
            while not done:
                count_steps += 1
                # Use `test_act()` for testing instead of the noisy `act()`
                act = self.test_act(obs)

                obs, reward, terminal, truncated, info = self.env.step(act)
                reward = np.sum(reward)



                if isinstance(self, MarlAgent):
                    total_rewards[-1] = {
                        a_id: r + reward[a_id]
                        for a_id, r in total_rewards[-1].items()}
                else:
                    total_rewards[-1] += reward

                done = terminal or truncated    
                if done and hasattr(self.env.unwrapped, 'run_optimal_power_flow'):
                    infos.append(info)
                    # TODO: Move this to special OPF-agent class
                    success = self.env.unwrapped.run_optimal_power_flow()
                    objs.append(sum(self.env.unwrapped.calculate_objective()))
                    if success:
                        baseline_objs.append(self.env.unwrapped.get_optimal_objective())
                    else:
                        baseline_objs.append(np.nan)

                    if (not np.isnan(baseline_objs[-1])
                            and baseline_objs[-1] < objs[-1]):
                        print(f'Baseline worse than DRL agent!',
                              baseline_objs[-1], objs[-1])

            if count_episodes >= test_episodes or count_steps >= test_steps:
                break

        print('Average episode return:', np.mean(total_rewards))

        # TODO: Rather call it 'optimal_state' or something?!
        if hasattr(self.env.unwrapped, 'run_optimal_power_flow'):
            # Compute rmse and mape
            total_rewards = np.array(total_rewards)
            objs = np.array(objs)
            print('rewards: ', objs)
            baseline_objs = np.array(baseline_objs)
            print('Baseline rewards: ', baseline_objs)
            print('base higher: ', baseline_objs > objs)
            # Sometimes, there is no baseline -> Filter nan entries
            wrong_entries = np.isnan(baseline_objs)
            solved_baseline_objs = baseline_objs[~wrong_entries]

            regret = solved_baseline_objs - objs[~wrong_entries]
            rmse = np.sqrt(np.square(regret).mean())
            mape = np.mean(np.abs(regret / solved_baseline_objs)) * 100
            mpe = np.mean(regret / solved_baseline_objs) * 100
            # Normal MPE does not make sense to compare -> add np.abs()
            mpe_corr = np.mean(regret / np.abs(solved_baseline_objs)) * 100

            # Consider constraint satisfaction
            valids = [info_dict['valids'] for info_dict in infos]
            total_valids = np.array([np.all(v) for v in valids])
            valid_and_opf_correct = np.logical_and(total_valids, ~wrong_entries)
            print('valid solutions: ', total_valids[~wrong_entries])

            valid_rewards = objs[valid_and_opf_correct]
            valid_base = baseline_objs[valid_and_opf_correct]

            valid_regret = valid_base - valid_rewards
            valid_rmse = np.sqrt(np.square(valid_regret).mean())
            valid_mape = np.mean(np.abs(valid_regret / valid_base)) * 100
            valid_mpe = np.mean(valid_regret / valid_base) * 100
            valid_mpe_corr = np.mean(
                valid_regret / np.abs(valid_base)) * 100

            valid_share = np.mean(total_valids)
            valid_share_possible = sum(total_valids) / sum(~wrong_entries)
            # Compute valid share for every single constraint separately
            separate_valid_shares = np.array([np.mean(v) for v in zip(*valids)])

            # How big are the violations of the constraints? (if OPF succeeded)
            # percentage_violations = np.array([info_dict['percentage_violations'] for info_dict in infos])
            # mean_percentage_violations = np.mean(percentage_violations[~wrong_entries], axis=0)
            # invalid_mean_percentage_violations = mean_percentage_violations / (1-separate_valid_shares)
            mean_violations = np.array([info_dict['violations'] for info_dict in infos])
            mean_violations = np.mean(mean_violations[~wrong_entries], axis=0)
            invalid_mean_violations = mean_violations / (1-separate_valid_shares)

            # What penalties did the agent receive?
            penalties = np.array([info_dict['unscaled_penalties'] for info_dict in infos])
            mean_penalties = np.mean(penalties[~wrong_entries], axis=0)

            # DRL has valid solution and better performance than OPF?
            outperform = np.logical_and(
                valid_and_opf_correct, objs > baseline_objs)
            # DRL has valid solution that OPF did not find?
            outperform_constraints = np.logical_and(total_valids, wrong_entries)

            print('_________________________________')
            print('Test completed')
            print('Deviation from baseline:')
            print('mean regret: ', np.mean(regret))
            print('rmse: ', rmse)
            print('mape: ', mape, '%')
            print('')

        self.test_time += time.time() - t
        total_time = time.time() - self.start_time
        # Write results to csv files
        results = {
            'step': self.step,
            'total_time': total_time,
            'train_time': self.train_time,
            'test_time': self.test_time,
            'env_time': self.env_time,
            'env_and_train_time': self.env_time + self.train_time,
            'n_steps': count_steps,
            'n_episodes': count_episodes,
            'average_return': np.mean(total_rewards)}

        if hasattr(self.env.unwrapped, 'run_optimal_power_flow'):
            results['failed_share'] = np.mean(wrong_entries)
            results['mean_obj'] = np.mean(objs)
            results['mape'] = mape
            results['mpe'] = mpe
            results['mpe_corr'] = mpe_corr
            results['rmse'] = rmse
            results['regret'] = np.mean(regret)
            results['valid_mape'] = valid_mape
            results['valid_mpe'] = valid_mpe
            results['valid_mpe_corr'] = valid_mpe_corr
            results['valid_rmse'] = valid_rmse
            results['valid_regret'] = np.mean(valid_regret)
            results['valid_share'] = valid_share
            results['valid_share_possible'] = valid_share_possible
            results['sep_valid_share'] = separate_valid_shares
            results['outperform_share'] = np.mean(outperform)
            results['outperform_constraints'] = np.mean(outperform_constraints)
            # results['mean_percentage_violations'] = mean_percentage_violations
            # results['invalid_mean_percentage_violations'] = invalid_mean_percentage_violations
            results['mean_violations'] = mean_violations
            results['invalid_mean_violations'] = invalid_mean_violations
            results['mean_penalties'] = mean_penalties

        self.test_results_to_csv(results)

        return results

    def test_results_to_csv(self, results: dict):
        # If iterable, convert to multiple entries for better usage later
        results_ = {}
        for key, value in results.items():
            if hasattr(value, '__len__'):
                d = {f'{key}_{idx}': v for idx, v in enumerate(value)}
                results_.update(d)
            else:
                results_[key] = value

        with open(self.path + 'test_returns.csv', 'a') as f:
            w = csv.DictWriter(f, results_.keys())
            if f.tell() == 0:
                w.writeheader()
            w.writerow(results_)

    @ abstractmethod
    def act(self, obs):
        pass

    @ abstractmethod
    def test_act(self, obs):
        """ Act without exploration for testing, e.g. remove noise. """
        pass

    @ abstractmethod
    def learn(self, obs, act, reward, next_obs, done):
        # TODO: Maybe rename to agent.step()
        pass

    def store_model(self):
        pass

    def load_model(self):
        pass

class MarlAgent(DrlAgent):
    def __init__(self, env, gamma, n_envs=1, path='temp/', name='Unnamed Agent',
                 autoscale_obs=True, reward_scaling=1, *args, **kwargs):
        # TODO: Use super init?!
        if autoscale_obs:
            env = PettingZooObsScaler(env)
        self.env = env
        self.a_ids = env.possible_agents
        self.action_spaces = env.action_spaces
        self.observation_spaces = env.observation_spaces
        self.state_space = env.state_space

        self.n_obs = sum(len(s.low) for s in env.observation_spaces.values())
        self.n_act = sum(len(a.low) for a in env.action_spaces.values())
        self.n_states = len(env.state_space.low)
        self.n_agents = len(self.a_ids)

        self.gamma = gamma
        self.reward_scaling = reward_scaling
        self.n_envs = n_envs
        self.step = 0
        self.name = name
        self.path = path
        self.evaluator = evaluation.Eval(agent_names=self.a_ids, path=path)

        self.reset()

        self.test_env = copy.deepcopy(self.env)

    def run_sequential(self, n_steps, test_interval=99999999, test_steps=10):
        """ When the pettingzoo environment is sequential (agents act one after
        the other), we need a new run method. """
        # TODO: This whole method is still WIP
        next_test = test_interval
        dones = {a_id: False for a_id in self.a_ids}
        obss = {}
        next_obss = {}
        acts = {}
        rewards = {}
        returns = {a_id: 0 for a_id in self.a_ids}
        infos = {}

        start_step = self.step
        self.start_time = time.time()
        self.n_train_steps = n_steps

        self.test_env.reset()

        for a_id in self.test_env.agent_iter():
            next_obss[a_id], rewards[a_id], dones[a_id], infos[a_id] = self.test_env.last()
            returns[a_id] += rewards[a_id]

            if a_id == self.a_ids[-1]:
                # Agents went full circle
                self.step += 1
                next_state = self.test_env.state()

            # TODO: Is this really correct? All agents get trained with state_1 this way. However, state changes with every action. So maybe we should actually store the state per agent (then the other actions are actually not required anymore...)

            if a_id in obss.keys():
                self.single_learn(a_id, obss[a_id], acts[a_id],
                                  rewards[a_id] * self.reward_scaling,
                                  next_obss[a_id], dones[a_id],
                                  state, next_state)

            obss[a_id] = next_obss[a_id]

            if a_id == self.a_ids[-1]:
                # The agents used the previous state for learning -> overwrite
                state = next_state

            if sum(dones.values()) == len(dones):
                obss = {}
                self.test_env.reset()
                dones = {a: False for a in self.a_ids}
                self.evaluator.step(returns, self.step)
                returns = {a: 0 for a in self.a_ids}
                continue

            assert self.test_env.agent_selection == a_id

            acts[a_id] = self.single_act(a_id, next_obss[a_id], noisy=True)
            self.test_env.step(acts[a_id])

            if self.step >= next_test and a_id == self.a_ids[-1]:
                # TODO: Code repetition
                self.test(test_steps=test_steps)
                next_test = self.step + test_interval

            if self.step - start_step >= self.n_train_steps:
                # TODO: Store results of testing instead of training
                self.evaluator.plot_reward()
                return self.evaluator

    def single_learn(self, a_id, obs, act, reward, next_obs, done, state=None,
                     next_state=None, env_idx=0):
        """ Collect all data to then store them at once. """
        self.obss[a_id] = obs
        self.acts[a_id] = act
        self.rewards[a_id] = reward
        self.next_obss[a_id] = next_obs
        self.dones[a_id] = done
        if len(self.obss) == len(self.a_ids):
            self.learn(self.obss, self.acts, self.rewards, self.next_obss,
                       self.dones, state, next_state)

            self.reset()

    def reset(self):
        self.obss = {}
        self.acts = {}
        self.rewards = {}
        self.next_obss = {}
        self.dones = {}

    def test_sequential(self, test_episodes=None, test_steps=50):
        # TODO: Currently mainly copy-pasted from run_sequential!
        print('-------------------------------')
        print('Start testing')
        print('-------------------------------')
        dones = {a_id: False for a_id in self.a_ids}
        obss = {}
        next_obss = {}
        acts = {}
        rewards = {}
        returns = [{a_id: 0 for a_id in self.a_ids}]
        infos = {}
        count_steps = 0
        count_episodes = 0

        # Local evaluator only for this test
        # evaluator = evaluation.Eval(agent_names=self.a_ids, path=self.path)

        # start_step = self.step
        start_time = time.time()

        try:
            self.env.reset(test=True)
        except TypeError:
            self.env.reset()

        for a_id in self.env.agent_iter():
            if a_id == self.a_ids[-1]:
                # Agents went full circle
                count_steps += 1

            next_obss[a_id], rewards[a_id], dones[a_id], infos[a_id] = self.env.last()
            returns[-1][a_id] += rewards[a_id]
            obss[a_id] = next_obss[a_id]

            if sum(dones.values()) == len(dones):
                obss = {}
                try:
                    self.env.reset(test=True)
                except TypeError:
                    self.env.reset()

                dones = {a_id: False for a_id in self.a_ids}
                returns.append({a_id: 0 for a_id in self.a_ids})
                count_episodes += 1

                continue

            assert self.env.agent_selection == a_id

            acts[a_id] = self.single_act(a_id, next_obss[a_id], noisy=False)

            self.env.step(acts[a_id])

            if count_steps >= test_steps:
                # TODO: Store results of testing
                print('-------------------------------')
                print('Done testing')
                print('-------------------------------')

                results = {
                    'step': self.step,
                    'time': time.time() - self.start_time,
                    'n_steps': count_steps,
                    'n_episodes': count_episodes}
                average_returns = {
                    'average_return_' + a_id:
                    sum(r[a_id] for r in returns) / len(returns)
                    for a_id in self.a_ids}
                results.update(average_returns)

                print('Average test returns:')
                for a_id in average_returns.keys():
                    print(a_id, ': ', average_returns[a_id])

                self.test_results_to_csv(results)
                return

    @ abstractmethod
    def single_act(self, agent_id: str, obs, noisy: bool):
        """ Only the single agent "agent_id" acts. """
        pass


class TargetNetMixin():
    """ Mixin for agents that have a target net, which needs to be updated. """
    # TODO: This way, these are only functions and a mixin is not really required...

    def _hard_target_update(self, net, target_net):
        """ Set parameters of target net equal to q net. """
        # TODO: Add more efficient implementation here
        self._soft_target_update(net, target_net, tau=1)

    def _soft_target_update(self, net, target_net, tau=0.001):
        params = dict(net.state_dict())
        target_params = dict(target_net.state_dict())

        for name in params:
            params[name] = (
                tau * params[name].clone()
                + (1 - tau) * target_params[name].clone()
            )

        target_net.load_state_dict(params)


# def space_scaler(action, from_space, to_space):
#     return to_space.low + (to_space.high - to_space.low) * (
#         (action - from_space.low) / (from_space.high - from_space.low))


# class ScalerObs():
#     # TODO: Again use wrappers instead
#     def __init__(self, low, high):
#         self.low = low
#         self.high = high

#         if np.inf in self.high or np.inf in self.low:
#             print('Warning: inf in observation space -> scaling impossible')

#     def __call__(self, obs):
#         # TODO: This can be done better!
#         if np.inf in self.high or np.inf in self.low:
#             return obs

#         # (obs - self.low) / (self.high - self.low)
#         scaled_obs = (2 * (obs - self.low) / (self.high - self.low)) - 1

#         # Minor offset to prevent rounding errors to do harm
#         eps = 1e-2
#         if ((scaled_obs + eps) < -1).any() or ((scaled_obs - eps) > 1).any():
#             worst_entry1 = min(
#                 scaled_obs[(scaled_obs + eps) < -1], default='empty')
#             worst_entry2 = max(
#                 scaled_obs[(scaled_obs - eps) > 1], default='empty')
#             print(
#                 f'Warning: observation out of observation space! (worst entries {worst_entry1} and {worst_entry2})')

#         return scaled_obs


# class PettingZooActScaler(pettingzoo.utils.wrappers.BaseParallelWrapper):
#     def __init__(self, env, nn_action_spaces: dict):
#         super().__init__(env)
#         self.nn_action_spaces = nn_action_spaces
#         assert all(isinstance(self.action_space(a_id), Box) for a_id in getattr(
#             self, 'possible_agents', [])), "should only use Scaling for Box spaces"
#
#     def step(self, action: dict):
#         scaled_act = {}
#         for a_id in self.possible_agents:
#             from_space = self.nn_action_spaces[a_id]
#             to_space = self.action_space(a_id)
#             new_act = to_space.low + (to_space.high - to_space.low) * (
#                 (action[a_id] - from_space.low) / (from_space.high - from_space.low))
#
#             scaled_act[a_id] = np.clip(new_act, to_space.low, to_space.high)
#
#         return super().step(scaled_act)
#
#     def __str__(self):
#         return str(self.env)


# class PettingZooActScalerSequential(pettingzoo.utils.wrappers.BaseWrapper):
#     def __init__(self, env, nn_action_spaces: dict):
#         super().__init__(env)
#         self.nn_action_spaces = nn_action_spaces
#         assert all(isinstance(self.action_space(a_id), Box) for a_id in getattr(
#             self, 'possible_agents', [])), "should only use Scaling for Box spaces"
#
#     def step(self, action):
#         from_space = self.nn_action_spaces[self.env.agent_selection]
#         to_space = self.action_space(self.env.agent_selection)
#         scaled_act = to_space.low + (to_space.high - to_space.low) * (
#             (action - from_space.low) / (from_space.high - from_space.low))
#
#         scaled_act = np.clip(scaled_act, to_space.low, to_space.high)
#
#         return super().step(scaled_act)
#
#     def __str__(self):
#         return str(self.env)
#
#     def __getattr__(self, name):
#         """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
#         # TODO: With newest pettingzoo version, this is not necessary anymore
#         if name.startswith("_"):
#             raise AttributeError(
#                 f"accessing private attribute '{name}' is prohibited")
#         return getattr(self.env, name)


# class PettingZooObsScaler(pettingzoo.utils.wrappers.BaseWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         assert all(isinstance(self.observation_space(a_id), Box) for a_id in getattr(
#             self, 'possible_agents', [])), "should only use Scaling for Box spaces"
#
#     def observe(self, agent_id):
#         """ Re-scale to range [0, 1] """
#         obs_space = self.env.observation_spaces[agent_id]
#         obs = self.env.observations[agent_id]
#         try:
#             assert (obs >= obs_space.low - 0.000001).all()
#             assert (obs <= obs_space.high + 0.000001).all()
#         except AssertionError:
#             # import pdb
#             # pdb.set_trace()
#             pass
#         return (obs - obs_space.low) / (obs_space.high - obs_space.low)
#
#     def __str__(self):
#         return str(self.env)
#
#     def __getattr__(self, name):
#         """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
#         # TODO: With newest pettingzoo version, this is not necessary anymore
#         if name.startswith("_"):
#             raise AttributeError(
#                 f"accessing private attribute '{name}' is prohibited")
#         return getattr(self.env, name)
