import os
from collections.abc import Callable
import importlib
import copy
import logging
import inspect
import warnings
import numpy as np
import pandas as pd
import pandapower as pp
from scipy import stats
from typing import Tuple
import gymnasium as gym
from drl.util import seeding
from drl.unpublished.ddpg1step import Ddpg1Step
import matplotlib.pyplot as plt
from reward import RewardFunction
import constraints
import objective
import seaborn as sns
from drl.ddpg import Ddpg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from drl.td3 import Td3
from drl.unpublished.td31step import Td31Step

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")


def compute_line_loss(Bus, Branch, V0=12.66, tol=1e-6, max_iter=300):
    Vbase = 12.66
    Sbase = 10
    base_power_kW = Sbase * 1000
    Zbase = Vbase ** 2 / Sbase

    def prepare_data(Bus, Branch):

        bus_data = {}
        for row in Bus:
            bus = int(row[0])
            P = row[1] / (1000 * Sbase)  # kW -> pu
            Q = row[2] / (1000 * Sbase)  # kVar -> pu
            bus_data[bus] = {'P': P, 'Q': Q}

        branch_data = []
        for row in Branch:
            from_bus = int(row[1])
            to_bus = int(row[2])
            R = row[3] / Zbase
            X = row[4] / Zbase
            branch_data.append({
                'from': from_bus,
                'to': to_bus,
                'R': R,
                'X': X,
                'Z': complex(R, X)
            })
        return bus_data, branch_data

    def backward_forward_sweep(bus_data, branch_data, max_iter=100, tol=1e-6, initial_voltage_pu=1.0):
        nodes = sorted(bus_data.keys())
        V = {bus: complex(initial_voltage_pu, 0) for bus in nodes}
        I = {i: complex(0, 0) for i in range(len(branch_data))}

        for it in range(max_iter):
            V_prev = V.copy()

            for idx in reversed(range(len(branch_data))):
                line = branch_data[idx]
                to_bus = line['to']

                S = complex(bus_data[to_bus]['P'], bus_data[to_bus]['Q'])
                I_load = np.conj(S / V[to_bus]) if abs(V[to_bus]) > 1e-10 else 0.0

                child_lines = [i for i, l in enumerate(branch_data) if l['from'] == to_bus]
                I[idx] = I_load + sum(I[child_idx] for child_idx in child_lines)

            for idx in range(len(branch_data)):
                line = branch_data[idx]
                from_bus = line['from']
                to_bus = line['to']
                V[to_bus] = V[from_bus] - I[idx] * line['Z']

            max_diff = max(abs(V[bus] - V_prev[bus]) for bus in nodes)
            if max_diff < tol:
                break

        return V, I

    busnum = Bus.shape[0]

    bus_data, branch_data = prepare_data(Bus, Branch)

    initial_voltage_pu = V0 / Vbase

    V_pu, I_pu = backward_forward_sweep(
        bus_data, branch_data,
        max_iter=max_iter, tol=tol, initial_voltage_pu=initial_voltage_pu
    )

    Ploss = np.zeros((busnum, busnum))
    Qloss = np.zeros((busnum, busnum))
    for idx, branch in enumerate(branch_data):
        from_bus = branch['from'] - 1
        to_bus = branch['to'] - 1
        I2 = abs(I_pu[idx]) ** 2
        R_pu = branch['R']
        X_pu = branch['X']
        Ploss_pu = I2 * R_pu
        Qloss_pu = I2 * X_pu
        Ploss[from_bus, to_bus] = Ploss_pu * base_power_kW
        Qloss[from_bus, to_bus] = Qloss_pu * base_power_kW

    Vbus = np.zeros((busnum, 1))
    for bus in range(1, busnum + 1):
        Vbus[bus - 1, 0] = abs(V_pu[bus]) * Vbase

    return Ploss, Qloss, Vbus


class PowerFlowNotAvailable(Exception):
    pass


class OpfEnv(gym.Env):
    def __init__(self,
                 net: pp.pandapowerNet,
                 action_keys: tuple[tuple[str, str, np.ndarray], ...],
                 observation_keys: tuple[tuple[str, str, np.ndarray], ...],

                 state_keys: tuple[tuple[str, str, np.ndarray], ...] = None,

                 profiles: dict[str, pd.DataFrame] = None,
                 evaluate_on: str = 'validation',
                 steps_per_episode: int = 1,

                 bus_wise_obs: bool = False,
                 reward_function: str | RewardFunction = 'summation',
                 reward_function_params: dict = None,
                 diff_objective: bool = False,

                 add_res_obs: bool = False,
                 add_time_obs: bool = False,
                 add_act_obs: bool = False,
                 add_mean_obs: bool = False,
                 train_data: str = 'simbench',
                 test_data: str = 'simbench',
                 sampling_params: dict = None,
                 constraint_params: dict = {},
                 custom_constraints: list = None,
                 autoscale_actions: bool = True,
                 diff_action_step_size: float = None,

                 clipped_action_penalty: float = 0.0,

                 initial_action: str = 'center',
                 objective_function: Callable[[pp.pandapowerNet], np.ndarray | float] = None,
                 power_flow_solver: Callable[[pp.pandapowerNet], None] = None,
                 optimal_power_flow_solver: Callable[[pp.pandapowerNet], None] = None,
                 seed: int = None,
                 **kwargs):

        self.net = net
        self.obs_keys = observation_keys
        self.state_keys = state_keys or copy.copy(observation_keys)
        self.act_keys = action_keys
        self.profiles = profiles
        self.objective_data = []

        if not profiles:
            pass

            # Define the power flow and OPF solvers (default to pandapower)
        self._run_power_flow = power_flow_solver or self.forward_backward_power_flow
        if optimal_power_flow_solver is None:
            self._run_optimal_power_flow = self.default_optimal_power_flow
        elif optimal_power_flow_solver is False:
            # No optimal power flow solver available
            self._run_optimal_power_flow = raise_opf_not_converged
        else:
            self._run_optimal_power_flow = optimal_power_flow_solver

        # Define objective function
        if objective_function is None:
            self.objective_function = objective.get_pandapower_costs
        else:
            assert_only_net_in_signature(objective_function)
            self.objective_function = objective_function

        self.evaluate_on = evaluate_on
        self.train_data = train_data
        self.test_data = test_data
        self.sampling_params = sampling_params or {}

        # Define the observation space
        self.add_act_obs = add_act_obs
        if add_act_obs:
            # The agent can observe its previous actions
            self.obs_keys.extend(self.act_keys)

        self.add_time_obs = add_time_obs
        # Add observations that require previous pf calculation
        if add_res_obs is True:
            # Default: Add all results that are usually available
            add_res_obs = ('voltage_magnitude', 'voltage_angle',
                           'line_loading', 'trafo_loading', 'ext_grid_power')
        if add_res_obs:
            # Tricky: Only use buses with actual units connected. Otherwise, too many auxiliary buses are included.
            bus_idxs = set(self.net.load.bus) | set(self.net.sgen.bus) | set(self.net.gen.bus) | set(
                self.net.storage.bus)
            add_obs = []
            if 'voltage_magnitude' in add_res_obs:
                add_obs.append(('res_bus', 'vm_pu', np.sort(list(bus_idxs))))
            if 'voltage_angle' in add_res_obs:
                add_obs.append(('res_bus', 'va_degree', np.sort(list(bus_idxs))))
            if 'line_loading' in add_res_obs:
                add_obs.append(('res_line', 'loading_percent', self.net.line.index))
            if 'trafo_loading' in add_res_obs:
                add_obs.append(('res_trafo', 'loading_percent', self.net.trafo.index))
            if 'ext_grid_power' in add_res_obs:
                add_obs.append(('res_ext_grid', 'p_mw', self.net.ext_grid.index))
                add_obs.append(('res_ext_grid', 'q_mvar', self.net.ext_grid.index))
            self.obs_keys.extend(add_obs)

        self.add_mean_obs = add_mean_obs

        # Define observation, state, and action spaces
        self.bus_wise_obs = bus_wise_obs
        self.observation_space = get_obs_and_state_space(
            self.net, self.obs_keys, add_time_obs, add_mean_obs,
            seed=seed, bus_wise_obs=bus_wise_obs)
        self.state_space = get_obs_and_state_space(
            self.net, self.state_keys, seed=seed)
        n_actions = sum([len(idxs) for _, _, idxs in self.act_keys])
        self.action_space = gym.spaces.Box(0, 1, shape=(n_actions,), seed=seed)

        # Action space details
        self.autoscale_actions = autoscale_actions
        self.diff_action_step_size = diff_action_step_size
        self.clipped_action_penalty = clipped_action_penalty
        self.initial_action = initial_action

        self.steps_per_episode = steps_per_episode

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO: Not implemented yet. Required only for partially observable envs

        # Is a powerflow calculation required to get new observations in reset?
        self.pf_for_obs = False
        for unit_type, _, _ in self.obs_keys:
            if 'res_' in unit_type:
                self.pf_for_obs = True
                break

        self.diff_objective = diff_objective
        if diff_objective:
            # An initial power flow is required to compute the initial objective
            self.pf_for_obs = True

        # Define data distribution for training and testing
        self.test_steps, self.validation_steps, self.train_steps = define_test_train_split(**kwargs)

        # Constraints
        if custom_constraints is None:
            self.constraints = constraints.create_default_constraints(
                self.net, constraint_params)
        else:
            self.constraints = custom_constraints

        # Define reward function
        reward_function_params = reward_function_params or {}
        if isinstance(reward_function, str):
            # Load by string (e.g. 'Summation' or 'summation')
            reward_class = load_class_from_module(
                reward_function, 'reward')
            self.reward_function = reward_class(
                env=self, **reward_function_params)
        elif isinstance(reward_function, RewardFunction):
            # User-defined reward function
            self.reward_function = reward_function

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.info = {}
        self.current_simbench_step = None
        self.step_in_episode = 0

        if not options:
            options = {}

        self.test = options.get('test', False)
        step = options.get('step', None)
        self.apply_action = options.get('new_action', True)

        self._sampling(step, self.test, self.apply_action)

        if self.initial_action == 'random':
            # Use random actions as starting point so that agent learns to handle that
            act = self.action_space.sample()
        else:
            # Reset all actions to default values
            act = (self.action_space.low + self.action_space.high) / 2
        self._apply_actions(act)

        if self.pf_for_obs is True:
            self.run_power_flow()
            if not self.power_flow_available:
                logging.warning(
                    'Failed powerflow calculcation in reset. Try again!')
                return self.reset()

            self.initial_obj = self.calculate_objective(diff_objective=False)

        obs = self._get_obs(self.obs_keys, self.add_time_obs, self.add_mean_obs)

        return obs, copy.deepcopy(self.info)

    def _sampling(self,
                  step=None,
                  test=False,
                  sample_new=True,
                  *args, **kwargs) -> None:

        self.set_power_flow_unavailable()

        data_distr = self.test_data if test is True else self.train_data

        kwargs.update(self.sampling_params)

        if data_distr == 'noisy_simbench' or 'noise_factor' in kwargs.keys():
            if sample_new:
                self._set_simbench_state(step, test, *args, **kwargs)

        elif data_distr == 'simbench':
            if sample_new:
                self._set_simbench_state(
                    step, test, noise_factor=0.0, *args, **kwargs)

        elif data_distr == 'full_uniform':
            self._sample_uniform(sample_new=sample_new)

        elif data_distr == 'normal_around_mean':
            self._sample_normal(sample_new=sample_new, **kwargs)

        elif data_distr == 'mixed':

            r = self.np_random.random()
            data_probs = kwargs.get('data_probabilities', (0.5, 0.75, 1.0))
            if r < data_probs[0]:
                self._set_simbench_state(step, test, *args, **kwargs)
            elif r < data_probs[1]:
                self._sample_uniform(sample_new=sample_new)
            else:
                self._sample_normal(sample_new=sample_new, **kwargs)

    def _sample_uniform(self, sample_keys=None, sample_new=True) -> None:
        assert sample_new, 'Currently only implemented for sample_new=True'
        if not sample_keys:
            sample_keys = self.state_keys
        for unit_type, column, idxs in sample_keys:

            if 'res_' not in unit_type:
                self._sample_from_range(unit_type, column, idxs)

    def _sample_from_range(self,
                           unit_type,
                           column,
                           idxs) -> None:

        df = self.net[unit_type]

        try:
            low = df[f'min_min_{column}'].loc[idxs]
        except KeyError:
            low = df[f'min_{column}'].loc[idxs]
        try:
            high = df[f'max_max_{column}'].loc[idxs]
        except KeyError:
            high = df[f'max_{column}'].loc[idxs]

        r = self.np_random.uniform(low, high, size=(len(idxs),))

        try:
            # Constraints are scaled, which is why we need to divide by scaling
            self.net[unit_type].loc[idxs, column] = r / df.scaling[idxs]
        except AttributeError:
            # If scaling factor is not defined, assume scaling=1
            self.net[unit_type].loc[idxs, column] = r

    def _sample_normal(self, relative_std=None, truncated=False, sample_new=True, **kwargs):
        """ Sample data around mean values from simbench data. """
        assert sample_new, 'Currently only implemented for sample_new=True'
        for unit_type, column, idxs in self.state_keys:
            if 'res_' in unit_type or 'poly_cost' in unit_type:
                continue

            df = self.net[unit_type].loc[idxs]

            try:
                mean = df[f'mean_{column}'].to_numpy()
            except KeyError:

                mean = np.zeros(len(idxs))

            try:
                max_values = (df[f'max_max_{column}'] / df.scaling).to_numpy()
                min_values = (df[f'min_min_{column}'] / df.scaling).to_numpy()
            except KeyError:

                max_values = np.ones(len(idxs))
                min_values = np.zeros(len(idxs))

            diff = max_values - min_values

            if relative_std:
                std = relative_std * diff
            else:
                try:
                    std = df[f'std_dev_{column}'].to_numpy()
                except KeyError:

                    std = diff * 0.1

            if truncated:
                random_values = stats.truncnorm.rvs(
                    min_values, max_values, mean, std * diff, len(mean)
                )
            else:
                random_values = self.np_random.normal(mean, std * diff, len(mean))
                random_values = np.clip(random_values, min_values, max_values)

            self.net[unit_type].loc[idxs, column] = random_values

    def _set_simbench_state(self,
                            step: int = None,
                            test: bool = False,
                            sample_new: bool = True,
                            apply_action=None,
                            noise_factor: float = 0.0,
                            noise_distribution: str = 'uniform',
                            interpolate_steps: bool = False,
                            **kwargs) -> None:

        # --- guard profiles ---
        if not self.profiles or ('load', 'q_mvar') not in self.profiles:
            return

        total_n_steps = len(self.profiles[('load', 'q_mvar')])
        # 选 step
        if step is None:
            if test and self.evaluate_on == 'test':
                step = self.np_random.choice(self.test_steps)
            elif test and self.evaluate_on == 'validation':
                step = self.np_random.choice(self.validation_steps)
            else:
                step = self.np_random.choice(self.train_steps)
        else:
            assert 0 <= step < total_n_steps, f"step {step} 超出范围"

        self.current_simbench_step = step

        for (unit_type, actuator), df in self.profiles.items():
            if df.shape[1] == 0:
                continue
            data = df.loc[step, self.net[unit_type].index]

            if interpolate_steps and step < total_n_steps - 1:
                next_data = df.loc[step + 1, self.net[unit_type].index]
                α = self.np_random.random()
                data = data * α + next_data * (1 - α)

            self.net[unit_type].loc[self.net[unit_type].index, actuator] = data

        if apply_action is not None:
            pass

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert not np.isnan(action).any()
        self.info = {}
        self.step_in_episode += 1

        if self.apply_action:
            correction = self._apply_actions(action, self.diff_action_step_size)
            self.run_power_flow()

            if not self.power_flow_available:
                logging.critical(
                    f'\nPowerflow not converged and reason unknown! Run diagnostic tool to at least find out what went wrong: {pp.diagnostic(self.net)}')
                self.info['valids'] = np.array([False] * 5)
                self.info['violations'] = np.array([1] * 5)
                self.info['unscaled_penalties'] = np.array([1] * 5)
                self.info['penalty'] = 5
                self.info['objective'] = 0.0
                return np.array([np.nan]), np.nan, True, False, copy.deepcopy(self.info)

        reward, objective = self.calculate_reward()
        self.info['objective'] = objective

        if self.clipped_action_penalty and self.apply_action:
            reward -= correction * self.clipped_action_penalty

        if self.steps_per_episode == 1:
            terminated = True
            truncated = False
        elif self.step_in_episode >= self.steps_per_episode:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        obs = self._get_obs(self.obs_keys, self.add_time_obs, self.add_mean_obs)
        assert not np.isnan(obs).any()

        return obs, reward, terminated, truncated, copy.deepcopy(self.info)

    def _apply_actions(self, action, diff_action_step_size=None) -> float:
        self.set_power_flow_unavailable()
        # Clip invalid actions
        action = np.clip(action, self.action_space.low, self.action_space.high)

        counter = 0
        for unit_type, actuator, idxs in self.act_keys:
            if len(idxs) == 0:
                continue

            df = self.net[unit_type]
            partial_act = action[counter:counter + len(idxs)]

            # Handle missing min/max columns
            try:
                if self.autoscale_actions:
                    min_action = df.get(f'min_{actuator}', df.get(f'min_min_{actuator}', 0)).loc[idxs]
                    max_action = df.get(f'max_{actuator}', df.get(f'max_max_{actuator}', 1)).loc[idxs]
                else:
                    min_action = df.get(f'min_min_{actuator}', 0).loc[idxs]
                    max_action = df.get(f'max_max_{actuator}', 1).loc[idxs]
            except KeyError:
                # If even the fallback fails, use default values
                min_action = pd.Series(0, index=idxs)
                max_action = pd.Series(1, index=idxs)

            delta_action = (max_action - min_action).values

            # Always use continuous action space [0, 1]
            if diff_action_step_size:
                # Agent sets incremental setpoints instead of absolute ones.
                previous_setpoints = self.net[unit_type][actuator].loc[idxs].values
                if 'scaling' in df.columns:
                    previous_setpoints *= df.scaling.loc[idxs]

                # Make sure decreasing the setpoint is possible as well
                partial_act = partial_act * 2 - 1
                setpoints = partial_act * diff_action_step_size * delta_action + previous_setpoints
            else:
                # Agent sets absolute setpoints in range [min, max]
                setpoints = partial_act * delta_action + min_action

            # Autocorrect impossible setpoints
            if not self.autoscale_actions or diff_action_step_size:
                if f'max_{actuator}' in df.columns:
                    mask = setpoints > df[f'max_{actuator}'].loc[idxs]
                    setpoints[mask] = df[f'max_{actuator}'].loc[idxs][mask]
                if f'min_{actuator}' in df.columns:
                    mask = setpoints < df[f'min_{actuator}'].loc[idxs]
                    setpoints[mask] = df[f'min_{actuator}'].loc[idxs][mask]

            if 'scaling' in df.columns:
                # Scaling column sometimes not existing
                setpoints /= df.scaling.loc[idxs]

            if actuator == 'closed' or actuator == 'in_service':
                # Special case: Only binary actions
                setpoints = np.round(setpoints).astype(bool)
            elif actuator == 'tap_pos' or actuator == 'step':
                # Special case: Only discrete actions
                setpoints = np.round(setpoints)

            self.net[unit_type].loc[idxs, actuator] = setpoints
            counter += len(idxs)

        # Did the action need to be corrected to be in bounds?
        mean_correction = np.mean(abs(
            self.get_current_actions(from_results_table=False) - action))
        return mean_correction

    def calculate_objective(self, net=None, diff_objective=False) -> np.ndarray:
        net = net or self.net
        try:
            if diff_objective:
                objective = -self.objective_function(net) - self.initial_obj
            else:
                objective = -self.objective_function(net)

            self.objective_data.append(objective.sum() if isinstance(objective, np.ndarray) else objective)
            return objective
        except Exception as e:
            logging.error(f"Error in calculate_objective: {str(e)}")
            return np.array([0.0])

    def calculate_violations(self, net=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        net = net or self.net
        valids = []
        violations = []
        penalties = []
        for constraint in self.constraints:
            result = constraint.get_violation_metrics(net)
            valids.append(result['valid'])
            violations.append(result['violation'])
            penalties.append(result['penalty'])

        return np.array(valids), np.array(violations), np.array(penalties)

    def _get_obs(self, obs_keys, add_time_obs=False, add_mean_obs=False
                 ) -> np.ndarray:
        obss = [(self.net[unit_type].loc[idxs, column].to_numpy())
                if (unit_type != 'load' or not self.bus_wise_obs)
                else get_bus_aggregated_obs(self.net, 'load', column, idxs)
                for unit_type, column, idxs in obs_keys]

        if add_mean_obs:
            mean_obs = [np.mean(partial_obs) for partial_obs in obss
                        if len(partial_obs) > 1]
            obss.append(mean_obs)

        if add_time_obs and self.current_simbench_step is not None:
            time_obs = get_simbench_time_observation(
                self.profiles, self.current_simbench_step)
            obss = [time_obs] + obss

        return np.concatenate(obss)

    def get_state(self) -> np.ndarray:

        return self._get_obs(self.state_keys)

    def render(self, **kwargs):

        ax = pp.plotting.simple_plot(self.net, **kwargs)
        return ax

    def get_current_actions(self, net=None, from_results_table=True) -> np.ndarray:
        # Attention: These are not necessarily the actions of the RL agent
        # because some re-scaling might have happened!
        # These are the actions from the original action space [0, 1]
        net = net or self.net
        res_prefix = 'res_' if from_results_table else ''
        action = []
        for unit_type, column, idxs in self.act_keys:
            setpoints = net[f'{res_prefix}{unit_type}'].loc[idxs, column]

            # If data not taken from results table, scaling required
            if not from_results_table and 'scaling' in net[unit_type].columns:
                setpoints *= net[unit_type].scaling.loc[idxs]

            # Action space depends on autoscaling
            min_id = 'min_' if self.autoscale_actions else 'min_min_'
            max_id = 'max_' if self.autoscale_actions else 'max_max_'
            min_values = net[unit_type][f'{min_id}{column}'].loc[idxs]
            max_values = net[unit_type][f'{max_id}{column}'].loc[idxs]

            action.append((setpoints - min_values) / (max_values - min_values))

        return np.concatenate(action)

    def get_actions(self) -> np.ndarray:

        if self.power_flow_available:
            return self.get_current_actions(from_results_table=True)
        return self.get_current_actions(from_results_table=False)

    def get_optimal_actions(self) -> np.ndarray:

        self.ensure_optimal_power_flow_available()
        # The pandapower OPF stores the optimal settings only in the results table
        return self.get_current_actions(self.optimal_net, from_results_table=True)

    def is_state_valid(self) -> bool:

        self.ensure_power_flow_available()
        valids, _, _ = self.calculate_violations(self.net)
        return valids.all()

    def is_optimal_state_valid(self) -> bool:

        self.ensure_optimal_power_flow_available()
        valids, _, _ = self.calculate_violations(self.optimal_net)
        return valids.all()

    def get_objective(self) -> float:
        """ Returns the currrent value of the objective function. """
        self.ensure_power_flow_available()
        return sum(self.calculate_objective(self.net))

    def get_optimal_objective(self) -> float:
        """ Returns the optimal value of the objective function. Warning: Can
        only be called if :meth:`run_optimal_power_flow` method was called before. """
        self.ensure_optimal_power_flow_available()
        return sum(self.calculate_objective(self.optimal_net))

    def run_power_flow(self, **kwargs):
        success = self._run_power_flow(self.net, **kwargs)
        self.power_flow_available = bool(success)
        return success

    def run_optimal_power_flow(self, **kwargs):

        self.optimal_net = copy.deepcopy(self.net)
        try:
            self._run_optimal_power_flow(self.optimal_net, **kwargs)
            self.optimal_power_flow_available = True
            return True
        except pp.optimal_powerflow.OPFNotConverged:
            logging.warning('OPF not converged!!!')
            return False

    def ensure_power_flow_available(self):
        if not self.power_flow_available:
            raise PowerFlowNotAvailable('Please call `run_power_flow` first!')

    def ensure_optimal_power_flow_available(self):
        if not self.optimal_power_flow_available:
            raise PowerFlowNotAvailable('Please call `run_optimal_power_flow` first!')

    def set_power_flow_unavailable(self):
        """ Reset the power flow availability to indicate that a new power flow
        or OPF calculation is required. """
        self.power_flow_available = False
        self.optimal_power_flow_available = False

    @staticmethod
    def default_optimal_power_flow(net, calculate_voltage_angles=False, **kwargs):

        try:
            pp.runopp(net,
                      calculate_voltage_angles=calculate_voltage_angles,
                      init='pf',
                      # verbose=True
                      **kwargs)
        except Exception as e:
            logging.error(f"OPF failed: {str(e)}")
            raise pp.optimal_powerflow.OPFNotConverged(str(e))

    def forward_backward_power_flow(self, net):

        P_load = net.load.groupby('bus')['p_mw'].sum()
        Q_load = net.load.groupby('bus')['q_mvar'].sum()
        P_pv = net.sgen.groupby('bus')['p_mw'].sum()
        Q_pv = net.sgen.groupby('bus')['q_mvar'].sum()
        P_strg = net.storage.groupby('bus')['p_mw'].sum()
        Q_strg = net.storage.groupby('bus')['q_mvar'].sum() if hasattr(net.storage, 'q_mvar') else pd.Series(0,
                                                                                                             index=net.storage.index)

        Pnet = (P_load.reindex(net.bus.index, fill_value=0) -
                P_pv.reindex(net.bus.index, fill_value=0) -
                P_strg.reindex(net.bus.index, fill_value=0))
        Qnet = (Q_load.reindex(net.bus.index, fill_value=0) -
                Q_pv.reindex(net.bus.index, fill_value=0) -
                Q_strg.reindex(net.bus.index, fill_value=0))

        Bus = np.column_stack([
            net.bus.index.values + 1,
            Pnet.values * 1000,
            Qnet.values * 1000
        ])

        br = []
        for idx, row in net.line.iterrows():
            br.append([
                idx + 1,
                row.from_bus + 1,
                row.to_bus + 1,
                row.r_ohm_per_km * row.length_km,
                row.x_ohm_per_km * row.length_km
            ])
        Branch = np.array(br)

        Ploss_mat, Qloss_mat, Vbus = compute_line_loss(Bus, Branch)

        net.res_bus = pd.DataFrame(index=net.bus.index)
        net.res_bus['vm_pu'] = (Vbus.flatten() / 12.66)
        net.res_bus['va_degree'] = 0.0  # 若需要可补充

        net.res_line = pd.DataFrame(index=net.line.index,
                                    columns=['pl_mw', 'ql_mvar'],
                                    data=0.0)
        for idx, row in net.line.iterrows():
            i, j = row.from_bus, row.to_bus
            net.res_line.at[idx, 'pl_mw'] = Ploss_mat[i, j] / 1000
            net.res_line.at[idx, 'ql_mvar'] = Qloss_mat[i, j] / 1000

        net.res_sgen = pd.DataFrame(index=net.sgen.index)
        net.res_sgen['p_mw'] = net.sgen['p_mw']
        net.res_sgen['q_mvar'] = net.sgen['q_mvar']

        self.power_flow_available = True
        return True
    # ========== END ==========


class OpfCase(OpfEnv):
    def __init__(self, daily_pv_profile=None,
                 pv_capacity_kw=2000,
                 train_data='normal_around_mean',
                 test_data='normal_around_mean',
                 # ========== 电压调节参数 ==========
                 voltage_regulation_enabled=True,  # 启用电压调节
                 v_upper=1.05,  # 电压上限 (p.u.)
                 v_lower=0.95,  # 电压下限 (p.u.)
                 v_nominal=1.0,  # 额定电压 (p.u.)
                 voltage_penalty_factor=10.0,  # 电压越限惩罚系数 ρ1
                 multi_objective=True,  # 多目标优化开关
                 voltage_weight=0,  # 电压目标权重
                 loss_weight=1,  # 网损目标权重
                 # =================================
                 *args, **kwargs):

        self.pv_capacity_kw = pv_capacity_kw

        # ========== 电压调节参数初始化 ==========
        self.voltage_regulation_enabled = voltage_regulation_enabled
        self.v_upper = v_upper
        self.v_lower = v_lower
        self.v_nominal = v_nominal
        self.voltage_penalty_factor = voltage_penalty_factor
        self.multi_objective = multi_objective
        self.voltage_weight = voltage_weight
        self.loss_weight = loss_weight

        # 电压统计信息
        self.voltage_violation_count = 0
        self.total_voltage_checks = 0
        self.voltage_success_rate = 1.0
        self.hourly_voltage_stats = []
        self.daily_voltage_violations = 0
        # =======================================

        if daily_pv_profile is not None:
            self.fixed_daily_pv = daily_pv_profile
        else:

            self.fixed_daily_pv = np.array([
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.1, 0.2, 0.2, 0.3,
                0.4, 0.4, 0.5, 0.6, 0.6,
                0.5, 0.4, 0.2, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ])

        self.current_hour = 0
        self.current_day = 0
        self.daily_losses = []

        assert 'simbench' not in train_data and 'simbench' not in test_data, "Only non-simbench networks are supported."

        net = self._define_opf()

        self.current_hour = 0
        self.current_day = 0
        self.daily_losses = []

        assert 'simbench' not in train_data and 'simbench' not in test_data, "Only non-simbench networks are supported."

        net = self._define_opf()

        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
            ('storage', 'soc_percent', net.storage.index),
            ('res_sgen', 'p_mw', net.sgen.index),
            # ========== 添加电压观测 ==========
            ('res_bus', 'vm_pu', net.bus.index),  # 所有母线电压
            # =================================
        ]
        add_time_obs = True

        act_keys = [
            ('storage', 'p_mw', net.storage.index),  # Storage control
        ]

        kwargs.setdefault('state_keys', [])

        super().__init__(
            net,
            act_keys,
            obs_keys,
            *args,
            **kwargs
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(24,),
            dtype=np.float32,
        )

        self.current_hour = 0

        for load_idx in net.load.index:
            self.net.load.loc[load_idx, 'min_p_mw'] = 1 * self.net.load.loc[load_idx, 'p_mw']
            self.net.load.loc[load_idx, 'max_p_mw'] = 1 * self.net.load.loc[load_idx, 'p_mw']
            self.net.load.loc[load_idx, 'min_q_mvar'] = 1 * self.net.load.loc[load_idx, 'q_mvar']
            self.net.load.loc[load_idx, 'max_q_mvar'] = 1 * self.net.load.loc[load_idx, 'q_mvar']

            self.load_time_profile = {
                1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.3,
                6: 0.4, 7: 0.7, 8: 0.7, 9: 0.7, 10: 0.8, 11: 0.8,
                12: 0.9, 13: 0.8, 14: 0.7, 15: 0.6, 16: 0.5, 17: 0.4,
                18: 0.6, 19: 0.7, 20: 0.9, 21: 0.8, 22: 0.7, 23: 0.5, 24: 0.4
            }

            self.daily_losses = []
            self.current_day_loss = 0.0

    # ========== 电压调节核心方法 ==========

    def get_all_bus_voltages(self):
        """
        获取所有母线电压 (p.u.)
        返回: numpy array of shape (n_buses,)
        """
        if not self.power_flow_available:
            return None
        return self.net.res_bus['vm_pu'].values

    def calculate_voltage_violation(self):
        """
        计算电压越限情况
        基于论文 MASAC 的分层奖励设计思路

        返回: {
            'v_max': 最大电压,
            'v_min': 最小电压,
            'v_mean': 平均电压,
            'v_std': 电压标准差,
            'upper_violation': 上限越限量,
            'lower_violation': 下限越限量,
            'total_violation': 总越限量,
            'n_violations': 越限节点数,
            'violation_ratio': 越限比例,
            'is_valid': 是否全部满足约束
        }
        """
        if not self.voltage_regulation_enabled or not self.power_flow_available:
            return None

        voltages = self.get_all_bus_voltages()
        if voltages is None:
            return None

        v_max = np.max(voltages)
        v_min = np.min(voltages)
        v_mean = np.mean(voltages)
        v_std = np.std(voltages)

        # 计算越限量 (参考论文中的 [·]+ 运算符)
        upper_violation = np.sum(np.maximum(voltages - self.v_upper, 0))
        lower_violation = np.sum(np.maximum(self.v_lower - voltages, 0))
        total_violation = upper_violation + lower_violation

        # 统计越限节点
        n_upper_violations = np.sum(voltages > self.v_upper)
        n_lower_violations = np.sum(voltages < self.v_lower)
        n_violations = n_upper_violations + n_lower_violations

        violation_ratio = n_violations / len(voltages)
        is_valid = (n_violations == 0)

        return {
            'v_max': v_max,
            'v_min': v_min,
            'v_mean': v_mean,
            'v_std': v_std,
            'upper_violation': upper_violation,
            'lower_violation': lower_violation,
            'total_violation': total_violation,
            'n_violations': n_violations,
            'n_upper_violations': n_upper_violations,
            'n_lower_violations': n_lower_violations,
            'violation_ratio': violation_ratio,
            'is_valid': is_valid,
            'all_voltages': voltages
        }

    def calculate_voltage_reward(self):
        """
        计算电压调节奖励（分层奖励设计）
        基于论文 MASAC: 约束与目标解耦

        奖励结构:
        1. 若电压越限: 大负奖励 = -ρ1 * total_violation
        2. 若电压满足: 电压优化奖励 = -voltage_deviation

        返回: (reward, voltage_info_dict)
        """
        if not self.voltage_regulation_enabled:
            return 0.0, None

        v_info = self.calculate_voltage_violation()
        if v_info is None:
            return 0.0, None

        # 统计更新
        self.total_voltage_checks += len(v_info['all_voltages'])
        self.voltage_violation_count += v_info['n_violations']

        if not v_info['is_valid']:
            # 情况1: 电压越限 - 给予大惩罚（约束优先）
            # 参考论文: R = -ρ1 * [max(V_t - V̄_t, 0) + max(V̲_t - V_t, 0)]
            reward = -self.voltage_penalty_factor * v_info['total_violation']
            self.daily_voltage_violations += v_info['n_violations']
        else:
            # 情况2: 电压满足 - 给予电压质量奖励
            # 参考论文多目标优化: 最小化电压偏差
            voltage_deviation = np.sum(np.abs(v_info['all_voltages'] - self.v_nominal))
            # 可选: 使用电压方差作为奖励（提高电压质量）
            voltage_quality_reward = -voltage_deviation - 0.5 * v_info['v_std']
            reward = voltage_quality_reward

        v_info['reward'] = reward
        return reward, v_info

    def calculate_voltage_success_rate(self):
        """计算电压调节成功率 S_v = (1 - N_o/N_v) × 100%"""
        if self.total_voltage_checks == 0:
            return 100.0
        violation_rate = self.voltage_violation_count / self.total_voltage_checks
        return (1 - violation_rate) * 100.0

    def reset_voltage_stats(self):
        """重置电压统计"""
        self.voltage_violation_count = 0
        self.total_voltage_checks = 0
        self.hourly_voltage_stats = []
        self.daily_voltage_violations = 0

    # =====================================

    def calculate_total_line_loss(self) -> float:

        return sum(self.daily_losses) if self.daily_losses else 0.0

    def _apply_time_based_load(self):

        current_hour = self.current_hour

        load_ratio = self.load_time_profile.get(current_hour, 1.0)  # 默认值为1.0

        for load_idx in self.net.load.index:

            if 'original_p_mw' not in self.net.load.columns:
                self.net.load['original_p_mw'] = self.net.load['p_mw'].copy()
                self.net.load['original_q_mvar'] = self.net.load['q_mvar'].copy()

            original_p = self.net.load.at[load_idx, 'original_p_mw']
            original_q = self.net.load.at[load_idx, 'original_q_mvar']
            self.net.load.at[load_idx, 'p_mw'] = original_p * load_ratio
            self.net.load.at[load_idx, 'q_mvar'] = original_q * load_ratio

    def _update_pv_output(self):

        ratio = self.fixed_daily_pv[self.current_hour - 1]

        pv_output_mw = (self.pv_capacity_kw / 1000) * ratio

        self.net.sgen['p_mw'] = pv_output_mw

    def calculate_line_losses(self, net):

        pl_mw = net.res_line['pl_mw'].values  # shape = (branch_num,)
        line_losses_w = pl_mw * 1e6  # MW -> W
        total_loss_w = np.sum(line_losses_w)
        return line_losses_w, total_loss_w

    def get_line_costs(self, net):

        _, total_loss_w = self.calculate_line_losses(net)
        total_loss_mw = total_loss_w / 1e6
        return np.array([total_loss_mw])

    def _define_opf(self):
        net = pp.create_empty_network()

        # 141总线数据
        Bus_141 = np.array([
            [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0],
            [6, 0, 0], [7, 0, 0], [8, 63.75, 39.53], [9, 8.5, 5.27], [10, 0, 0],
            [11, 0, 0], [12, 21.25, 13.17], [13, 63.75, 39.53], [14, 0, 0], [15, 0, 0],
            [16, 0, 0], [17, 127.5, 79.06], [18, 0, 0], [19, 0, 0], [20, 63.75, 39.53],
            [21, 63.75, 39.53], [22, 0, 0], [23, 63.75, 39.53], [24, 0, 0], [25, 0, 0],
            [26, 127.5, 79.06], [27, 63.75, 39.53], [28, 0, 0], [29, 63.75, 39.53], [30, 0, 0],
            [31, 0, 0], [32, 127.5, 79.06], [33, 0, 0], [34, 127.5, 79.06], [35, 255, 158.11],
            [36, 127.5, 79.06], [37, 42.5, 26.35], [38, 0, 0], [39, 17, 10.54], [40, 0, 0],
            [41, 63.75, 39.53], [42, 0, 0], [43, 0, 0], [44, 42.5, 26.35], [45, 0, 0],
            [46, 0, 0], [47, 0, 0], [48, 106.25, 65.88], [49, 127.5, 79.06], [50, 0, 0],
            [51, 106.25, 65.88], [52, 63.75, 39.53], [53, 85, 52.70], [54, 0, 0], [55, 0, 0],
            [56, 21.25, 13.17], [57, 0, 0], [58, 255, 158.11], [59, 127.5, 79.06], [60, 0, 0],
            [61, 255, 158.11], [62, 170, 105.41], [63, 0, 0], [64, 255, 158.11], [65, 127.5, 79.06],
            [66, 191.25, 118.59], [67, 42.5, 26.35], [68, 85, 52.70], [69, 255, 158.11], [70, 0, 0],
            [71, 255, 158.11], [72, 127.5, 79.06], [73, 255, 158.11], [74, 255, 158.11], [75, 38.25, 23.72],
            [76, 63.75, 39.53], [77, 127.5, 79.06], [78, 0, 0], [79, 427.125, 264.86], [80, 637.5, 395.28],
            [81, 0, 0], [82, 127.5, 79.06], [83, 63.75, 39.53], [84, 191.25, 118.59], [85, 0, 0],
            [86, 425, 263.52], [87, 127.5, 79.06], [88, 63.75, 39.53], [89, 55.25, 34.26], [90, 0, 0],
            [91, 0, 0], [92, 0, 0], [93, 0, 0], [94, 93.5, 57.98], [95, 0, 0],
            [96, 127.5, 79.06], [97, 0, 0], [98, 255, 158.11], [99, 0, 0], [100, 255, 158.11],
            [101, 12.75, 7.91], [102, 0, 0], [103, 106.25, 65.88], [104, 0, 0], [105, 255, 158.11],
            [106, 127.5, 79.06], [107, 427.125, 264.86], [108, 0, 0], [109, 637.5, 395.28], [110, 637.5, 395.28],
            [111, 21.25, 13.17], [112, 425, 263.52], [113, 63.75, 39.53], [114, 0, 0], [115, 0, 0],
            [116, 255, 158.11], [117, 55.25, 34.26], [118, 0, 0], [119, 93.5, 57.98], [120, 0, 0],
            [121, 0, 0], [122, 0, 0], [123, 85, 52.70], [124, 106.25, 65.88], [125, 0, 0],
            [126, 0, 0], [127, 63.75, 39.53], [128, 63.75, 39.53], [129, 93.5, 57.98], [130, 95.625, 59.29],
            [131, 0, 0], [132, 63.75, 39.53], [133, 38.25, 23.72], [134, 29.75, 18.45], [135, 21.25, 13.17],
            [136, 63.75, 39.53], [137, 46.75, 28.99], [138, 42.5, 26.35], [139, 42.5, 26.35], [140, 127.5, 79.06],
            [141, 63.75, 39.53]
        ])

        # 141分支数据
        Branch_141 = np.array([
            [1, 1, 20, 0.0577, 0.0409], [2, 2, 3, 0.1725, 0.1223], [3, 3, 4, 0.0009, 0.0006],
            [4, 4, 5, 0.0092, 0.0065], [5, 5, 6, 0.0068, 0.0049], [6, 6, 7, 0.0469, 0.0625],
            [7, 7, 8, 0.0736, 0.0981], [8, 8, 9, 0.0649, 0.0459], [9, 9, 10, 0.0507, 0.0359],
            [10, 10, 11, 0.0116, 0.0082], [11, 11, 12, 0.1291, 0.0913], [12, 12, 13, 0.1227, 0.0866],
            [13, 13, 14, 0.0488, 0.0345], [14, 14, 15, 0.0957, 0.0677], [15, 15, 16, 0.0860, 0.0609],
            [16, 16, 17, 0.0398, 0.0282], [17, 17, 18, 0.0828, 0.0566], [18, 18, 19, 0.0186, 0.0132],
            [19, 19, 20, 0.0559, 0.0395], [20, 20, 21, 0.0365, 0.0246], [21, 21, 22, 0.0573, 0.0307],
            [22, 22, 23, 0.0263, 0.0191], [23, 23, 24, 0.0683, 0.0497], [24, 24, 25, 0.0398, 0.0282],
            [25, 25, 26, 0.0729, 0.0530], [26, 26, 27, 0.0335, 0.0244], [27, 27, 28, 0.0584, 0.0414],
            [28, 28, 29, 0.0655, 0.0463], [29, 61, 62, 0.0411, 0.0291], [30, 60, 63, 0.0353, 0.0250],
            [31, 63, 64, 0.1047, 0.0741], [32, 64, 65, 0.0674, 0.0477], [33, 65, 66, 0.0302, 0.0214],
            [34, 66, 67, 0.0456, 0.0323], [35, 67, 68, 0.0218, 0.0154], [36, 70, 72, 0.0700, 0.0495],
            [37, 42, 73, 0.0231, 0.0164], [38, 73, 74, 0.0030, 0.0064], [39, 43, 75, 0.0379, 0.0268],
            [40, 44, 76, 0.0552, 0.0391], [41, 46, 77, 0.0516, 0.0436], [42, 76, 78, 0.0167, 0.0110],
            [43, 78, 79, 0.0415, 0.0101], [44, 79, 80, 0.1003, 0.0244], [45, 79, 81, 0.1513, 0.0370],
            [46, 81, 82, 0.0033, 0.0008], [47, 47, 83, 0.0085, 0.0062], [48, 49, 84, 0.0517, 0.0449],
            [49, 50, 85, 0.0147, 0.0036], [50, 85, 86, 0.0037, 0.0016], [51, 86, 87, 0.0000, 0.00001],
            [52, 7, 88, 0.0174, 0.0231], [53, 88, 89, 0.0469, 0.0625], [54, 89, 90, 0.0299, 0.0398],
            [55, 90, 91, 0.0212, 0.0283], [56, 91, 92, 0.0315, 0.0420], [57, 92, 93, 0.0280, 0.0373],
            [58, 93, 94, 0.0206, 0.0274], [59, 94, 95, 0.0206, 0.0274], [60, 89, 96, 0.0687, 0.0486],
            [61, 96, 97, 0.0970, 0.0686], [62, 97, 98, 0.0902, 0.0196], [63, 97, 99, 0.0033, 0.0008],
            [64, 131, 132, 0.0347, 0.0245], [65, 131, 133, 0.0920, 0.0669], [66, 121, 134, 0.0841, 0.0612],
            [67, 16, 135, 0.0527, 0.0373], [68, 16, 136, 0.0302, 0.0214], [69, 18, 137, 0.0584, 0.0414],
            [70, 23, 138, 0.0769, 0.0559], [71, 29, 30, 0.0342, 0.0248], [72, 30, 31, 0.0128, 0.0091],
            [73, 31, 32, 0.0347, 0.0245], [74, 2, 33, 0.0443, 0.0314], [75, 33, 34, 0.0020, 0.0009],
            [76, 5, 35, 0.2274, 0.0554], [77, 5, 36, 0.1265, 0.1565], [78, 6, 37, 0.0055, 0.0073],
            [79, 37, 38, 0.2036, 0.1440], [80, 38, 39, 0.0938, 0.0663], [81, 39, 40, 0.0347, 0.0245],
            [82, 40, 41, 0.0918, 0.0650], [83, 41, 42, 0.2318, 0.1640], [84, 42, 43, 0.1207, 0.0854],
            [85, 43, 44, 0.0443, 0.0314], [86, 44, 45, 0.0405, 0.0288], [87, 45, 46, 0.0160, 0.0127],
            [88, 46, 47, 0.0636, 0.0450], [89, 47, 48, 0.0417, 0.0295], [90, 48, 49, 0.0732, 0.0510],
            [91, 49, 50, 0.0828, 0.0556], [92, 50, 51, 0.0398, 0.0282], [93, 51, 52, 0.0225, 0.0159],
            [94, 38, 53, 0.0841, 0.0595], [95, 42, 54, 0.0161, 0.0114], [96, 54, 55, 0.0527, 0.0373],
            [97, 55, 56, 0.0893, 0.0632], [98, 56, 57, 0.0867, 0.0613], [99, 57, 58, 0.0674, 0.0477],
            [100, 58, 59, 0.0469, 0.0332], [101, 55, 60, 0.0334, 0.0236], [102, 60, 61, 0.0327, 0.0232],
            [103, 63, 69, 0.0366, 0.0259], [104, 55, 70, 0.0231, 0.0164], [105, 70, 71, 0.0120, 0.0029],
            [106, 99, 100, 0.0033, 0.0008], [107, 91, 101, 0.0231, 0.0164], [108, 101, 102, 0.0578, 0.0409],
            [109, 102, 103, 0.0889, 0.0217], [110, 103, 104, 0.0629, 0.0153], [111, 104, 105, 0.1170, 0.0285],
            [112, 104, 106, 0.0114, 0.0026], [113, 92, 107, 0.0849, 0.0207], [114, 94, 108, 0.0612, 0.0260],
            [115, 108, 109, 0.0452, 0.0192], [116, 94, 110, 0.0033, 0.0008], [117, 7, 111, 0.0719, 0.0509],
            [118, 10, 112, 0.1070, 0.0261], [119, 11, 113, 0.0347, 0.0245], [120, 13, 114, 0.0623, 0.0441],
            [121, 114, 115, 0.0668, 0.0473], [122, 115, 116, 0.0040, 0.0010], [123, 14, 117, 0.0506, 0.0366],
            [124, 15, 118, 0.0161, 0.0114], [125, 118, 119, 0.0462, 0.0327], [126, 119, 120, 0.0424, 0.0300],
            [127, 120, 121, 0.0507, 0.0359], [128, 121, 122, 0.0732, 0.0518], [129, 122, 123, 0.0584, 0.0414],
            [130, 123, 124, 0.0610, 0.0432], [131, 124, 125, 0.0783, 0.0554], [132, 125, 126, 0.0834, 0.0607],
            [133, 126, 127, 0.0347, 0.0245], [134, 127, 128, 0.0570, 0.0420], [135, 128, 129, 0.0585, 0.0425],
            [136, 129, 130, 0.0103, 0.0073], [137, 119, 131, 0.0355, 0.0253], [138, 25, 139, 0.0950, 0.0673],
            [139, 30, 140, 0.0519, 0.0377], [140, 31, 141, 0.0584, 0.0414]
        ])

        # 创建141个总线
        for i in range(1, 142):
            pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}", in_service=True)

        # 创建线路
        for branch in Branch_141:
            line_idx = pp.create_line(
                net,
                from_bus=int(branch[1]) - 1,  # 转换为0-based索引
                to_bus=int(branch[2]) - 1,  # 转换为0-based索引
                length_km=1,
                std_type="NAYY 4x50 SE",
                name=f"Line {int(branch[0])}"
            )

            # 设置线路的电阻和电抗
            net.line.loc[line_idx, "r_ohm_per_km"] = branch[3]
            net.line.loc[line_idx, "x_ohm_per_km"] = branch[4]

        # 创建负荷
        for bus in Bus_141:
            pp.create_load(
                net,
                bus=int(bus[0]) - 1,  # 转换为0-based索引
                p_mw=bus[1] / 1000,  # kW -> MW
                q_mvar=bus[2] / 1000  # kVar -> MVar
            )

        # 保存原始负荷值
        net.load['original_p_mw'] = net.load['p_mw']
        net.load['original_q_mvar'] = net.load['q_mvar']

        # 创建外部电网
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0)

        # 创建光伏节点 - 根据141节点算例配置多个光伏节点
        pv_nodes = [18, 24, 32]  # 对应节点19, 25, 33
        pv_capacity_mw = self.pv_capacity_kw / 1000 / len(pv_nodes)  # 平均分配到每个光伏节点

        for bus_idx in pv_nodes:
            pp.create_sgen(net, bus=bus_idx - 1, p_mw=0, q_mvar=0,
                           min_p_mw=0, max_p_mw=pv_capacity_mw,
                           name=f"PV at bus {bus_idx}")

        # 创建储能节点 - 选择节点33（索引32）
        storage_bus = 32  # 节点33
        storage_capacity = 50  # 增加储能容量以适应141节点系统
        storage_power = 5  # 增加储能功率

        pp.create_storage(net, bus=storage_bus - 1, p_mw=0, q_mvar=0,
                          min_p_mw=-storage_power,
                          max_p_mw=storage_power,
                          min_e_mwh=0.2 * storage_capacity,
                          max_e_mwh=storage_capacity,
                          soc_percent=50,
                          name="Battery Storage")
        return net

    def reset(self, **kwargs):

        self.daily_losses = []
        self.current_hour = 1
        self.current_day += 1

        # ========== 重置电压统计 ==========
        self.reset_voltage_stats()
        # =================================

        print(f"")
        print(f"===== Day {self.current_day} - 24-hour simulation begins =====")

        for idx in self.net.load.index:
            self.net.load.at[idx, 'p_mw'] = self.net.load.at[idx, 'original_p_mw']
            self.net.load.at[idx, 'q_mvar'] = self.net.load.at[idx, 'original_q_mvar']

        self._apply_time_based_load()
        self._update_pv_output()

        self.net.storage.loc[:, 'soc_percent'] = 50.0
        initial_soc = self.net.storage['soc_percent'].values[0]
        print(f"Initial energy storage SOC: {initial_soc:.1f}%")

        self.forward_backward_power_flow(self.net)

        obs = self._get_obs(self.obs_keys, self.add_time_obs, self.add_mean_obs)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs, {}

    def step(self, action_24h: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict]:
        hourly_rewards = []
        hourly_info = []
        self.daily_losses = []
        total_loss_with_storage = 0.0
        total_loss_no_storage = 0.0
        total_delivery_with_storage = 0.0
        total_delivery_no_storage = 0.0
        storage_idx = self.net.storage.index[0]

        # ========== 电压统计重置 ==========
        daily_voltage_stats = []
        # =================================

        for hour in range(1, 25):

            self.current_hour = hour
            self._apply_time_based_load()
            self._update_pv_output()

            current_action = action_24h[hour - 1]
            self._apply_storage_action(np.array([current_action]))
            current_p = self.net.storage.at[storage_idx, 'p_mw']
            storage_soc = self.net.storage['soc_percent'].values[0]

            self.forward_backward_power_flow(self.net)

            # ========== 电压调节计算 ==========
            voltage_reward = 0.0
            voltage_info = None
            if self.voltage_regulation_enabled:
                voltage_reward, voltage_info = self.calculate_voltage_reward()
            # =================================

            line_loss_with_storage = self.get_line_costs(self.net)[0] * 1000
            total_loss_with_storage += line_loss_with_storage
            load_power = self.net.load['p_mw'].sum() * 1000
            pv_output = self.net.sgen['p_mw'].sum() * 1000
            storage_discharge_power = current_p * 1000 if current_p > 0 else 0
            storage_charge_power = -current_p * 1000 if current_p < 0 else 0

            hourly_delivery_with_storage = abs(pv_output) + abs(load_power) + abs(storage_discharge_power) + abs(
                storage_charge_power)
            total_delivery_with_storage += hourly_delivery_with_storage

            line_loss_rate_with_storage = self.calculate_loss_rate(
                line_loss_with_storage,
                hourly_delivery_with_storage
            )

            original_storage_p = self.net.storage.at[storage_idx, 'p_mw']
            self.net.storage.at[storage_idx, 'p_mw'] = 0
            self.forward_backward_power_flow(self.net)
            line_loss_no_storage = self.get_line_costs(self.net)[0] * 1000  # MW -> kW
            self.net.storage.at[storage_idx, 'p_mw'] = original_storage_p
            total_loss_no_storage += line_loss_no_storage

            hourly_delivery_no_storage = abs(pv_output) + abs(load_power)
            total_delivery_no_storage += hourly_delivery_no_storage

            line_loss_rate_no_storage = self.calculate_loss_rate(
                line_loss_no_storage,
                hourly_delivery_no_storage
            )

            valids, violations, penalties = self.calculate_violations()
            penalty = np.sum(penalties)
            valid = valids.all()

            # ========== 多目标奖励计算 ==========
            # 基于论文的多目标优化思路
            loss_objective = -line_loss_with_storage / 1000  # 网损目标

            if self.multi_objective and self.voltage_regulation_enabled:
                # 多目标: 结合网损和电压调节
                # 参考论文 MOMADRL 的向量奖励设计
                total_reward = (self.loss_weight * loss_objective +
                                self.voltage_weight * voltage_reward)
            else:
                # 单目标或电压调节未启用
                total_reward = loss_objective + voltage_reward

            # 添加约束惩罚
            if not valid:
                total_reward -= 10
            # =================================

            hourly_rewards.append(total_reward)
            self.daily_losses.append(line_loss_with_storage)

            # ========== 记录电压统计 ==========
            if voltage_info:
                daily_voltage_stats.append({
                    'hour': hour,
                    **voltage_info
                })
            # =================================

            storage_action_type = "charge" if current_p < 0 else "discharge"
            storage_power_kw = abs(current_p) * 1000

            # ========== 增强打印信息 ==========
            voltage_str = ""
            if voltage_info:
                voltage_str = (f", Vmax={voltage_info['v_max']:.3f}, "
                               f"Vmin={voltage_info['v_min']:.3f}, "
                               f"Valid={voltage_info['is_valid']}")

            print(f"Hour {hour:2}: Loss={line_loss_with_storage:7.4f} kW, "
                  f"PV={pv_output:7.1f} kW, "
                  f"Storage={storage_action_type} {storage_power_kw:7.1f} kW, "
                  f"SOC={storage_soc:6.1f}%, "
                  f"Reward={total_reward:7.3f} "
                  f"LossRate={line_loss_rate_with_storage:5.2f}% (Base:{line_loss_rate_no_storage:5.2f}%)"
                  f"{voltage_str}")
            # =================================

            hourly_info.append({
                "hour": hour,
                "line_loss_with_storage": line_loss_with_storage,
                "line_loss_no_storage": line_loss_no_storage,
                "loss_rate_with_storage": line_loss_rate_with_storage,
                "loss_rate_no_storage": line_loss_rate_no_storage,
                "pv_output": pv_output,
                "storage_power": current_p * 1000,
                "storage_soc": storage_soc,
                "penalty": penalty,
                "valid": valid,
                "reward": total_reward,
                "objective": loss_objective,
                "delivery_with_storage": hourly_delivery_with_storage,
                "delivery_no_storage": hourly_delivery_no_storage,
                # ========== 添加电压信息 ==========
                "voltage_info": voltage_info,
                # =================================
            })

        total_reward = np.sum(hourly_rewards)

        if total_delivery_with_storage > 0:
            daily_loss_rate_with_storage = (total_loss_with_storage / total_delivery_with_storage) * 100
        else:
            daily_loss_rate_with_storage = np.nan

        if total_delivery_no_storage > 0:
            daily_loss_rate_no_storage = (total_loss_no_storage / total_delivery_no_storage) * 100
        else:
            daily_loss_rate_no_storage = np.nan

        # ========== 计算电压成功率 ==========
        voltage_success_rate = self.calculate_voltage_success_rate()
        # =================================

        print(f"\n[DAY SUMMARY] Day {self.current_day}:")
        print(f"Bus loss (with energy storage): {total_loss_with_storage:.1f} kW")
        print(f"Bus loss (without energy storage): {total_loss_no_storage:.1f} kW")
        print(f"Line loss reduction: {total_loss_no_storage - total_loss_with_storage:.1f} kW")
        print(f"Bus loss rate (with energy storage): {daily_loss_rate_with_storage:.2f}%")
        print(f"Bus loss rate (without energy storage): {daily_loss_rate_no_storage:.2f}%")
        print(f"Improve: {daily_loss_rate_no_storage - daily_loss_rate_with_storage:.2f}%")

        # ========== 打印电压统计 ==========
        if self.voltage_regulation_enabled:
            print(f"\n[VOLTAGE REGULATION SUMMARY]")
            print(f"Voltage success rate: {voltage_success_rate:.2f}%")
            print(f"Daily voltage violations: {self.daily_voltage_violations}")
            if daily_voltage_stats:
                max_v = max([s['v_max'] for s in daily_voltage_stats])
                min_v = min([s['v_min'] for s in daily_voltage_stats])
                print(f"Daily voltage range: [{min_v:.3f}, {max_v:.3f}] p.u.")
                print(f"Target range: [{self.v_lower}, {self.v_upper}] p.u.")
        # =================================

        print(f"\nTotal transmitted electricity (with energy storage): {total_delivery_with_storage:.1f} kW")
        print(f"Total transmitted electricity (without energy storage): {total_delivery_no_storage:.1f} kW")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Average hourly reward: {total_reward / 24:.3f}")

        self.current_hour = 1
        self._apply_time_based_load()
        self._update_pv_output()
        self.forward_backward_power_flow(self.net)

        next_day_obs = self._get_obs(self.obs_keys, self.add_time_obs, self.add_mean_obs)
        next_day_obs = np.clip(next_day_obs, -1.0, 1.0)

        done = True

        info = {
            "hourly_info": hourly_info,
            "total_reward": total_reward,
            "hourly_rewards": hourly_rewards,
            "total_loss_with_storage": total_loss_with_storage,
            "total_loss_no_storage": total_loss_no_storage,
            "daily_loss_rate_with_storage": daily_loss_rate_with_storage,
            "daily_loss_rate_no_storage": daily_loss_rate_no_storage,
            "average_hourly_reward": total_reward / 24,
            "day": self.current_day - 1,
            "total_delivery_with_storage": total_delivery_with_storage,
            "total_delivery_no_storage": total_delivery_no_storage,
            # ========== 添加电压统计信息 ==========
            "voltage_regulation_enabled": self.voltage_regulation_enabled,
            "voltage_success_rate": voltage_success_rate,
            "daily_voltage_violations": self.daily_voltage_violations,
            "voltage_stats": daily_voltage_stats,
            # =====================================
        }

        return next_day_obs, np.array(hourly_rewards), total_reward, done, False, info

    def _apply_storage_action(self, action: np.ndarray):

        idx = self.net.storage.index[0]
        max_p = self.net.storage.at[idx, 'max_p_mw']

        p_cmd = float(action[0]) * max_p

        e_max = self.net.storage.at[idx, 'max_e_mwh']
        soc = self.net.storage.at[idx, 'soc_percent']
        current_energy = soc / 100 * e_max

        if p_cmd < 0:

            charge_energy = -p_cmd
            max_charge_energy = (100 - soc) / 100 * e_max
            available_charge_energy = e_max - current_energy

            allowed_charge_energy = min(max_charge_energy, available_charge_energy)

            if charge_energy > allowed_charge_energy:
                p_cmd = -allowed_charge_energy

        elif p_cmd > 0:
            discharge_energy = p_cmd

            if discharge_energy > current_energy:
                p_cmd = current_energy

        self.net.storage.at[idx, 'p_mw'] = p_cmd

        delta = -p_cmd / e_max * 100
        self.net.storage.at[idx, 'soc_percent'] = np.clip(soc + delta, 0, 100)

    def calculate_loss_rate(self, loss_power, delivery_power):

        if delivery_power > 0:
            return (loss_power / delivery_power) * 100
        return np.nan


sns.set_style("whitegrid")
sns.set_palette("deep")

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 35,
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'legend.fontsize': 32,
    'figure.titlesize': 35,
    'figure.figsize': (16, 12),
    'lines.linewidth': 4,
    'axes.linewidth': 2,
    'grid.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})


def get_simbench_time_observation(current_step: int, total_n_steps: int = 24 * 4 * 366):
    dayly, weekly, yearly = (24 * 4, 7 * 24 * 4, total_n_steps)
    time_obs = []
    for timeframe in (dayly, weekly, yearly):
        timestep = current_step % timeframe
        cyclical_time = 2 * np.pi * timestep / timeframe
        time_obs.append(np.sin(cyclical_time))
        time_obs.append(np.cos(cyclical_time))

    return np.array(time_obs)


def define_test_train_split(test_share=0.2, random_test_steps=False,
                            validation_share=0.2, random_validation_steps=False,
                            **kwargs):
    """ Return the indices of the simbench test data points. """
    assert test_share + validation_share <= 1.0
    if random_test_steps:
        assert random_validation_steps, 'Random test data does only make sense with also random validation data'

    n_data_points = 24 * 4 * 366
    all_steps = np.arange(n_data_points)

    # Define test dataset
    if test_share == 1.0:
        # Special case: Use the full simbench data set as test set
        return all_steps, np.array([]), np.array([])
    elif test_share == 0.0:
        test_steps = np.array([])
    elif random_test_steps:
        # Randomly sample test data steps from the whole year
        test_steps = np.random.choice(all_steps, int(n_data_points * test_share))
    else:
        # Use deterministic weekly blocks to ensure that all weekdays are equally represented
        # TODO: Allow for arbitrary blocks? Like days or months?
        n_test_weeks = int(52 * test_share)
        # Sample equidistant weeks from the whole year
        test_week_idxs = np.linspace(0, 51, num=n_test_weeks, dtype=int)
        one_week = 7 * 24 * 4
        test_steps = np.concatenate(
            [np.arange(idx * one_week, (idx + 1) * one_week) for idx in test_week_idxs])

    # Define validation dataset
    remaining_steps = np.array(tuple(set(all_steps) - set(test_steps)))
    if validation_share == 1.0:
        return np.array([]), all_steps, np.array([])
    elif validation_share == 0.0:
        validation_steps = np.array([])
    elif random_validation_steps:
        validation_steps = np.random.choice(remaining_steps, int(n_data_points * validation_share))
    else:
        if random_test_steps:
            test_week_idxs = np.array([])

        n_validation_weeks = int(52 * validation_share)
        # Make sure to use only validation weeks that are not already test weeks
        remaining_week_idxs = np.array(tuple(set(np.arange(52)) - set(test_week_idxs)))
        week_pseudo_idxs = np.linspace(0, len(remaining_week_idxs) - 1,
                                       num=n_validation_weeks, dtype=int)
        validation_week_idxs = remaining_week_idxs[week_pseudo_idxs]
        validation_steps = np.concatenate(
            [np.arange(idx * one_week, (idx + 1) * one_week) for idx in validation_week_idxs])

    # Use remaining steps as training steps
    train_steps = np.array(tuple(set(remaining_steps) - set(validation_steps)))

    return test_steps, validation_steps, train_steps


def load_class_from_module(class_name: str, from_module: str) -> Callable:
    """ Load a pre-implemented class from a module. """
    module = importlib.import_module(from_module)
    try:
        return getattr(module, class_name)
    except AttributeError:
        try:
            return getattr(module, class_name.capitalize())
        except AttributeError:
            raise AttributeError(
                f'Class {class_name} not found in module {from_module}!')


def get_obs_and_state_space(net: pp.pandapowerNet, obs_or_state_keys: list,
                            add_time_obs: bool = False, add_mean_obs: bool = False,
                            seed: int = None, last_n_obs: int = 1,
                            bus_wise_obs=False) -> gym.spaces.Box:
    lows, highs = [], []

    if add_time_obs:
        # Time is always given as observation of length 6 in range [-1, 1]
        # at the beginning of the observation!
        lows.append(-np.ones(6))
        highs.append(np.ones(6))

    for unit_type, column, idxs in obs_or_state_keys:
        if 'res_' in unit_type:
            unit_type = unit_type[4:]
        elif 'max_' in column or 'min_' in column:
            column = column[4:]

        if column == 'va_degree':
            l = np.full(len(idxs), -30)
            h = np.full(len(idxs), +30)
        else:
            try:
                if f'min_min_{column}' in net[unit_type].columns:
                    l = net[unit_type][f'min_min_{column}'].loc[idxs].to_numpy()
                else:
                    l = net[unit_type][f'min_{column}'].loc[idxs].to_numpy()
                if f'max_max_{column}' in net[unit_type].columns:
                    h = net[unit_type][f'max_max_{column}'].loc[idxs].to_numpy()
                else:
                    h = net[unit_type][f'max_{column}'].loc[idxs].to_numpy()

                if f'mean_{column}' not in net[unit_type].columns:
                    raise KeyError(f"no mean_{column}")
                mean = net[unit_type][f'mean_{column}'].loc[idxs].to_numpy()
            except KeyError:

                l = np.zeros(len(idxs))
                h = np.ones(len(idxs))
                mean = np.zeros(len(idxs))

        try:
            if 'min' in column or 'max' in column:
                raise AttributeError
            l = l / net[unit_type].scaling.loc[idxs].to_numpy()
            h = h / net[unit_type].scaling.loc[idxs].to_numpy()
        except AttributeError:
            logging.info(f'Scaling for {unit_type} not defined: assume scaling=1')

        if bus_wise_obs and unit_type == 'load':
            buses = sorted(set(net[unit_type].bus))
            l = [sum(l[net[unit_type].bus == bus]) for bus in buses]
            h = [sum(h[net[unit_type].bus == bus]) for bus in buses]
            mean = [sum(mean[net[unit_type].bus == bus]) for bus in buses]

        for _ in range(last_n_obs):
            if len(l) > 0 and len(l) == len(h):
                lows.append(l)
                highs.append(h)
                if add_mean_obs:
                    lows.append(mean)
                    highs.append(mean)

    assert not sum(pd.isna(l).any() for l in lows)
    assert not sum(pd.isna(h).any() for h in highs)

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0), seed=seed
    )


def get_bus_aggregated_obs(net, unit_type, column, idxs) -> np.ndarray:
    """ Aggregate power values that are connected to the same bus to reduce
    state space. """
    df = net[unit_type].iloc[idxs]
    return df.groupby(['bus'])[column].sum().to_numpy()


def assert_only_net_in_signature(function):
    """ Ensure that the function only takes a pandapower net as argument. """
    signature = inspect.signature(function)
    message = 'Function must only take a pandapower net as argument!'
    assert list(signature.parameters.keys()) == ['net'], message


def raise_opf_not_converged(net, **kwargs):
    raise pp.optimal_powerflow.OPFNotConverged(
        "OPF solver not available for this environment.")


def plot_3d_voltage_heatmap(voltage_matrix, title, save_path, v_upper=1.05, v_lower=0.95):
    """
    生成3D立体热力图，显示141总线网络24小时电压波形

    参数:
        voltage_matrix: 电压矩阵 (24小时 x 141总线)
        title: 图形标题
        save_path: 保存路径
        v_upper: 电压上限
        v_lower: 电压下限
    """
    # 创建网格
    hours = np.arange(1, 25)  # 1-24小时
    buses = np.arange(1, 142)  # 1-141总线

    X, Y = np.meshgrid(hours, buses)

    # 电压矩阵转置以适应网格
    Z = voltage_matrix.T  # 形状应为 (141, 24)

    # 创建3D图形
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 创建曲面图
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.8,
                           linewidth=0.5, antialiased=True,
                           vmin=v_lower - 0.05, vmax=v_upper + 0.05)

    # 设置视角
    ax.view_init(elev=30, azim=135)

    # 设置坐标轴标签
    ax.set_xlabel('Time (hour)', fontsize=14, labelpad=10)
    ax.set_ylabel('Bus', fontsize=14, labelpad=10)
    ax.set_zlabel('Voltage (p.u.)', fontsize=14, labelpad=10)

    # 设置坐标轴范围
    ax.set_xlim(1, 24)
    ax.set_ylim(1, 141)
    ax.set_zlim(v_lower - 0.05, v_upper + 0.05)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(1, 25, 4))
    ax.set_yticks(np.arange(1, 142, 20))
    ax.set_zticks(np.arange(v_lower - 0.05, v_upper + 0.06, 0.05))

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Voltage (p.u.)', fontsize=12)

    # 添加标题
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"3D电压热力图已保存到: {save_path}")


def plot_comparison_3d_voltage_heatmaps(voltage_matrix_no_control, voltage_matrix_control,
                                        best_day, save_dir, v_upper=1.05, v_lower=0.95):
    """
    生成对比的3D立体热力图：(a) 无任何控制方法；(b) 采用所提控制方法

    参数:
        voltage_matrix_no_control: 无控制时的电压矩阵 (24小时 x 141总线)
        voltage_matrix_control: 有控制时的电压矩阵 (24小时 x 141总线)
        best_day: 最佳训练天数
        save_dir: 保存目录
        v_upper: 电压上限
        v_lower: 电压下限
    """
    # 创建大图包含两个子图
    fig = plt.figure(figsize=(24, 10))

    # 子图1: 无任何控制方法
    ax1 = fig.add_subplot(121, projection='3d')
    hours = np.arange(1, 25)
    buses = np.arange(1, 142)
    X1, Y1 = np.meshgrid(hours, buses)
    Z1 = voltage_matrix_no_control.T

    surf1 = ax1.plot_surface(X1, Y1, Z1, cmap=cm.jet, alpha=0.8,
                             linewidth=0.5, antialiased=True,
                             vmin=v_lower - 0.05, vmax=v_upper + 0.05)

    ax1.view_init(elev=30, azim=135)
    ax1.set_xlabel('Time (hour)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Bus', fontsize=12, labelpad=10)
    ax1.set_zlabel('Voltage (p.u.)', fontsize=12, labelpad=10)
    ax1.set_xlim(1, 24)
    ax1.set_ylim(1, 141)
    ax1.set_zlim(v_lower - 0.05, v_upper + 0.05)
    ax1.set_xticks(np.arange(1, 25, 4))
    ax1.set_yticks(np.arange(1, 142, 20))
    ax1.set_title('(a) Without any control method', fontsize=14, fontweight='bold', pad=15)

    # 添加水平参考平面
    xx1, yy1 = np.meshgrid(np.arange(1, 25, 1), np.arange(1, 142, 1))
    zz_upper1 = np.full_like(xx1, v_upper, dtype=float)
    zz_lower1 = np.full_like(xx1, v_lower, dtype=float)

    ax1.plot_surface(xx1, yy1, zz_upper1, alpha=0.2, color='red')
    ax1.plot_surface(xx1, yy1, zz_lower1, alpha=0.2, color='red')

    # 子图2: 采用所提控制方法
    ax2 = fig.add_subplot(122, projection='3d')
    X2, Y2 = np.meshgrid(hours, buses)
    Z2 = voltage_matrix_control.T

    surf2 = ax2.plot_surface(X2, Y2, Z2, cmap=cm.jet, alpha=0.8,
                             linewidth=0.5, antialiased=True,
                             vmin=v_lower - 0.05, vmax=v_upper + 0.05)

    ax2.view_init(elev=30, azim=135)
    ax2.set_xlabel('Time (hour)', fontsize=12, labelpad=10)
    ax2.set_ylabel('Bus', fontsize=12, labelpad=10)
    ax2.set_zlabel('Voltage (p.u.)', fontsize=12, labelpad=10)
    ax2.set_xlim(1, 24)
    ax2.set_ylim(1, 141)
    ax2.set_zlim(v_lower - 0.05, v_upper + 0.05)
    ax2.set_xticks(np.arange(1, 25, 4))
    ax2.set_yticks(np.arange(1, 142, 20))
    ax2.set_title('(b) With the proposed control method', fontsize=14, fontweight='bold', pad=15)

    # 添加水平参考平面
    xx2, yy2 = np.meshgrid(np.arange(1, 25, 1), np.arange(1, 142, 1))
    zz_upper2 = np.full_like(xx2, v_upper, dtype=float)
    zz_lower2 = np.full_like(xx2, v_lower, dtype=float)

    ax2.plot_surface(xx2, yy2, zz_upper2, alpha=0.2, color='red')
    ax2.plot_surface(xx2, yy2, zz_lower2, alpha=0.2, color='red')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(surf2, cax=cbar_ax)
    cbar.set_label('Voltage (p.u.)', fontsize=12)

    # 添加主标题
    fig.suptitle(f'24-hour bus voltage profiles in 141-bus network (Best Episode {best_day})',
                 fontsize=16, fontweight='bold', y=0.95)

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    save_path = os.path.join(save_dir, "voltage_3d_heatmaps_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"对比3D电压热力图已保存到: {save_path}")


def run_opf_daily(
        daily_pv_profile=None,
        pv_capacity_kw=2000,
        day_steps=1,
        n_train_days=2000,
        n_test_days=500,
        seed=None,
        log_dir="data/log/",
        image_dir="data/images/",
        # ========== 电压调节实验参数 ==========
        voltage_regulation_enabled=True,
        v_upper=1.05,
        v_lower=0.95,
        voltage_penalty_factor=10.0,
        multi_objective=True,
        voltage_weight=0.0,
        loss_weight=1,
        # ========== 收敛检测参数 ==========
        convergence_window=100,  # 滑动平均窗口大小
        convergence_threshold=0.95,  # 达到最终平均奖励的阈值比例
        # ====================================
):
    """
    运行考虑电压调节的DRL训练实验

    参数:
        voltage_regulation_enabled: 是否启用电压调节
        v_upper: 电压上限 (p.u.)
        v_lower: 电压下限 (p.u.)
        voltage_penalty_factor: 电压越限惩罚系数
        multi_objective: 是否使用多目标优化
        voltage_weight: 电压目标权重
        loss_weight: 网损目标权重
    """

    seed = seed or seeding.generate_seed()
    seeding.apply_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    expdir = os.path.join(log_dir, f"daily_opf_voltage_reg_{seed}")
    os.makedirs(expdir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    image_expdir = os.path.join(image_dir, f"daily_opf_voltage_reg_{seed}")
    os.makedirs(image_expdir, exist_ok=True)

    # 创建环境，传入电压调节参数
    env = OpfCase(
        daily_pv_profile=daily_pv_profile,
        pv_capacity_kw=pv_capacity_kw,
        steps_per_episode=day_steps,
        voltage_regulation_enabled=voltage_regulation_enabled,
        v_upper=v_upper,
        v_lower=v_lower,
        voltage_penalty_factor=voltage_penalty_factor,
        multi_objective=multi_objective,
        voltage_weight=voltage_weight,
        loss_weight=loss_weight
    )

    agent = Td31Step(env, name=f"DailyOPF_VoltageReg_{seed}", seed=seed, path=expdir)

    # 训练统计
    daily_train_losses = []
    daily_train_rewards = []
    daily_voltage_success_rates = []  # 电压成功率
    daily_voltage_violations = []  # 电压越限次数

    # 用于存储电压热力图数据
    voltage_matrix_no_control_best = None  # 无控制时最佳天的电压矩阵
    voltage_matrix_control_best = None  # 有控制时最佳天的电压矩阵
    best_day_for_heatmap = 1  # 最佳天编号

    best_loss = float("inf")
    best_loss_rate = float("inf")
    best_voltage_success_rate = 0.0  # 最佳电压成功率
    best_actions = None
    best_day_hourly_metrics = None
    train_day_hourly_metrics = []

    print("=" * 60)
    print("Starting DRL Training with Voltage Regulation")
    print(f"Voltage limits: [{v_lower}, {v_upper}] p.u.")
    print(f"Multi-objective: {multi_objective}")
    if multi_objective:
        print(f"  - Loss weight: {loss_weight}")
        print(f"  - Voltage weight: {voltage_weight}")
    print(f"Voltage penalty factor: {voltage_penalty_factor}")
    print("=" * 60)

    for day in range(1, n_train_days + 1):
        obs, info = env.reset()

        action_24h = agent.act(obs)  # shape: (24,)
        next_obs, hourly_rewards, total_reward, done, truncated, info = env.step(action_24h)
        hourly_info = info.get("hourly_info", [])

        daily_loss_with_storage = info["total_loss_with_storage"]
        daily_loss_rate_with_storage = info.get("daily_loss_rate_with_storage", np.nan)
        daily_train_losses.append(daily_loss_with_storage)
        daily_train_rewards.append(total_reward)

        # ========== 记录电压统计 ==========
        voltage_success_rate = info.get("voltage_success_rate", 100.0)
        daily_voltage_success_rates.append(voltage_success_rate)
        daily_voltage_violations.append(info.get("daily_voltage_violations", 0))
        # =================================

        # 更新最佳模型（综合考虑网损和电压成功率）
        is_better = False
        if multi_objective and voltage_regulation_enabled:
            # 多目标: 网损低且电压成功率高
            if (daily_loss_with_storage < best_loss * 1.05 and
                    voltage_success_rate > best_voltage_success_rate):
                is_better = True
        else:
            # 单目标: 仅考虑网损
            if daily_loss_with_storage < best_loss:
                is_better = True

        if is_better:
            best_loss = daily_loss_with_storage
            best_loss_rate = daily_loss_rate_with_storage
            best_voltage_success_rate = voltage_success_rate
            best_actions = action_24h.copy()
            best_day_hourly_metrics = hourly_info
            best_day_for_heatmap = day

            # 收集最佳天的电压数据用于热力图
            if voltage_regulation_enabled and hourly_info:
                # 收集有控制时的电压数据
                voltage_matrix_control = []
                for hour_info in hourly_info:
                    if hour_info.get("voltage_info"):
                        voltage_matrix_control.append(hour_info["voltage_info"]["all_voltages"])
                    else:
                        voltage_matrix_control.append([np.nan] * 141)
                voltage_matrix_control_best = np.array(voltage_matrix_control)  # 24x141

                # 为了比较，我们需要运行无控制的情况
                # 创建无控制环境
                env_no_control = OpfCase(
                    daily_pv_profile=daily_pv_profile,
                    pv_capacity_kw=pv_capacity_kw,
                    steps_per_episode=day_steps,
                    voltage_regulation_enabled=voltage_regulation_enabled,
                    v_upper=v_upper,
                    v_lower=v_lower,
                    voltage_penalty_factor=voltage_penalty_factor,
                    multi_objective=multi_objective,
                    voltage_weight=voltage_weight,
                    loss_weight=loss_weight
                )

                # 重置到相同状态
                obs_no_control, _ = env_no_control.reset()

                # 运行无控制动作（全零）
                action_24h_no_control = np.zeros(24)
                _, _, _, _, _, info_no_control = env_no_control.step(action_24h_no_control)

                # 收集无控制时的电压数据
                hourly_info_no_control = info_no_control.get("hourly_info", [])
                voltage_matrix_no_control = []
                for hour_info in hourly_info_no_control:
                    if hour_info.get("voltage_info"):
                        voltage_matrix_no_control.append(hour_info["voltage_info"]["all_voltages"])
                    else:
                        voltage_matrix_no_control.append([np.nan] * 141)
                voltage_matrix_no_control_best = np.array(voltage_matrix_no_control)  # 24x141

        agent.learn(obs, action_24h, total_reward, next_obs, done)

        if day % 10 == 0 or day == n_train_days:
            avg_success_rate = np.mean(daily_voltage_success_rates[-10:]) if daily_voltage_success_rates else 100.0
            print(
                f"  Trained {day}/{n_train_days} days, "
                f"Loss={daily_loss_with_storage:.3f}kW, "
                f"LossRate={daily_loss_rate_with_storage:.2f}%, "
                f"Reward={total_reward:.3f}, "
                f"V_Success={voltage_success_rate:.1f}%"
            )

    # ========== 训练结果可视化 ==========
    days = np.arange(1, n_train_days + 1)

    # 1. 网损曲线
    plt.figure(figsize=(18, 14))
    plt.plot(days, daily_train_losses, "-o", markersize=12, color=sns.color_palette()[0], linewidth=4)
    plt.xlabel("Training Episode", fontname='Times New Roman', fontsize=50)
    plt.ylabel("Daily Total Loss (kW)", fontname='Times New Roman', fontsize=50)
    plt.title("Training Process - Line Loss", fontname='Times New Roman', fontsize=50)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_expdir, "train_line_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 奖励曲线
    plt.figure(figsize=(18, 14))
    plt.plot(days, daily_train_rewards, "-^", markersize=12, color=sns.color_palette()[2], linewidth=4)
    plt.xlabel("Training Episode", fontname='Times New Roman', fontsize=50)
    plt.ylabel("Episode Reward", fontname='Times New Roman', fontsize=50)
    plt.title("Training Process - Episode Reward", fontname='Times New Roman', fontsize=50)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_expdir, "train_daily_reward.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 电压成功率曲线（新增）
    if voltage_regulation_enabled:
        plt.figure(figsize=(18, 14))
        plt.plot(days, daily_voltage_success_rates, "-s", markersize=12, color=sns.color_palette()[1], linewidth=4)
        plt.axhline(y=95, color='r', linestyle='--', linewidth=3, label='Target 95%')
        plt.xlabel("Training Episode", fontname='Times New Roman', fontsize=50)
        plt.ylabel("Voltage Success Rate (%)", fontname='Times New Roman', fontsize=50)
        plt.title("Training Process - Voltage Regulation Success Rate", fontname='Times New Roman', fontsize=50)
        plt.legend(prop={'family': 'Times New Roman', 'size': 32})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim([0, 105])
        plt.tight_layout()
        plt.savefig(os.path.join(image_expdir, "train_voltage_success_rate.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 电压越限次数曲线（新增）
        plt.figure(figsize=(18, 14))
        plt.plot(days, daily_voltage_violations, "-d", markersize=12, color=sns.color_palette()[3], linewidth=4)
        plt.xlabel("Training Episode", fontname='Times New Roman', fontsize=50)
        plt.ylabel("Daily Voltage Violations", fontname='Times New Roman', fontsize=50)
        plt.title("Training Process - Voltage Violations", fontname='Times New Roman', fontsize=50)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(image_expdir, "train_voltage_violations.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nTraining completed, total {n_train_days} days")
    print(f"Best day is day {best_day_for_heatmap}")
    print(f"Daily total loss with RL-PVES = {best_loss:.3f} kW")
    print(f"Daily total loss rate = {best_loss_rate:.2f}%")
    if voltage_regulation_enabled:
        print(f"Voltage success rate = {best_voltage_success_rate:.2f}%")
    # ========== 新增：平均每日最终奖励和收敛回合数 ==========
    # 计算平均每日最终奖励
    avg_daily_reward = np.mean(daily_train_rewards)
    print(f"\n=== Summary Statistics ===")
    print(f"Average daily reward over training: {avg_daily_reward:.4f}")

    # 收敛回合数检测（基于滑动平均）
    convergence_episode = None
    if len(daily_train_rewards) >= convergence_window:
        # 计算滑动平均
        moving_avg = np.convolve(daily_train_rewards, np.ones(convergence_window) / convergence_window, mode='valid')
        # 找到第一个滑动平均达到 avg_daily_reward * threshold 的回合
        target = avg_daily_reward * convergence_threshold
        convergence_indices = np.where(moving_avg >= target)[0]
        if len(convergence_indices) > 0:
            convergence_episode = convergence_indices[0] + convergence_window
            print(
                f"Convergence episode (first time {convergence_window}-episode moving avg reaches {convergence_threshold * 100:.0f}% of final avg): {convergence_episode}")
        else:
            print("No convergence detected within the threshold.")
    else:
        print(f"Not enough episodes to compute convergence (need at least {convergence_window}).")
    # =======================================================
    # ========== 生成3D电压热力图 ==========
    if voltage_regulation_enabled and voltage_matrix_no_control_best is not None and voltage_matrix_control_best is not None:
        print(f"\nGenerating 3D voltage heatmaps for best day (Day {best_day_for_heatmap})...")

        # 生成单独的热力图
        plot_3d_voltage_heatmap(
            voltage_matrix_no_control_best,
            "24-hour bus voltage profiles in 141-bus network (Without any control method)",
            os.path.join(image_expdir, "voltage_3d_no_control.png"),
            v_upper, v_lower
        )

        plot_3d_voltage_heatmap(
            voltage_matrix_control_best,
            "24-hour bus voltage profiles in 141-bus network (With the proposed control method)",
            os.path.join(image_expdir, "voltage_3d_with_control.png"),
            v_upper, v_lower
        )

        # 生成对比热力图
        plot_comparison_3d_voltage_heatmaps(
            voltage_matrix_no_control_best,
            voltage_matrix_control_best,
            best_day_for_heatmap,
            image_expdir,
            v_upper, v_lower
        )
    # ===================================

    # ========== 最佳性能日详细可视化 ==========
    if best_day_hourly_metrics:
        hours = [m["hour"] for m in best_day_hourly_metrics]
        line_loss_with_storage = [m["line_loss_with_storage"] for m in best_day_hourly_metrics]
        line_loss_no_storage = [m["line_loss_no_storage"] for m in best_day_hourly_metrics]
        pv_outputs = [m["pv_output"] for m in best_day_hourly_metrics]
        storage_powers = [m["storage_power"] for m in best_day_hourly_metrics]
        socs = [m["storage_soc"] for m in best_day_hourly_metrics]
        loss_rate_with_storage = [m["loss_rate_with_storage"] for m in best_day_hourly_metrics]
        loss_rate_no_storage = [m["loss_rate_no_storage"] for m in best_day_hourly_metrics]

        # 提取电压信息
        voltage_info_list = [m.get("voltage_info") for m in best_day_hourly_metrics]
        v_max_list = [v["v_max"] if v else 1.0 for v in voltage_info_list]
        v_min_list = [v["v_min"] if v else 1.0 for v in voltage_info_list]
        v_mean_list = [v["v_mean"] if v else 1.0 for v in voltage_info_list]

        valid_loss_rates_with = [r for r in loss_rate_with_storage if not np.isnan(r)]
        valid_loss_rates_without = [r for r in loss_rate_no_storage if not np.isnan(r)]
        avg_loss_rate_with_storage = np.mean(valid_loss_rates_with) if valid_loss_rates_with else np.nan
        avg_loss_rate_no_storage = np.mean(valid_loss_rates_without) if valid_loss_rates_without else np.nan

        def plot_line_loss_comparison(hours, line_loss_no_storage, line_loss_with_storage, best_day, save_dir):
            plt.figure(figsize=(16, 12))
            plt.plot(hours, line_loss_no_storage, 's--', color=sns.color_palette()[3],
                     label='Without ESS', markersize=12, linewidth=4)
            plt.plot(hours, line_loss_with_storage, 'o-', color=sns.color_palette()[0],
                     label='With RL-PVES', markersize=12, linewidth=4)
            plt.title(f"Hourly Line Loss Comparison - Best Episode Performance (Episode {best_day}) ",
                      fontname='Times New Roman', fontsize=35)
            plt.xlabel("Hour of Day", fontname='Times New Roman', fontsize=35)
            plt.ylabel("Line Loss (kW)", fontname='Times New Roman', fontsize=35)
            plt.xticks(hours, fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.legend(prop={'family': 'Times New Roman', 'size': 32})
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "line_loss_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()

        def plot_pv_output(hours, pv_outputs, save_dir):
            plt.figure(figsize=(16, 12))
            plt.plot(hours, pv_outputs, marker='o', color=sns.color_palette()[2], markersize=12, linewidth=4)
            plt.title("PV Output", fontname='Times New Roman', fontsize=35)
            plt.xlabel("Hour of Day", fontname='Times New Roman', fontsize=35)
            plt.ylabel("Power (kW)", fontname='Times New Roman', fontsize=35)
            plt.xticks(hours, fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max(pv_outputs) * 1.2)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "pv_output.png"), dpi=300, bbox_inches='tight')
            plt.close()

        def plot_storage_power(hours, storage_powers, save_dir):

            plt.figure(figsize=(16, 10))

            storage_powers = np.array(storage_powers)
            charging_powers = np.where(storage_powers < 0, storage_powers, 0)
            discharging_powers = np.where(storage_powers > 0, storage_powers, 0)

            plt.bar(hours, charging_powers, color=sns.color_palette()[3], alpha=0.7, label='Charging', width=0.6)
            plt.bar(hours, discharging_powers, color=sns.color_palette()[0], alpha=0.7, label='Discharging', width=0.6)

            plt.axhline(0, color='black', lw=2)
            plt.xlabel("Hour", fontname='Times New Roman', fontsize=35)
            plt.ylabel("Power (kW)", fontname='Times New Roman', fontsize=35)
            plt.title("Storage Power", fontname='Times New Roman', fontsize=35)
            plt.xticks(hours, fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.legend(prop={'family': 'Times New Roman', 'size': 32})
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "storage_power.png"), dpi=300, bbox_inches='tight')
            plt.close()

        def plot_storage_soc(hours, socs, save_dir):

            plt.figure(figsize=(16, 10))

            plt.plot(hours, socs, marker='o', color=sns.color_palette()[1], markersize=12, linewidth=4, label='SOC')
            plt.xlabel("Hour", fontname='Times New Roman', fontsize=35)
            plt.ylabel("SOC (%)", fontname='Times New Roman', fontsize=35)
            plt.title("Storage State of Charge", fontname='Times New Roman', fontsize=35)
            plt.xticks(hours, fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.ylim(0, 100)
            plt.legend(prop={'family': 'Times New Roman', 'size': 32})
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "storage_soc.png"), dpi=300, bbox_inches='tight')
            plt.close()

        def plot_loss_rate_comparison(hours, loss_rate_no_storage, loss_rate_with_storage, best_day, save_dir):
            plt.figure(figsize=(16, 12))

            valid_idx = [i for i in range(len(hours))
                         if not np.isnan(loss_rate_with_storage[i])
                         and not np.isnan(loss_rate_no_storage[i])]

            if valid_idx:
                valid_hours = [hours[i] for i in valid_idx]
                valid_rates_with = [loss_rate_with_storage[i] for i in valid_idx]
                valid_rates_without = [loss_rate_no_storage[i] for i in valid_idx]

                plt.plot(valid_hours, valid_rates_without, 's--', color=sns.color_palette()[3],
                         label='Without ESS', markersize=12, linewidth=4)
                plt.plot(valid_hours, valid_rates_with, 'o-', color=sns.color_palette()[0],
                         label='With RL-PVES', markersize=12, linewidth=4)

                improvements = [valid_rates_without[i] - valid_rates_with[i] for i in range(len(valid_hours))]
                max_improvement = max(improvements)
                max_idx = improvements.index(max_improvement)

                plt.scatter(valid_hours[max_idx], valid_rates_with[max_idx],
                            s=200, facecolors='none', edgecolors=sns.color_palette()[2],
                            label=f'Max Improvement ({max_improvement:.1f}%)')

            plt.title(f"Hourly Loss Rate Comparison - Best Episode Performance (Episode {best_day})",
                      fontname='Times New Roman',
                      fontsize=35)
            plt.xlabel("Hour", fontname='Times New Roman', fontsize=35)
            plt.ylabel("Line Loss Rate (%)", fontname='Times New Roman', fontsize=35)
            plt.xticks(range(1, 25), fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.legend(prop={'family': 'Times New Roman', 'size': 32})
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_rate_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()

        def plot_daily_loss_rate_comparison(avg_loss_rate_without, avg_loss_rate_with, best_day, save_dir):
            plt.figure(figsize=(16, 12))
            labels = ['Without ESS', 'With RL-PVES']
            values = [avg_loss_rate_without, avg_loss_rate_with]

            colors = [sns.color_palette()[3], sns.color_palette()[0]]
            bars = plt.bar(labels, values, color=colors)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}%', ha='center', va='bottom',
                         fontname='Times New Roman', fontsize=35)

            improvement = avg_loss_rate_without - avg_loss_rate_with
            if improvement > 0:
                plt.annotate(f'Improvement: {improvement:.2f}%',
                             xy=(1, avg_loss_rate_with),
                             xytext=(0.5, (avg_loss_rate_without + avg_loss_rate_with) / 2),
                             arrowprops=dict(arrowstyle='->', color=sns.color_palette()[2], lw=3.0),
                             fontname='Times New Roman', fontsize=35, ha='center')

            plt.title(f"Daily Average Loss Rate Comparison - -Best Episode Performance (Episode {best_day})",
                      fontname='Times New Roman',
                      fontsize=35)
            plt.ylabel("Line Loss Rate (%)", fontname='Times New Roman', fontsize=35)
            plt.xticks(fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "daily_loss_rate_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # ========== 新增：电压曲线可视化 ==========
        def plot_voltage_profile(hours, v_max_list, v_min_list, v_mean_list, v_upper, v_lower, best_day, save_dir):
            """绘制24小时电压曲线"""
            plt.figure(figsize=(16, 12))

            plt.plot(hours, v_max_list, '^-', color=sns.color_palette()[0],
                     markersize=10, linewidth=3, label='V_max')
            plt.plot(hours, v_mean_list, 'o-', color=sns.color_palette()[1],
                     markersize=10, linewidth=3, label='V_mean')
            plt.plot(hours, v_min_list, 'v-', color=sns.color_palette()[2],
                     markersize=10, linewidth=3, label='V_min')

            # 绘制约束边界
            plt.axhline(y=v_upper, color='r', linestyle='--', linewidth=3, label=f'Upper Limit ({v_upper})')
            plt.axhline(y=v_lower, color='r', linestyle='--', linewidth=3, label=f'Lower Limit ({v_lower})')
            plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Nominal (1.0)')

            # 填充越限区域
            plt.fill_between(hours, v_upper, max(v_max_list) * 1.01,
                             where=[v > v_upper for v in v_max_list],
                             alpha=0.3, color='red', label='Upper Violation')
            plt.fill_between(hours, min(v_min_list) * 0.99, v_lower,
                             where=[v < v_lower for v in v_min_list],
                             alpha=0.3, color='orange', label='Lower Violation')

            plt.title(f"24-Hour Voltage Profile - Best Episode (Episode {best_day})",
                      fontname='Times New Roman', fontsize=35)
            plt.xlabel("Hour of Day", fontname='Times New Roman', fontsize=35)
            plt.ylabel("Voltage (p.u.)", fontname='Times New Roman', fontsize=35)
            plt.xticks(hours, fontname='Times New Roman', fontsize=35)
            plt.yticks(fontname='Times New Roman', fontsize=35)
            plt.legend(prop={'family': 'Times New Roman', 'size': 28}, loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim([min(v_lower - 0.02, min(v_min_list) * 0.99),
                      max(v_upper + 0.02, max(v_max_list) * 1.01)])
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "voltage_profile_best_day.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # 执行所有可视化
        plot_line_loss_comparison(hours, line_loss_no_storage, line_loss_with_storage, best_day_for_heatmap,
                                  image_expdir)
        plot_pv_output(hours, pv_outputs, image_expdir)
        plot_storage_power(hours, storage_powers, image_expdir)
        plot_storage_soc(hours, socs, image_expdir)
        plot_loss_rate_comparison(hours, loss_rate_no_storage, loss_rate_with_storage, best_day_for_heatmap,
                                  image_expdir)
        plot_daily_loss_rate_comparison(avg_loss_rate_no_storage, avg_loss_rate_with_storage, best_day_for_heatmap,
                                        image_expdir)

        # 电压曲线可视化
        if voltage_regulation_enabled and voltage_info_list:
            plot_voltage_profile(hours, v_max_list, v_min_list, v_mean_list,
                                 v_upper, v_lower, best_day_for_heatmap, image_expdir)

    # ========== 测试阶段 ==========
    test_losses = []
    test_loss_rates = []
    test_hourly_metrics = []
    test_voltage_success_rates = []  # 测试电压成功率

    print("\n" + "=" * 60)
    print("Starting Testing Phase")
    print("=" * 60)

    for test_day in range(1, n_test_days + 1):
        obs, info = env.reset()

        action_24h = agent.act(obs)

        next_obs, hourly_rewards, total_reward, done, truncated, info = env.step(action_24h)

        test_losses.append(info["total_loss_with_storage"])
        test_loss_rates.append(info.get("daily_loss_rate_with_storage", np.nan))
        test_hourly_metrics.append(info["hourly_info"])
        test_voltage_success_rates.append(info.get("voltage_success_rate", 100.0))

    # 测试结果可视化
    tdays = np.arange(1, n_test_days + 1)

    plt.figure(figsize=(16, 12))
    plt.plot(tdays, test_losses, "-x", color=sns.color_palette()[0], markersize=12, linewidth=4)
    plt.xlabel("Test Episode", fontname='Times New Roman', fontsize=35)
    plt.ylabel("Daily Total Loss (kW)", fontname='Times New Roman', fontsize=35)
    plt.title("Test Phase - Line Loss", fontname='Times New Roman', fontsize=35)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_expdir, "test_line_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(16, 12))
    plt.plot(tdays, test_loss_rates, "-s", color=sns.color_palette()[1], markersize=12, linewidth=4)
    plt.xlabel("Test Days", fontname='Times New Roman', fontsize=35)
    plt.ylabel("Daily Loss Rate (%)", fontname='Times New Roman', fontsize=35)
    plt.title("Test Phase - Loss Rate", fontname='Times New Roman', fontsize=35)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_expdir, "test_loss_rate.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 测试电压成功率
    if voltage_regulation_enabled:
        plt.figure(figsize=(16, 12))
        plt.plot(tdays, test_voltage_success_rates, "-o", color=sns.color_palette()[2],
                 markersize=12, linewidth=4)
        plt.axhline(y=95, color='r', linestyle='--', linewidth=3, label='Target 95%')
        plt.xlabel("Test Episode", fontname='Times New Roman', fontsize=35)
        plt.ylabel("Voltage Success Rate (%)", fontname='Times New Roman', fontsize=35)
        plt.title("Test Phase - Voltage Success Rate", fontname='Times New Roman', fontsize=35)
        plt.legend(prop={'family': 'Times New Roman', 'size': 32})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim([0, 105])
        plt.tight_layout()
        plt.savefig(os.path.join(image_expdir, "test_voltage_success_rate.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 统计结果
    avg_test_loss = np.mean(test_losses)
    avg_test_loss_rate = np.mean([r for r in test_loss_rates if not np.isnan(r)])
    avg_test_voltage_success = np.mean(test_voltage_success_rates) if voltage_regulation_enabled else 100.0

    print(f"\n{'=' * 60}")
    print(f"Test results for {n_test_days} days:")
    print(f"Average 24-hour loss = {avg_test_loss:.4f} kW")
    print(f"Average total loss rate = {avg_test_loss_rate:.2f}%")
    if voltage_regulation_enabled:
        print(f"Average voltage success rate = {avg_test_voltage_success:.2f}%")
    print(f"{'=' * 60}")

    # 测试日小时级损耗率可视化
    if test_hourly_metrics:
        plt.figure(figsize=(16, 10))
        colors = sns.color_palette("husl", min(len(test_hourly_metrics), 10))

        for day_idx, day_metrics in enumerate(test_hourly_metrics[:10]):  # 最多显示10天
            hours = [m["hour"] for m in day_metrics]
            loss_rates = [m["loss_rate_with_storage"] for m in day_metrics]

            valid_hours = [h for h, r in zip(hours, loss_rates) if not np.isnan(r)]
            valid_rates = [r for r in loss_rates if not np.isnan(r)]

            plt.plot(valid_hours, valid_rates, marker='o', markersize=8,
                     color=colors[day_idx], linewidth=3, label=f"Test Day {day_idx + 1}")

        plt.xlabel("Hour", fontname='Times New Roman', fontsize=25)
        plt.ylabel("Loss Rate (%)", fontname='Times New Roman', fontsize=25)
        plt.title("Hourly Loss Rate on Test Days", fontname='Times New Roman', fontsize=25)
        plt.legend(prop={'family': 'Times New Roman', 'size': 22})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(1, 25), fontname='Times New Roman', fontsize=25)
        plt.yticks(fontname='Times New Roman', fontsize=25)

        plt.tight_layout()
        plt.savefig(os.path.join(image_expdir, "test_hourly_loss_rates.png"), dpi=300, bbox_inches='tight')
        plt.close()

    agent.plot_training_curve()
    # 保存训练数据用于合并绘图
    np.save(os.path.join(image_dir, "train_rewards_TD3.npy"), daily_train_rewards)
    np.save(os.path.join(image_dir, "train_line_loss_TD3.npy"), daily_train_losses)
    np.save(os.path.join(image_dir, "train_voltage_violations_TD3.npy"), daily_voltage_violations)
    return {
        "train_losses": daily_train_losses,
        "train_rewards": daily_train_rewards,
        "avg_daily_reward": avg_daily_reward,
        "convergence_episode": convergence_episode,
        "best_actions": best_actions,
        "best_loss_rate": best_loss_rate,
        "best_voltage_success_rate": best_voltage_success_rate,
        "test_losses": test_losses,
        "test_loss_rates": test_loss_rates,
        "avg_test_loss": avg_test_loss,
        "avg_test_loss_rate": avg_test_loss_rate,
        "test_voltage_success_rates": test_voltage_success_rates,
        "avg_test_voltage_success": avg_test_voltage_success,
        "best_day_hourly_metrics": best_day_hourly_metrics,
    }


if __name__ == "__main__":
    image_dir = "data/images/"
    os.makedirs(image_dir, exist_ok=True)

    # 运行带电压调节的实验
    results = run_opf_daily(
        image_dir=image_dir,
        voltage_regulation_enabled=True,  # 启用电压调节
        v_upper=1.05,  # 电压上限
        v_lower=0.95,  # 电压下限
        voltage_penalty_factor=10.0,  # 越限惩罚系数
        multi_objective=True,  # 多目标优化
        voltage_weight=0,  # 电压目标权重
        loss_weight=1,  # 网损目标权重
    )
    # 打印新增的统计信息（可选）
    print(f"\nFinal Summary from returned dictionary:")
    print(f"Average daily reward: {results['avg_daily_reward']:.4f}")
    if results['convergence_episode']:
        print(f"Convergence episode: {results['convergence_episode']}")
    else:
        print("Convergence episode: Not detected")