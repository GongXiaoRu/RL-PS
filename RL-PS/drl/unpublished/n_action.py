import copy
from collections.abc import Iterable
import random
import time

from drl.agent import DrlAgent
from drl.ddpg import Ddpg, Td3
from drl.sac import Sac
from drl.util.seeding import seed_env


class NActionMixin(DrlAgent):
    """ Idea: Apply every action n times to the same obs because the 
    environment is computationally expensive to apply new actions. """
    def __init__(self, env, act_interval=10, **kwargs):
        self.act_interval = act_interval
        super().__init__(env, **kwargs)

    def run(self, n_steps, test_interval=999999, test_steps=10, 
            schedule_hps=None):
        next_test = test_interval

        if isinstance(schedule_hps, Iterable):
            schedule_hps = list(schedule_hps)
            # Sort by step in ascending order so that first entry is next entry
            schedule_hps.sort(key=lambda x: x[0])

        start_step = self.step
        self.start_time = time.time()
        self.n_train_steps = n_steps

        self.envs = [copy.deepcopy(self.env) for _ in range(self.n_envs)]
        [seed_env(env, random.randint(0, 100000)) for env in self.envs]

        dones = [True] * self.n_envs
        obss = [None] * self.n_envs
        states = [None] * self.n_envs
        total_rewards = [None] * self.n_envs
        while True:
            for idx, env in enumerate(self.envs):
                # TODO: Create extra function/methods for these MarlAgent special stuff
                if dones[idx] is True:
                    # Environment is done -> reset & collect episode data
                    if total_rewards[idx] is not None:
                        self.evaluator.step(total_rewards[idx], self.step)
                    t = time.time()
                    new_action = self.step % self.act_interval == 0
                    obss[idx], info = self.envs[idx].reset(options={'new_action': new_action})
                    self.env_time += time.time() - t
                    try:
                        states[idx] = self.envs[idx].state()
                    except TypeError:
                        states[idx] = self.envs[idx].state
                    except AttributeError:
                        states[idx] = None

                    total_rewards[idx] = 0
                    dones[idx] = False

                while len(schedule_hps) > 0 and schedule_hps[0][0] == self.step:
                    # Apply hyperparameter change
                    step, hp_name, hp_value = schedule_hps.pop(0)
                    setattr(self, hp_name, hp_value)

                if self.step % self.act_interval == 0:
                    act = self.act(obss[idx])

                t = time.time()
                next_obs, reward, terminal, truncated, info = self.envs[idx].step(act)
                self.env_time += time.time() - t

                try:
                    next_state = self.envs[idx].state()
                except TypeError:
                    next_state = self.envs[idx].state
                except AttributeError:
                    next_state = None

                t = time.time()
                self.learn(
                    obss[idx], act, reward * self.reward_scaling, next_obs,
                    terminal, states[idx], next_state, info, env_idx=idx)
                self.train_time += time.time() - t

                obss[idx] = next_obs
                states[idx] = next_state
                dones[idx] = terminal or truncated

                total_rewards[idx] += reward

            self.step += 1
            if self.step >= next_test:
                self.test(test_steps=test_steps)
                next_test = self.step + test_interval

            if self.step - start_step >= self.n_train_steps:
                # TODO: Store results of testing instead of training
                self.evaluator.plot_reward()
                return self.evaluator    
            

class NActionDdpg(NActionMixin, Ddpg):
    pass


class NActionTd3(NActionMixin, Td3):
    pass


class NActionSac(NActionMixin, Sac):
    pass