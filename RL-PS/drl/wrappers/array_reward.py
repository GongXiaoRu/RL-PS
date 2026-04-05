from abc import abstractmethod

import gymnasium as gym
import numpy as np


class ArrayReward(gym.Wrapper):
    """ Replaces scalar reward with an array of rewards. Works for 
    MuJoCo environments."""

    def __init__(self, env):
        super().__init__(env)
        env.reset()
        info = env.step(env.action_space.sample())[4]
        n_rewards = len(self.get_reward_array(info))
        try: 
            low = env.reward_space.low[0]
            high = env.reward_space.high[0]
        except AttributeError:
            low = -np.inf
            high = np.inf
        
        self.reward_space = gym.spaces.Box(
            low=np.array([low] * n_rewards),
            high=np.array([high] * n_rewards),
            dtype=np.float32)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['scalar_reward'] = reward
        reward = self.get_reward_array(info)
        return obs, reward, terminated, truncated, info

    @abstractmethod
    def get_reward_array(self, info):
        return np.array([v for k, v in info.items() if 'reward_' in k])
