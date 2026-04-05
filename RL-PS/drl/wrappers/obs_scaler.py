""" Wrapper to automatically scale the observation space of an environment. """

import logging

import gymnasium as gym
import numpy as np

class ObsScaler(gym.ObservationWrapper):
    def __init__(self, env, low=-1, high=1):
        super().__init__(env)
        self.old_low = self.observation_space.low
        self.old_high = self.observation_space.high
        self.new_low = low
        self.new_high = high
        self.eps = 1e-2
        if np.inf in self.old_high or -np.inf in self.old_low:
            logging.warning('inf in observation space -> scaling impossible')
            # TODO: At least scale the other dimensions that are not inf
            self.observation_space = env.observation_space
            self.observation = self.do_nothing
        else:
            self.observation_space = gym.spaces.Box(
                low, high, shape=env.observation_space.shape)

    def observation(self, obs):
        obs = ((self.new_high - self.new_low) * (obs - self.old_low) / 
               (self.old_high - self.old_low) + self.new_low)
        if (((obs + self.eps) < self.new_low).any() 
                or ((obs - self.eps) > self.new_high).any()):
            logging.warning('Observation out of bounds after scaling.')
            logging.warning(f'Min entry {min(obs)} and max {max(obs)}')
        return obs
    
    def do_nothing(self, obs):
        return obs
