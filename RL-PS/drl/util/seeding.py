import random
import os

import gymnasium
import numpy as np
import torch


def generate_seed():
    return int.from_bytes(os.urandom(3), byteorder='big')


def apply_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_env(env, seed):
    env.np_random = gymnasium.utils.seeding.np_random(seed)[0]
    env.action_space.seed(seed)
    env.observation_space.seed(seed)