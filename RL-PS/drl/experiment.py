#!/usr/bin/env python3
"""
Simple way of running experiments to compare different implementations and
or hyperparameter against each other. The reward curve gets plotted in the
end.

"""

import argparse
import ast
import collections
from datetime import datetime
import importlib
import json
import os
import random
import time

import gymnasium
import numpy as np

from drl.util import seeding

from drl.ddpg import *
from drl.reinforce import *
from drl.dqn import *
# from drl.maddpg import *
from drl.sac import *

from drl.unpublished.ddpg1step import *
from drl.unpublished.ensemble_ddpg import *


def run_experiment(args):
    # args = argparser()
    experiment(cli_args=args, **args)


def argparser(arg_dic):
    argparser = arg_dic
    argparser.add_argument(
        '--environment-name',
        type=str,
        default='CartPole-v1',
        help="Either provide the name of the gymnasium environment class or "
        "provide import path to a custom creator function that generates your "
        "environment. For example: 'path.to:function_def'"
    )
    argparser.add_argument(
        '--num-experiments',
        type=int,
        default=1,
        help="Repeat experiment how many times with different seeds?"
    )
    argparser.add_argument(
        '--steps-to-train',
        type=int,
        default=1e5,
        help="Number of training steps per agent"
    )
    argparser.add_argument(
        '--agent-classes',
        help="Give a list of agent class names as string separated by comma.",
        type=str,
        default='Reinforce'
    )
    argparser.add_argument(
        '--hyperparams',
        help="Give a list of hyperparams settings as dicts separated by ';',"
        """for example "{'n_envs': 1}; {'n_envs': 2}" """,
        type=str,
        default=''
    )
    argparser.add_argument(
        '--schedule-hps',
        help="Provide a schedule for hyperparameters as a tuples of tuples in the form: ((step, hp_parameter_name, hp_value),...,last_tuple)",
        type=str,
        default='()'
    )
    argparser.add_argument(
        '--env-hyperparams',
        help="Give a dict of hyperparams settings ,"
        """for example "{'load_scaling': 2.0}" """,
        type=str,
        default=''
    )
    argparser.add_argument(
        '--wrappers',
        help="Give a list of wrapper (class_name, args) tuples as string separated by comma. The wrappers are applied in the order they are given.",
        type=str,
        default='()'
    )
    argparser.add_argument(
        '--store-results',
        help="Store the results in a separate folder?",
        action='store_true'
    )
    argparser.add_argument(
        '--seed',
        help="Which seed to use to make experiments reproducible?",
        type=int
    )
    argparser.add_argument(
        '--test-interval',
        help="Perform tests at which interval? Provide either an integer (e.g. 1000) for the interval or an iterable of integers (e.g. '[2000, 3000]') to define precisely at which steps to test.",
        default=9999999
    )
    argparser.add_argument(
        '--test-steps',
        help="How many test steps per test?",
        type=int,
        default=1000
    )
    argparser.add_argument(
        '--test-episodes',
        help="How many test episodes per test?",
        type=int,
        default=None
    )
    argparser.add_argument(
        '--path',
        help="Store results to which path?",
        type=str,
        default='data/'
    )
    argparser.add_argument(
        '--comment',
        help="Comment to describe your experiment (stored in meta-data.txt)",
        type=str,
        default='No comment'
    )

    args = argparser.parse_args()
    args_dict = vars(args)

    # Convert string to list of agent classes
    agent_names = args.agent_classes
    if '[' in agent_names and ']' in agent_names:
        agent_names = agent_names[1:-1]
    agent_names = agent_names.replace(" ", "").replace("\t", "")
    args_dict['agent_classes'] = agent_names.split(',')

    # Convert hyperparameters to list of dicts
    args_dict['hyperparams'] = str_to_list_of_dicts(args.hyperparams)
    args_dict['env_hyperparams'] = str_to_list_of_dicts(args.env_hyperparams)

    args_dict['wrappers'] = eval(args.wrappers)
    args_dict['schedule_hps'] = eval(args.schedule_hps)
    args_dict['test_interval'] = eval(args.test_interval)

    return args_dict


def experiment(
        agent_classes: list,
        environment_name: str,
        num_experiments: int,
        steps_to_train: int,
        store_results: bool,
        path: str,
        seed: int,
        hyperparams: list=[{}],
        schedule_hps: tuple=(),
        env_hyperparams: list=[{}],
        wrappers: tuple=(),
        test_interval=99999999,
        test_steps=10,
        test_episodes=None,
        comment='',
        cli_args=None):

    if not cli_args:
        cli_args = {}

    for agent_name in agent_classes:
        for agent_hp in hyperparams:
            print('Start experiments with agent class: ', agent_name)
            for env_hp in env_hyperparams:
                for _ in range(num_experiments):
                    single_experiment(seed, agent_name, environment_name,
                                      agent_hp, env_hp, wrappers, store_results,
                                      path, steps_to_train, test_interval,
                                      test_steps, test_episodes, comment,
                                      schedule_hps, cli_args)


def single_experiment(seed, agent_name, environment_name, agent_hp, env_hp,
                      wrappers, store_results, path, steps_to_train,
                      test_interval, test_steps, test_episodes, comment,
                      schedule_hps, cli_args,
                      early_stopping_conditions: list=None,
                      **kwargs):
    if not seed:
        local_seed = seeding.generate_seed()
    else:
        # TODO: Currently always same seed is used. Useless for multiple experiments.
        local_seed = seed
    seeding.apply_seed(local_seed)

    env = create_environment(environment_name, env_hp, seed=seed)
    env = apply_wrappers(env, wrappers)

    short_env_name = environment_name.split(':')[-1]
    short_agent_name = agent_name.split(':')[-1]
    name = (f'{short_env_name}_{short_agent_name}')

    if store_results:
        path = create_experiment_folder(
            name, environment_name, agent_name, seed, agent_hp,
            env_hp, path, steps_to_train, test_interval, comment, cli_args)
    else:
        path = 'temp/'

    print('Start experiment: ', name)
    print(f'Seed is: {local_seed}')
    agent_class = get_agent_class(agent_name)
    # agent = agent_class(
    #     env, name=name, path=path, seed=seed, schedule_hps=schedule_hps,
    #     **agent_hp)
    agent = Ddpg1Step(env, name=name, path=path, seed=seed, schedule_hps=schedule_hps,
        **agent_hp)

    if isinstance(test_interval, collections.abc.Iterable):
        test_interval = iter(sorted(test_interval))

    evaluation_metrics = []
    while True:
        if type(test_interval) == int:
            n_steps = test_interval
        elif isinstance(test_interval, collections.abc.Iterator):
            try:
                n_steps = next(test_interval) - agent.step
                assert n_steps > 0
            except StopIteration:
                # No further tests required
                n_steps = np.inf

        agent.train(n_steps=min(n_steps, steps_to_train-agent.step))
        evaluation_metrics.append(agent.test(test_steps=test_steps, test_episodes=test_episodes))

        if agent.step >= steps_to_train:
            print('Training finished!')
            return evaluation_metrics, False

        if early_stopping_conditions:
            for condition in early_stopping_conditions:
                if condition(evaluation_metrics):
                    print(f'Early stopping condition met at step {agent.step}!')
                    return evaluation_metrics, True


def get_agent_class(agent_name):
    if ':' not in agent_name:
        return eval(agent_name)
    else:
        module_name, function_name = agent_name.split(':')
        module = importlib.import_module(module_name)
        return getattr(module, function_name)


def create_environment(env_name, env_hyperparams: dict, seed=None):
    if ':' not in env_name:
        env = gymnasium.make(env_name)
        seeding.seed_env(env, seed)
        return env
    elif 'pettingzoo' in env_name:
        module_name = env_name.replace(':', '.')
        module = importlib.import_module(module_name)
        return module.env(seed=seed, **env_hyperparams)
    else:
        module_name, class_name = env_name.split(':')
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(**env_hyperparams)


def apply_wrappers(env, wrappers: tuple):
    for wrapper_class, args in wrappers:
        if ':' in wrapper_class:
            module_name, wrapper_class = wrapper_class.split(':')
        else:
            module_name = 'gymnasium.wrappers'
        module = importlib.import_module(module_name)
        env = getattr(module, wrapper_class)(env, **args)
    return env


def str_to_list_of_dicts(string):
    # TODO: This is far to complicated. Simply use eval!
    if string:
        if '[' in string and ']' in string:
            string = string[1:-1]
        # Make string better splitable
        # TODO: Add to documentation
        string = string.replace('},{', '};{')
        list_of_strings = string.split(';')
        list_of_dicts = [str_to_dict(hp) for hp in list_of_strings]
    else:
        list_of_dicts = [dict()]

    return list_of_dicts


def str_to_dict(string):
    string = string.replace(" {", "{").replace("\t", " ")
    dictionary = ast.literal_eval(string)
    return dictionary


def create_experiment_folder(name, env_name, agent_name, seed, hyperparams,
                             env_hyperparams, path_dir, n_steps, test_interval,
                             comment, args:dict):
    for _ in range(10):
        # Create path folder
        now = datetime.now().isoformat(timespec='seconds')
        full_name = f'{now}_{name}'
        path = os.path.join(path_dir, full_name.replace(':', '.'), '')
        # Make sure that folder does not already exist
        try:
            os.makedirs(os.path.dirname(path))
            break
        except FileExistsError:
            print('folder already exists')
            time.sleep(random.random() * 3)

    # Store args in JSON file in path to remember how experiment was created
    with open(os.path.join(path, 'console_args.json'), 'w') as f:
        json.dump(args, f, indent=4)

    # Store hyperparams in JSON file in path
    with open(os.path.join(path, 'agent_hyperparams.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)

    with open(os.path.join(path, 'env_hyperparams.json'), 'w') as f:
        json.dump(env_hyperparams, f, indent=4)

    # Store some meta-data
    with open(os.path.join(path, 'meta-data.txt'), 'w') as f:
        f.write(f'Description: {name}\n')
        f.write(f'Environment: {env_name}\n')
        f.write(f'DRL Algorithm: {agent_name}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Training steps: {n_steps}\n')
        f.write(f'DRL-agent hyperparams: {hyperparams}\n')
        f.write(f'Environment hyperparams: {env_hyperparams}\n')
        f.write(f'Comment: {comment}')

    return path
#
#
# if __name__ == '__main__':
#     main()
