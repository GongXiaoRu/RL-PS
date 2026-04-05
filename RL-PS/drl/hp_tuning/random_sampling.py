import json 
import random

from drl.experiment import single_experiment, argparser


def random_sampling(search_space, fixed_hyperparameters):
    if not fixed_hyperparameters:
        fixed_hyperparameters = dict()
    if not search_space:
        return fixed_hyperparameters

    hyperparameters = fixed_hyperparameters.copy()

    for hp_name, range in search_space['discrete'].items():
        hyperparameters[hp_name] = random.choice(range)

    for hp_name, range in search_space['continuous'].items():
        hyperparameters[hp_name] = random.uniform(*range)

    return hyperparameters


def main(agent_sampling_method=random_sampling, env_sampling_method=random_sampling):
    with open('search_space.json') as f:
        search_space = json.load(f)

    with open('env_search_space.json') as f:
        env_search_space = json.load(f)

    args = argparser()

    for _ in range(args['num_experiments']):
        # Sample agent hyperparameters
        for agent_name in args['agent_classes']:
            random_hyperparameters = agent_sampling_method(search_space, args['hyperparams'][0])
            args['agent_hp'] = random_hyperparameters
            
            # Sample environment hyperparameters
            random_hyperparameters = env_sampling_method(env_search_space, args['env_hyperparams'][0])
            args['env_hp'] = random_hyperparameters
            
            single_experiment(agent_name=agent_name, cli_args=args, **args)


if __name__ == '__main__':
    main()
