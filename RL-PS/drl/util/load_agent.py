

import drl.experiment


def load_agent(path, seed=None):
    if path[-1] != '/':
        path += '/'

    with open(path + '/meta-data.txt') as f:
        lines = f.readlines()
    env_name = lines[1].split(' ')[1][:-1]
    algo = lines[2][15:][:-1]
    hyperparams = drl.experiment.str_to_dict(lines[6][23:][:-1])
    env_hyperparams = drl.experiment.str_to_dict(lines[7][25:][:-1])

    env = drl.experiment.create_environment(env_name, env_hyperparams, seed)

    agent_class = drl.experiment.get_agent_class(algo)
    name = algo + '_' + str(hyperparams) + '_' + str(env_hyperparams)

    agent = agent_class(
        env, name=name, seed=seed, path=path, **hyperparams)
    agent.load_model()

    return agent
