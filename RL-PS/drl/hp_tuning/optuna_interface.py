import copy
import json
import os

import numpy as np
import optuna

from drl.experiment import single_experiment, argparser


def create_objective(agent_name, args, hp_sampling_method=None, 
                     env_hp_sampling_method=None, metric='average_return',
                     last_n_steps=1, n_seeds=3, mean=False, multi_objective=False,
                     **kwargs):

    if multi_objective:
        # Using the median for multi-objective optimization does not make sense
        mean = True

    args['agent_hp'] = args['hyperparams'][0]
    args['env_hp'] = args['env_hyperparams'][0]

    # Create objective function and return it
    def objective(trial):

        args_local = copy.deepcopy(args)
        args_local['path'] = os.path.join(args['path'], f'run_{trial.number}')

        # Sample and update hyperparameters
        if hp_sampling_method:
            hps = hp_sampling_method(trial)
            args_local['agent_hp'].update(hps)
        if env_hp_sampling_method:
            hps = env_hp_sampling_method(trial)
            args_local['env_hp'].update(hps)

        if multi_objective:
            pruning_conditions = []
        else:
            # Create early stopping condition for the agent training
            def test_for_pruning(metrics):
                # TODO: Currently only for custom metric
                metric_value = metric(metrics[-last_n_steps:])
                if hasattr(args_local['test_interval'], '__len__'):
                    step = args_local['test_interval'][len(metrics) - 1]
                elif type(args_local['test_interval']) == int:
                    step = args_local['test_interval'] * len(metrics)
                trial.report(metric_value, step)
                if trial.should_prune():
                    return True
                return False
            pruning_conditions = [test_for_pruning]

        # Train the agent n times to achieve more robust performance estimation
        all_metrics = []
        for i in range(n_seeds):
            # Train the model and return the evaluation metric
            eval_metrics, pruned = single_experiment(agent_name=agent_name, 
                                             early_stopping_conditions=pruning_conditions,
                                             cli_args=args_local, **args_local)

            eval_metrics = eval_metrics[-last_n_steps:]
            # Either standard metric or custom metric
            if type(metric) == str:
                # Average over last n steps
                nth_metric_value = np.mean([m[metric] for m in eval_metrics])
            else:
                nth_metric_value = metric(eval_metrics)

            # Mean over multiple runs
            all_metrics.append(nth_metric_value)
            if mean:
                metric_value = np.mean(all_metrics, axis=0)
            else:
                metric_value = np.median(all_metrics, axis=0)

            # Additionally check for pruning after each training run
            # Essentially: Does it make sense to start a new seed?
            if not multi_objective:
                trial.report(metric_value, args_local['steps_to_train'])
                if pruned or (trial.should_prune()) and (i + 1 != n_seeds):
                    raise optuna.TrialPruned()

        if hasattr(metric_value, '__len__'):
            # Multi-objective optimization
            print('metric =', tuple(metric_value))
            return tuple(metric_value)
        else:
            return metric_value

    return objective


def main(create_custom_objective=None, hp_sampling_method=None, 
         env_hp_sampling_method=None, metric='average_return', 
         storage=None, sampler=None, pruner=None, study_name=None,
         direction=None, load_if_exists=False, directions=None,
         **kwargs):
    # try:
    #     with open('search_space.json') as f:
    #         search_space = json.load(f)
    # except FileNotFoundError:
    #     search_space = None

    # try:
    #     with open('env_search_space.json') as f:
    #         env_search_space = json.load(f)
    # except FileNotFoundError:
    #     env_search_space = None

    args = argparser()

    multi_objective = bool(directions)

    if not study_name:
        study_name = args['path'].split('/')[-1]
    study_name = os.path.join(args['path'], study_name)
    if storage is True:
        storage = f'sqlite:///{os.path.join(args["path"], "storage.db")}'
        print(f'Store to {storage}')
        # Ensure that the directory exists (Otherwise db creation fails)
        os.makedirs(os.path.join(args['path']), exist_ok=True)

    # Sample agent hyperparameters
    for agent_name in args['agent_classes']:
        # random_hyperparameters = agent_sampling_method(search_space, args['hyperparams'][0])
        # args['agent_hp'] = random_hyperparameters
        
        # # Sample environment hyperparameters
        # random_hyperparameters = env_sampling_method(env_search_space, args['env_hyperparams'][0])
        # args['env_hp'] = random_hyperparameters
        
        if not create_custom_objective:
            objective = create_objective(agent_name, args, hp_sampling_method,
                                         env_hp_sampling_method, metric,
                                         multi_objective=multi_objective,
                                         **kwargs)

        else:
            objective = create_custom_objective(agent_name, args, hp_sampling_method,
                                                env_hp_sampling_method, metric,
                                                **kwargs)

        # Create a study and optimize
        study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name, direction=direction, load_if_exists=load_if_exists, directions=directions)
        remaining_trials = args['num_experiments']
        while remaining_trials > 0:
            n_trials = min(remaining_trials, 1)
            study.optimize(objective, n_trials=n_trials)
            remaining_trials -= n_trials

            # Store intermediate results
            print('----------------------------------------------')
            try:
                print('Best hyperparameters:', study.best_params)
                print('Best score:', study.best_value)
                # Store to JSON file 
                path = os.path.join(args['path'], 'best_hyperparameters.json')
                with open(path, 'w') as f:
                    json.dump(study.best_params, f, indent=4)
            except RuntimeError:
                # Multi-objective optimization instead
                print('Best trials:', study.best_trials)

            # Store to CSV file
            trials_df = study.trials_dataframe()
            trials_df.to_csv(os.path.join(args['path'], 'trials.csv'))


if __name__ == '__main__':
    main(sampler=optuna.samplers.GPSampler(),
            env_hp_sampling_method=opf_env_design_sampling,
            metric=custom_single_obj_metric,
            # directions=["minimize", "minimize"],
            direction='minimize',
            n_seeds=3)
