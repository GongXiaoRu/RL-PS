

from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd

# TODO: Split into functions
# TODO: Create text file with all results
# TODO: Add std (other mean values almost meaningless)


def main(path: str, which_hyperparam: str, metric: str='average_return', 
         last_n_steps: int=3, discrete_hp_steps: int=5, 
         top_n_percentile: int=None, bottom_n_percentile: int=None,):

    with open('search_space.json') as f:
        search_space = json.load(f)

    with open('env_search_space.json') as f:
        env_search_space = json.load(f)

    search_space_discrete = {**search_space['discrete'], **env_search_space['discrete']}
    search_space_continuous = {**search_space['continuous'], **env_search_space['continuous']}

    expected_means = {key: np.mean(value) for key, value in search_space_continuous.items()}
    expected_means.update({key: np.mean(value) for key, value in search_space_discrete.items()})

    # Collect all experiment runs with their hyperparameters and results
    result_list = []
    hps_list = []
    for run_path in os.listdir(path):
        full_path = os.path.join(path, run_path)

        with open(os.path.join(full_path, 'agent_hyperparams.json')) as f:
            hyperparams = json.load(f)

        with open(os.path.join(full_path, 'env_hyperparams.json')) as f:
            env_hyperparams = json.load(f)

        try:
            results = pd.read_csv(
                os.path.join(full_path, 'test_returns.csv'), index_col=0)
        except FileNotFoundError:
            continue
        
        # Compute metrics over last n steps to make sure we don't have outlier in the end
        end_results = dict(results.iloc[-last_n_steps:].mean())

        # TODO: Maybe also compute mean over all steps to consider convergence speed (also most robust to outliers)

        hps_list.append({**hyperparams, **env_hyperparams})
        result_list.append(end_results)

    result_df = pd.DataFrame(result_list)
    hp_df = pd.DataFrame(hps_list)
    
    n_runs = len(result_list)

    # Filter out results that are in the top or bottom n percentile
    result_df = result_df.sort_values(by=metric, ascending=False)
    if top_n_percentile:
        result_df = result_df[:int(len(result_df) * top_n_percentile / 100)]
    elif bottom_n_percentile:
        result_df = result_df[int(len(result_df) * top_n_percentile / 100):]
    hp_df = hp_df.iloc[result_df.index]

    if which_hyperparam == 'optimal' and not (top_n_percentile or bottom_n_percentile):
        determine_optimal(result_df, hp_df, metric, min_max='max')
        print('')
        determine_optimal(result_df, hp_df, metric, min_max='min')
            
    elif which_hyperparam in search_space_discrete:
        xys = []
        # Evaluate the results wrt to a single metric and a single hyperparameter
        one_hp_result = defaultdict(list)
        for hp, value in zip(hp_df[which_hyperparam], result_df[metric]):
            one_hp_result[str(hp)].append(value)
        
        for key, value in one_hp_result.items():
            print(f'{which_hyperparam}={key}:', round(sum(value) / len(value), 4), metric, f'({len(value)}/{n_runs} runs)')
            xys.append((f'{key} ({len(value)}x)', value))

        try:    
            xys = sorted(xys, key=lambda x: int(x[0].split('(')[0]), reverse=False)
        except ValueError:
            xys = sorted(xys, key=lambda x: x[0], reverse=False)
        boxplots(*zip(*xys), x_label=which_hyperparam, y_label=metric)

    elif which_hyperparam in search_space_continuous:
        # First sort the continuous hyperparameter values into n bins
        data = pd.DataFrame({'metric': result_df[metric], which_hyperparam: hp_df[which_hyperparam]})

        data['bins'] = pd.cut(data[which_hyperparam], bins=discrete_hp_steps, 
                              labels=list(range(discrete_hp_steps)))

        full_range = search_space_continuous[which_hyperparam]
        xys = []
        for bin in set(data['bins']):
            bin_data = data[data['bins'] == bin]
            
            bin_range_min = round((full_range[1] - full_range[0]) / discrete_hp_steps * bin + full_range[0], 4)
            bin_range_max = round(bin_range_min + (full_range[1] - full_range[0]) / discrete_hp_steps, 4)
            print(f'{which_hyperparam}={bin_range_min}-{bin_range_max}:', round(bin_data['metric'].mean(), 4), metric, f'({len(bin_data)}/{n_runs} runs)')

            xys.append((f'{bin_range_min}-{bin_range_max} ({len(bin_data)}x)', bin_data['metric']))

        boxplots(*zip(*xys), x_label=which_hyperparam, y_label=metric)


def determine_optimal(result_df, hp_df, metric:str, top_n: int=5, 
                      min_max: str='max'):
    """ Determine the optimal hyperparameters based on the top n results."""
    if min_max == 'max':
        top_hps_df = hp_df.iloc[:top_n]
        top_results_df = result_df.iloc[:top_n][metric]
    elif min_max == 'min':
        top_hps_df = hp_df.iloc[-top_n:]
        top_results_df = result_df.iloc[-top_n:][metric]

    print(f'Top hyperparameters ({min_max} metric) with mean {metric} of {round(top_results_df.mean(), 4)}:')

    for hp in top_hps_df.columns:
        print('----------------------')
        try:
            print(f'Mean of {hp}:', top_hps_df[hp].mean().round(4))
        except TypeError:
            pass
        try:
            if top_hps_df[hp].nunique() != len(top_hps_df[hp]):
                print(f'Most used for {hp}:', top_hps_df[hp].mode()[0])
        except TypeError:
            # The hp is probably a list -> Perform count in any way
            print(f'Most used for {hp}:', top_hps_df[hp].mode()[0])


def boxplots(xs, ys, x_label, y_label, save_path=None):
    """ Create boxplots for the given data."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.boxplot(ys)
    ax.set_xticklabels(xs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    main('20240506_test_hp_tuning4', 'actor_fc_dims', metric='average_return', 
         last_n_steps=4, top_n_percentile=80, discrete_hp_steps=3)
