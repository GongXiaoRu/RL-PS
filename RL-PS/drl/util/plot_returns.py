import argparse
from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # TODO: Use test returns instead to enable actual performcance comparison
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--directory',
        type=str,
        help="Directory of the data folder",
        default='data/'
    )
    argparser.add_argument(
        '--paths',
        type=str,
        help="Paths to one or multiple experiment folders"
    )
    argparser.add_argument(
        '--rolling-window',
        type=int,
        help="Average returns over the n last episodes",
        default=3,
    )
    argparser.add_argument(
        '--mean',
        help="Compute the mean?",
        action='store_true',
    )
    argparser.add_argument(
        '--columns',
        help="Plot and compare which columns? (e.g. mape, rmse, regret)",
        default='average_return',
    )
    argparser.add_argument(
        '--last',
        help="Plot only the last n data points",
        type=int,
        default='0',
    )
    argparser.add_argument(
        '--from-step',
        help="Plot only steps after `from-step`.",
        type=int,
        default='0',
    )
    argparser.add_argument(
        '--to-step',
        help="Plot only steps before `to-step`.",
        type=int,
        default=None,
    )
    argparser.add_argument(
        '--x-column',
        help="Use this column as x-axis",
        type=str,
        default='',
    )
    argparser.add_argument(
        '--std',
        help="Plot standard deviation as well?",
        action='store_true',
    )
    argparser.add_argument(
        '--remove-hyperparams',
        help="Tuple of hyperparams to delete from the data. Results in squashing of data that differ only in these hyperparams.",
        default="()",
    )

    args = argparser.parse_args()

    mean_reward = defaultdict(list)
    columns = args.columns.split(',')  # TODO Multiple columns!
    # columns = args.columns

    directory = args.directory
    if args.paths:
        paths = args.paths.split(',')
    else:
        paths = os.listdir(directory)

    for path in paths:
        full_path = directory + path

        if full_path[-1] != '/':
            full_path += '/'

        print('\n', full_path)

        run_paths = os.listdir(full_path)
        if 'meta-data.txt' not in run_paths:
            # Go one level deeper and evaluate all experiments in this folder
            all_y = defaultdict(list)
            all_x = defaultdict(list)
            for run_path in run_paths:
                print(run_path)

                try:
                    results = pd.read_csv(
                        full_path + run_path + '/test_returns.csv', index_col=0)

                    x = get_x_data(results, args.x_column)
                except (FileNotFoundError, NotADirectoryError):
                    # Experiment probably not finished yet
                    print('FileNotFoundError')
                    continue

                hyperparams = get_algo_plus_hyperparams(
                    full_path + run_path, eval(args.remove_hyperparams))

                for col in columns:
                    step_mask = results.index >= args.from_step
                    if args.to_step:
                        step_mask *= results.index <= args.to_step
                    all_y[hyperparams + '_' +
                          col].append(results[col].to_numpy()[step_mask])
                    all_x[hyperparams + '_' +
                          col].append(x.to_numpy()[step_mask])

            for idx, (hyperparams, values) in enumerate(all_y.items()):
                if args.mean:
                    print(hyperparams)
                    # if 'normal' in hyperparams or 'uniform' in hyperparams:
                    #     continue
                    # TODO: How to deal with different test intervals?
                    mean_y, std = tolerant_mean(all_y[hyperparams])
                    mask = ~np.isnan(mean_y)
                    mean_y = mean_y[mask]
                    mean_y = pd.Series(mean_y, name='y')
                    longest = np.argmax([len(x) for x in all_x[hyperparams]])
                    label = f'{len(all_y[hyperparams])}x{hyperparams}'

                    plot_return(all_x[hyperparams][longest][mask], mean_y,
                                args.rolling_window, label, args.last,
                                std if args.std else None)
                else:
                    for idx, y in enumerate(values):
                        label = f'{len(values)}x{hyperparams}' if idx == 0 else None
                        next_color = True if idx == 0 else False
                        plot_return(all_x[hyperparams][idx],
                                    pd.Series(y, name='y'),
                                    args.rolling_window, label, args.last,
                                    next_color=next_color)

        else:
            # No sub-level folders
            results = pd.read_csv(
                full_path + '/test_returns.csv', index_col=0)
            y = results[columns]
            x = get_x_data(results, args.x_column)
            hyperparams = get_algo_plus_hyperparams(
                full_path, eval(args.remove_hyperparams))
            plot_return(x, y, args.rolling_window, hyperparams, args.last)

    if args.x_column:
        plt.xlabel(args.x_column)
    else:
        plt.xlabel('Steps')
    plt.ylabel(columns)

    # TODO: Improve legend! Include datetime? Or Hyperparams?
    # Idea: Create a diff of hyperparams/algo name/etc
    plt.legend(fontsize="8")
    plt.grid()

    plt.show()


def get_algo_plus_hyperparams(path, remove_hyperparams: tuple=()):
    with open(path + '/meta-data.txt') as f:
        lines = f.readlines()
    algo = lines[2][15:]
    with open(path + '/agent_hyperparams.json') as f:
        hp_dict = json.load(f)
    with open(path + '/env_hyperparams.json') as f:
        ehp_dict = json.load(f)

    for rhp in remove_hyperparams:
        hp_dict.pop(rhp, None)
        ehp_dict.pop(rhp, None)
    hyperparams = str(hp_dict)
    env_hyperparams = str(ehp_dict)

    return (algo + '_' + hyperparams + '_' + env_hyperparams).replace('\n', '')


def plot_return(x, y, window, label='None', last=0, std=None, next_color=True):
    ax = plt.gca()
    if next_color:
        c = next(ax._get_lines.prop_cycler)['color']
    else:
        c = ax.lines[-1].get_color()
    y = y.rolling(window=window, min_periods=min(window, 2)).mean()
    plt.plot(x[-last:], y[-last:], label=label, color=c, marker='o')
    try:
        print('Final mean:', round(y.values[-1].item(), 4), f'after {x[-1]} steps')
    except IndexError:
        pass
    if std is not None:
        ax.fill_between(x[-last:], y1=(y - std)[-last:],
                        y2=(y + std)[-last:], color=c, alpha=0.1)


def tolerant_mean(arrs):
    """ np.mean() for arrays with different lengths.
    From: https://stackoverflow.com/questions/10058227/calculating-mean-of-arrays-with-different-lengths
    """

    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l

    return np.nanmean(arr, axis=-1), np.nanstd(arr, axis=-1)


def get_x_data(results, x_column):
    if not x_column:
        return results.index
    return results[x_column]


if __name__ == '__main__':
    main()
