
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO: Maybe this complete thing is obsolete with test_returns.csv file?!


class Eval:
    def __init__(self, agent_names=['Unnamed agent'], average_last_eps=30,
                 plot_interval_steps=300, path='temp/'):
        # TODO: Average over last n step, not episoded (too high variance in eps)
        self.average_last_eps = average_last_eps
        self.plot_interval_steps = plot_interval_steps

        self.path = path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.plot = True

        self.steps = []
        self.times = []
        self.returns = {a_id: list() for a_id in agent_names}
        self.avrg_returns = {a_id: list() for a_id in agent_names}
        self.next_plot_counter = plot_interval_steps

    def step(self, return_, step: int, test=False):
        self.steps.append(step)

        for a_id in self.returns.keys():
            # TODO: Use scalar or dict, but only one of them, not both!
            if isinstance(return_, float) or isinstance(return_, int):
                if np.isnan(return_):
                    return_ = self.avrg_returns[a_id][-1]
                self.returns[a_id].append(return_)
            elif isinstance(return_, dict):
                if hasattr(return_[a_id], '__iter__'):
                    # If iterable: Use the sum of all returns
                    return_[a_id] = sum(return_[a_id])
                self.returns[a_id].append(return_[a_id])

                # Prevent NAN values
                if np.isnan(self.returns[a_id][-1]):
                    if len(self.avrg_returns[a_id]) == 0:
                        self.returns[a_id][-1] = 0
                    else:
                        # Replace with average of last returns
                        self.returns[a_id][-1] = self.avrg_returns[a_id][-1]
            else:
                for i, entry in enumerate(return_):
                    if np.isnan(entry):
                        return_[i] = self.avrg_returns[a_id][-1]
                self.returns[a_id].append(return_)

            self.avrg_returns[a_id].append(
                np.mean(self.returns[a_id][-self.average_last_eps:]))

        # self.store_return(return_, step, test)

        if step >= self.next_plot_counter and self.plot:
            self.next_plot_counter = step + self.plot_interval_steps
            self.plot_reward()
            print('Step: ', step, '| Episode: ', len(self.steps) - 1)

    def plot_reward(self):
        for a_id in self.avrg_returns.keys():
            plt.plot(self.steps[int(self.average_last_eps / 10):],
                     self.avrg_returns[a_id][int(self.average_last_eps / 10):],
                     'x-', label=a_id)
        plt.xlabel('Step')
        plt.ylabel(f'Mean return of last {self.average_last_eps} eps')
        plt.legend()
        plt.grid(visible=True, which='major', axis='both')
        plt.savefig(self.path + 'return_course.png')
        plt.close('all')

    def store_return(self, return_, step, test=False):

        if isinstance(return_, np.ndarray):
            row = [step] + list(return_)
        elif isinstance(return_, dict):
            row = [step] + list(return_.values())
        else:
            row = [step, return_]

        path = self.path + 'test_' if test else self.path
        with open(f'{path}returns.csv', 'a') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(row)
