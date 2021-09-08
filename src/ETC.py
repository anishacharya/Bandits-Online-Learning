import numpy as np
import pickle
import math


def run(avg, m, num_iter, num_inst):
    reg = np.zeros((num_inst, num_iter))

    return reg


class ETC:
    def __init__(self, avg, m):  ## Initialization

        self.time = 0
        self.cum_regret = 0
        self.emp_means = np.zeros_like(avg) # empirical means of arms
        self.num_pulls = np.zeros_like(avg) # number of times that arm i has been pulled


        self.means = avg                    # true means of the arms
        self.m = m                          # num of explore steps per arm
        self.num_arms = avg.size            # num arms (k)
        self.best_arm = np.argmax(avg)      # best arm *


    def restart(self):
        # Reset counters
        self.time = 0
        self.cum_regret = 0
        self.emp_means = np.zeros_like(self.means)
        self.num_pulls = np.zeros_like(self.means)



    def get_best_arm(self):
    # For each time index, find the best arm according to ETC.


    def update_stats(self, rew, arm):  ## Update the empirical means, the number of pulls, and increment the time index

        ## Your code here

        return None

    def update_reg(self, rew_vec, arm):  ## Update the cumulative regret

        ## Your code here

        return None

    def iterate(self, rew_vec):  ## Iterate the algorithm

        ## Your code here

        return None


def get_reward(avg):
    return avg + np.random.normal(len(avg))


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_inst = int(1e4), 30
    m = 1
    reg = run(avg=mu, m=m, num_iter=num_iter, num_inst=num_inst)
