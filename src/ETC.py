import numpy as np
import pickle
import math


class ETC:
    def __init__(self, avg: np.ndarray, explore_steps: int):

        self.time = 0
        self.cum_regret = 0
        self.emp_means = np.zeros_like(avg)  # empirical means of arms
        self.num_pulls = np.zeros_like(avg)  # number of times that arm i has been pulled

        self.means = avg                     # true means of the arms
        self.m = explore_steps               # num of explore steps per arm
        self.num_arms = avg.size             # num arms (k)
        self.best_arm = np.argmax(avg)       # True best arm

    def restart(self):
        # Reset counters
        self.time = 0
        self.cum_regret = 0
        self.emp_means = np.zeros_like(self.means)
        self.num_pulls = np.zeros_like(self.means)

    def get_best_arm(self):
        # For each time index, find the best arm according to ETC.
        return np.argmax(self.emp_means)

    def update_stats(self, rew, arm):
        pass

    def update_reg(self, rew_vec, arm):
        pass

    def iterate(self, rew_vec):
        pass


def get_reward(avg):
    return avg + np.random.normal(len(avg))


def run(avg, explore_steps, iterations, num_repeat):
    regret = np.zeros((num_repeat, iterations))
    etc = ETC(avg=avg, explore_steps=explore_steps)

    for j in range(num_repeat):
        etc.restart()
        for t in range(iterations - 1):
            rew_vec = get_reward(avg)  # Reward Genie
            etc.iterate(rew_vec=rew_vec)

        regret[j, :] = np.asarray(etc.cum_regret)

    return regret


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_inst = int(1e4), 30
    m = 1
    reg = run(avg=mu, explore_steps=m, iterations=num_iter, num_repeat=num_inst)
