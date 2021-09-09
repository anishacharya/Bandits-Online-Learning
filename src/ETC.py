import numpy as np
import pickle
import math
import matplotlib.pyplot as plt


class ETC:
    def __init__(self, avg: np.ndarray, explore_steps: int):

        self.true_means = avg  # true means of the arms
        self.m = explore_steps  # num of explore steps per arm

        self.time = 0
        self.cum_regret = 0

        self.emp_means = np.zeros_like(self.true_means)  # empirical means of arms
        self.num_pulls = np.zeros_like(self.true_means)  # number of times that arm i has been pulled

        self.num_arms = avg.size  # num arms (k)
        self.explore_horizon = self.m * self.num_arms

        self.best_arm = int(np.argmax(self.true_means))  # True best arm

        self.arm_ix = None

    def restart(self):
        # Reset counters
        self.time = 0
        self.cum_regret = 0
        self.emp_means = np.zeros_like(self.true_means)
        self.num_pulls = np.zeros_like(self.true_means)

    def get_best_arm(self):
        # For each time index, find the best arm according to ETC.
        return np.argmax(self.emp_means)

    def update_stats(self, rew_vec):
        ni = self.num_pulls[self.arm_ix]

        # genie plays best arm
        genie_rew = rew_vec[self.best_arm]
        player_rew = rew_vec[self.arm_ix]
        self.cum_regret += (genie_rew - player_rew)

        if self.time < self.m * self.num_arms:
            # keep online average
            self.emp_means[self.arm_ix] = self.emp_means[self.arm_ix] * (ni / (ni + 1)) + player_rew / (ni + 1)

        self.num_pulls[self.arm_ix] += 1
        self.time += 1

    def get_reward(self):
        return self.true_means + np.random.normal(len(self.true_means))

    def iterate(self):
        # Explore Phase - Round Robin
        rew_vec = self.get_reward()

        if self.time < self.explore_horizon:
            self.arm_ix = self.time % self.num_arms

        elif self.time == self.explore_horizon:
            # calculate best arm explored empirical
            self.arm_ix = self.get_best_arm()

        self.update_stats(rew_vec=rew_vec)


def run(avg, explore_steps, iterations, num_repeat):
    regret = np.zeros((num_repeat, iterations))
    etc = ETC(avg=avg, explore_steps=explore_steps)
    for j in range(num_repeat):
        for t in range(iterations - 1):
            etc.iterate()
        etc.restart()

        regret[j, :] = np.asarray(etc.cum_regret)

    return regret


if __name__ == '__main__':
    mu = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_inst = int(1e4), 30
    m = 2
    reg = run(avg=mu,
              explore_steps=m,
              iterations=num_iter,
              num_repeat=num_inst)

    mean_runs = np.mean(reg, axis=0)
    std_runs = np.std(reg, axis=0)

    UB = mean_runs + 1 * std_runs
    LB = mean_runs - 1 * std_runs

    x = np.arange(len(mean_runs))
    plt.plot(x, mean_runs, color='b')
    plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5, color='b')

    plt.show()