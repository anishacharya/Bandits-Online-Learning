import numpy as np
import pickle
import math


class ETC:
    def __init__(self, avg: np.ndarray, explore_steps: int):

        self.time = 0
        self.cum_regret = 0
        self.emp_means = np.zeros_like(avg)  # empirical means of arms
        self.num_pulls = np.zeros_like(avg)  # number of times that arm i has been pulled

        self.means = avg  # true means of the arms
        self.m = explore_steps  # num of explore steps per arm
        self.num_arms = avg.size  # num arms (k)
        self.best_arm = np.argmax(avg)  # True best arm

        self.arm_ix = None

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

    def update_reg(self, rew_vec):
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
        return self.means + np.random.normal(len(self.means))

    def iterate(self):
        # Explore Phase - Round Robin
        rew_vec = self.get_reward()

        if self.time < self.m * self.num_arms:
            self.arm_ix = self.time % self.m

        elif self.time == self.m * self.num_arms:
            # calculate best arm explored empirical
            self.arm_ix = self.get_best_arm()

        self.update_stats(rew=rew_vec, arm=self.arm_ix)


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
