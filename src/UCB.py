import numpy as np
import matplotlib.pyplot as plt


class UCB:
    def __init__(self, avg: np.ndarray):

        self.true_means = avg  # true means of the arms
        sort = np.sort(self.true_means)[::-1]
        self.delta_min = sort[0] - sort[1]
        self.C = 1

        self.time = 0
        self.regret = []

        self.emp_means = np.zeros_like(self.true_means)  # empirical means of arms
        self.num_pulls = np.zeros_like(self.true_means)  # number of times that arm i has been pulled

        self.num_arms = avg.size  # num arms (k)
        self.best_arm = int(np.argmax(self.true_means))  # True best arm

        self.arm_ix = None

    def restart(self):
        # Reset counters
        self.time = 0
        self.regret = []
        self.emp_means = np.zeros_like(self.true_means)
        self.num_pulls = np.zeros_like(self.true_means)
        self.arm_ix = None

    def get_best_arm(self):
        # For each time index, find the best arm according to ETC.
        return np.argmax(self.emp_means)

    def update_stats(self, rew_vec):
        ni = self.num_pulls[self.arm_ix]

        # genie plays best arm
        genie_rew = rew_vec[self.best_arm]
        player_rew = rew_vec[self.arm_ix]
        self.regret.append((genie_rew - player_rew))

        # if explore_flag == 1:
        # keep online average
        self.emp_means[self.arm_ix] = self.emp_means[self.arm_ix] * (ni / (ni + 1)) + player_rew / (ni + 1)
        self.num_pulls[self.arm_ix] += 1

        self.time += 1

    def get_reward(self):
        return self.true_means + np.random.normal(0, 1, np.shape(self.true_means))

    def iterate(self):
        rew_vec = self.get_reward()

        if self.time < self.num_arms:
            # Pure explore Phase - Round Robin for 1 round
            self.arm_ix = self.time
            explore = 1

        else:
            # toss a coin
            epsilon = min(1, (self.C * self.num_arms) / (self.time * self.delta_min ** 2))
            explore = np.random.binomial(n=1, p=epsilon)

            if explore == 1:
                # case-1 explore
                # choose an arm uniformly at random
                self.arm_ix = np.random.randint(low=0, high=self.num_arms)
            else:
                # case-2 exploit
                self.arm_ix = self.get_best_arm()

        self.update_stats(rew_vec=rew_vec)


def run(avg, iterations, num_repeat):
    regret = np.zeros((num_repeat, iterations))
    ucb = UCB(avg=avg)
    for j in range(num_repeat):
        for t in range(iterations):
            ucb.iterate()

        # calculate cumulative regret
        regret[j, :] = np.cumsum(np.asarray(ucb.regret))
        ucb.restart()

    return regret


if __name__ == '__main__':
    mu = np.asarray([0.96, 0.7, 0.5, 0.6, 0.1])
    num_iter, num_inst = int(1e4), 10

    reg = run(avg=mu,
              iterations=num_iter,
              num_repeat=num_inst)

    mean_runs = np.mean(reg, axis=0)
    std_runs = np.std(reg, axis=0)

    UB = mean_runs + 1 * std_runs
    LB = mean_runs - 1 * std_runs

    x = np.arange(len(mean_runs))
    plt.plot(x, mean_runs, color='b')
    # plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5, color='b')

    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Cumulative Regret', fontsize=10)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.show()
