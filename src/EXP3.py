import numpy as np
import matplotlib.pyplot as plt


class UCB:
    def __init__(self, avg: np.ndarray):
        self.true_means = avg  # true means of the arms
        self.num_arms = avg.size  # num arms (k)
        self.best_arm = int(np.argmax(self.true_means))  # True best arm
        # sort = np.sort(self.true_means)[::-1]
        # self.delta_min = sort[0] - sort[1]
        # self.C = 1

        self.time = 0
        self.regret = []
        self.emp_means = np.zeros_like(self.true_means)  # empirical means of arms  \hat{\mu_j}
        self.num_pulls = np.zeros_like(self.true_means)  # number of times that arm i has been pulled T_j
        self.ucb_arr = 1e5 * np.ones_like(self.true_means)  # Upper confidence bounds i.e. U_j

        self.arm_ix = None

    def restart(self):
        # Reset counters
        self.time = 0
        self.regret = []

        self.emp_means = np.zeros_like(self.true_means)
        self.num_pulls = np.zeros_like(self.true_means)
        self.ucb_arr = 1e5 * np.ones_like(self.true_means)

        self.arm_ix = None

    def get_best_arm(self):
        # For each time index, find the best arm according to UCB
        return np.argmax(self.ucb_arr)

    def update_ucb(self):
        f = 1 + self.time * (np.log(self.time + 1) ** 2)
        for j in range(self.num_arms):
            # So that T[j-1] is not 0 ~ div by zero error else
            nj = 1 if self.num_pulls[j] == 0 else self.num_pulls[j]
            self.ucb_arr[j] = self.emp_means[j] + np.sqrt((2 * np.log(f)) / nj)

    def update_stats(self, rew_vec):
        # genie plays best arm
        genie_rew = rew_vec[self.best_arm]
        player_rew = rew_vec[self.arm_ix]
        self.regret.append((genie_rew - player_rew))

        ni = self.num_pulls[self.arm_ix]
        self.emp_means[self.arm_ix] = self.emp_means[self.arm_ix] * (ni / (ni + 1)) + player_rew / (ni + 1)
        self.num_pulls[self.arm_ix] += 1
        self.time += 1

    def get_reward(self):
        return self.true_means + np.random.normal(0, 1, np.shape(self.true_means))

    def iterate(self):
        # if self.time < self.num_arms:
        #     # So that T[j-1] is not 0 ~ div by zero error else
        #     self.arm_ix = self.time
        # else:
        self.update_ucb()
        self.arm_ix = self.get_best_arm()

        rew_vec = self.get_reward()
        self.update_stats(rew_vec=rew_vec)


def run(avg, iterations, num_repeat, eta, var):
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
    mu = np.asarray([0.8, 0.7, 0.5])
    num_iter, num_inst = int(2e3), 20

    eta = np.sqrt(np.log(mu.size) / (num_iter * mu.size))
    var = 0.01

    reg = run(avg=mu,
              iterations=num_iter,
              num_repeat=num_inst,
              eta=eta,
              var=var)

    mean_runs = np.mean(reg, axis=0)
    std_runs = np.std(reg, axis=0)

    UB = mean_runs + 1 * std_runs
    LB = mean_runs - 1 * std_runs

    x = np.arange(len(mean_runs))
    plt.plot(x, mean_runs, color='b')
    # plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5, color='b')

    plt.xlabel('Time (Log Scale)', fontsize=10)
    plt.ylabel('Cumulative Regret with UCB', fontsize=10)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.show()
