import numpy as np
import matplotlib.pyplot as plt


class EXP3:
    def __init__(self, avg: np.ndarray, lr: float, algo: str = 'exp3'):
        self.true_means = avg                            # true means of the arms
        self.num_arms = avg.size                         # num arms (k)
        self.best_arm = int(np.argmax(self.true_means))  # True best arm
        self.lr = lr

        self.algo = algo
        self.clip = 0.5 * self.lr
        self.gamma = 0.5 * self.lr

        self.restart()

    def restart(self):
        # Reset counters
        self.time = 0
        self.S = np.array([0.0] * self.num_arms)                 # S_t,j = initialize to zero
        self.P = None                                            # P_t,j = initialized uniformly at t=0 by update_exp3()
        self.arm_ix = None

        self.regret = []

    def get_best_arm(self):
        # For each time index, sample the best arm based off P_(t-1),j
        all_ix = np.arange(self.num_arms)
        return np.random.choice(a=all_ix, size=1, replace=False, p=self.P)
        # return np.argmax(self.P)

    def update_exp3(self):
        # calculate and update P_t,j
        exp_wt = np.exp(self.S * self.lr)
        self.P = exp_wt / sum(exp_wt)

    def update_stats(self, rew_vec):
        # update regret
        genie_rew = rew_vec[self.best_arm]
        player_rew = rew_vec[self.arm_ix]
        self.regret.append((genie_rew - player_rew))

        # update S
        if self.algo == 'exp3':
            self.S[self.arm_ix] += 1 - ((1 - rew_vec[self.arm_ix]) / self.P[self.arm_ix])
        elif self.algo == 'exp3_clip':
            clipped_estimate = (1 / self.clip) * min(1.0, (self.clip / self.P[self.arm_ix]))
            self.S[self.arm_ix] += 1 - (1 - rew_vec[self.arm_ix]) * clipped_estimate
        elif self.algo == 'exp3_ix':
            self.S[self.arm_ix] += 1 - ((1 - rew_vec[self.arm_ix]) / (self.P[self.arm_ix] + self.gamma))

        self.time += 1

    def get_reward(self):
        return self.true_means + np.random.normal(0, 0.01, np.shape(self.true_means))

    def iterate(self):
        self.update_exp3()
        self.arm_ix = self.get_best_arm()
        rew_vec = self.get_reward()
        self.update_stats(rew_vec=rew_vec)


def run(avg, iterations, num_repeat, eta):
    regret = np.zeros((num_repeat, iterations))
    exp3 = EXP3(avg=avg, lr=eta)

    for j in range(num_repeat):
        for t in range(iterations):
            exp3.iterate()

        # calculate cumulative regret
        regret[j, :] = np.cumsum(np.asarray(exp3.regret))
        exp3.restart()

    return regret


if __name__ == '__main__':
    mu = np.asarray([0.8, 0.7, 0.5])
    num_iter, num_inst = int(2e3), 20

    eta = np.sqrt(np.log(mu.size) / (num_iter * mu.size))

    reg = run(avg=mu,
              iterations=num_iter,
              num_repeat=num_inst,
              eta=eta)

    mean_runs = np.mean(reg, axis=0)
    std_runs = np.std(reg, axis=0)

    UB = mean_runs + 1 * std_runs
    LB = mean_runs - 1 * std_runs

    x = np.arange(len(mean_runs))
    plt.plot(x, mean_runs, color='b')
    # plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5, color='b')

    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Cumulative Regret with EXP3', fontsize=10)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.show()
