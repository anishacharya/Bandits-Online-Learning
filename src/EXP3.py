import numpy as np
import matplotlib.pyplot as plt


class EXP3:
    def __init__(self, avg: np.ndarray, lr: float, algo: str = 'exp3', reward_dist='normal'):
        self.true_means = avg                            # true means of the arms
        self.num_arms = avg.size                         # num arms (k)
        self.best_arm = int(np.argmax(self.true_means))  # True best arm
        self.lr = lr

        self.algo = algo
        self.clip = 0.5 * self.lr
        self.soft_clip = 0.5 * self.lr
        self.gamma = 0.5 * self.lr

        self.reward_dist = reward_dist
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
        elif self.algo == 'exp3_ix':
            self.S[self.arm_ix] += 1 - ((1 - rew_vec[self.arm_ix]) / (self.P[self.arm_ix] + self.gamma))
        elif self.algo == 'exp3_clip':
            clipped_estimate = (1 / self.clip) * min(1.0, (self.clip / self.P[self.arm_ix]))
            self.S[self.arm_ix] += 1 - (1 - rew_vec[self.arm_ix]) * clipped_estimate
        elif self.algo == 'exp3_soft_clip':
            clipped_estimate = (1 / self.soft_clip) * np.log(1.0 + (self.soft_clip / self.P[self.arm_ix]))
            self.S[self.arm_ix] += 1 - (1 - rew_vec[self.arm_ix]) * clipped_estimate

        self.time += 1

    def get_reward(self):
        if self.reward_dist == 'normal':
            return self.true_means + np.random.normal(0, 0.01, np.shape(self.true_means))
        elif self.reward_dist == 'bin':
            return np.random.binomial(n=1, p=self.true_means)
        else:
            raise NotImplementedError

    def iterate(self):
        if self.time > 5e4:
            self.true_means[9] = 0.5 + 4 * 0.1
            self.best_arm = int(np.argmax(self.true_means))

        self.update_exp3()
        self.arm_ix = self.get_best_arm()
        rew_vec = self.get_reward()
        self.update_stats(rew_vec=rew_vec)


def run(avg, iterations, num_repeat, eta=0.001, algo='exp3'):
    regret = np.zeros((num_repeat, iterations))
    exp3 = EXP3(avg=avg, lr=eta, algo=algo)

    for j in range(num_repeat):
        for t in range(iterations):
            exp3.iterate()

        # calculate cumulative regret
        regret[j, :] = np.cumsum(np.asarray(exp3.regret))
        exp3.restart()

    return regret


if __name__ == '__main__':
    # mu = np.asarray([0.8, 0.7, 0.5])
    num_arms = 10
    delta = 0.1
    mu = np.asarray([0.5] * num_arms)
    mu[8] += delta
    mu[9] -= delta

    num_iter, num_inst = int(2e3), 20

    eta = np.sqrt(np.log(mu.size) / (num_iter * mu.size))

    algos = ['exp3', 'exp3_ix', 'exp3_clip', 'exp3_soft_clip']

    for algo in algos:
        print('running algo {}'.format(algo))
        reg = run(avg=mu,
                  iterations=num_iter,
                  num_repeat=num_inst,
                  eta=eta,
                  algo=algo)

        mean_runs = np.mean(reg, axis=0)
        std_runs = np.std(reg, axis=0)

        UB = mean_runs + 3 * std_runs
        LB = mean_runs - 3 * std_runs

        x = np.arange(len(mean_runs))
        plt.plot(x, mean_runs, label=algo)
        # plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5)

    plt.legend()
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Cumulative Regret', fontsize=10)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')

    plt.show()
