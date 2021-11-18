import numpy as np
import matplotlib.pyplot as plt


class EXP3:
    def __init__(self, Delta:float, lr: float):
        #self.true_means = avg  # true means of the arms

        self.true_means = [0.5]*10
        self.true_means[8] = 0.5 + Delta
        self.true_means[9] = 0.5 - Delta
        self.true_means = np.asarray(self.true_means)
        
        self.num_arms = 10  # num arms (k)
        self.best_arm = 9 #int(np.argmax(self.true_means))  # True best arm
        self.lr = lr

        self.restart()

    def restart(self):
        # Reset counters
        self.time = 0
        self.L = np.array([0.0] * self.num_arms)                 # S_t,j = initialize to zero
        self.P = None                                            # P_t,j = initialized uniformly at t=0 by update_exp3()
        self.arm_ix = None

        self.regret = []

        self.true_means = [0.5]*10
        self.true_means[8] = 0.5 + Delta
        self.true_means[9] = 0.5 - Delta
        self.true_means = np.asarray(self.true_means)

        self.best_arm = 8

    def get_best_arm(self):
        # For each time index, sample the best arm based off P_(t-1),j
        all_ix = np.arange(self.num_arms)
        return np.random.choice(a=all_ix, size=1, replace=False, p=self.P)

    def update_exp3(self):
        # calculate and update P_t,j
        exp_wt = np.exp(-self.L * self.lr)
        self.P = exp_wt / sum(exp_wt)

    def update_stats(self, rew_vec):
        # update regret
        genie_rew = rew_vec[self.best_arm]
        player_rew = rew_vec[self.arm_ix]
        self.regret.append((genie_rew - player_rew))

        # update L
        self.L[self.arm_ix] += ((1 - rew_vec[self.arm_ix]) / (self.P[self.arm_ix] + 0.5*self.lr))

        self.time += 1

    def get_reward(self):
        #return self.true_means + np.random.normal(0, 0.01, np.shape(self.true_means))
        #rewards = []
        #for j in range(self.num_arms):
        #    rewards.append(np.random.binomial(n=1, p=self.true_means[j]))

        rewards = np.random.binomial(n=1, p=self.true_means)

        return rewards

    def iterate(self):
        self.time += 1
        
        if self.time > 5e4:
            self.true_means[9] = 0.5 + 4*Delta
            self.best_arm = 9
        
        self.update_exp3()
        self.arm_ix = self.get_best_arm()
        rew_vec = self.get_reward()
        #self.per_arm_total_rew  += rew_vec
        self.update_stats(rew_vec=rew_vec)


def run(Delta, iterations, num_repeat, eta):
    regret = np.zeros((num_repeat, iterations))
    exp3 = EXP3(Delta=Delta, lr=eta)

    for j in range(num_repeat):
        for t in range(iterations):
            exp3.iterate()

        # calculate cumulative regret
        regret[j, :] = np.cumsum(np.asarray(exp3.regret))
        print(j)
        mean_runs = np.mean(regret[0:j+1,:], axis=0)
        std_runs = np.std(regret[0:j+1,:], axis=0)
        print("Mean total regret at the end:", mean_runs[-1])
        print("Std:", std_runs[-1])
        exp3.restart()
        #print(j)

    return regret


if __name__ == '__main__':
    #mu = np.asarray([0.8, 0.7, 0.5])
    num_arms = 10
    num_iter, num_inst = int(1e5), 10
    Delta = 0.1

    #eta = np.sqrt(np.log(num_arms) / (num_iter * num_arms))
    eta = 0.01

    reg = run(Delta=Delta,
              iterations=num_iter,
              num_repeat=num_inst,
              eta=eta)

    mean_runs = np.mean(reg, axis=0)
    std_runs = np.std(reg, axis=0)

    print("Mean total regret at the end:", mean_runs[-1])
    print("Std:", std_runs[-1])

    UB = mean_runs + 1 * std_runs
    LB = mean_runs - 1 * std_runs

    x = np.arange(len(mean_runs))
    plt.plot(x, mean_runs, color='b')
    plt.fill_between(x, LB, UB, alpha=0.3, linewidth=0.5, color='b')

    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Cumulative Regret with EXP3-IX', fontsize=10)
    #plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.show()

    #print("Mean total regret at the end:", mean_runs[-1])
    #print("Std:", std_runs[-1])



