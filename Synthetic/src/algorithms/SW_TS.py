import numpy as np


class SW_TS(object):
    def __init__(self, T, env, task):
        self.T = T
        self.env = env
        self.task = task
        self.sigma = self.env.environment.sigma
        self.narms = self.env.environment.nsuperarms
        self.nlengths = self.env.environment.nlengths
        self.observations = np.zeros((self.narms, self.T), dtype=float)
        self.counts = np.zeros(self.narms, dtype=int)
        self.estimations = np.zeros(self.narms, dtype=float)
        self.cumulative_rewards = np.zeros(self.T, dtype=float)
        self.window_size = np.sqrt(self.T)
        self.selections = np.zeros((self.narms, self.T), dtype=int)
        
    def estimator(self, t, k):
        h = int(min(self.window_size, self.counts[k]))
        
        reward_sum = np.sum(self.observations[k,self.counts[k]-h:self.counts[k]])
        
        reward_alpha = 1 + reward_sum
        reward_beta = 1 + h - reward_sum
        
        mu = np.random.beta(reward_alpha, reward_beta)
        
        return mu
        
    def update(self, t, arm, feedback):
        self.selections[arm,t] += 1
        self.observations[arm,self.counts[arm]] = feedback
        self.counts[arm] += 1
        
        for i in range(self.narms):
            if self.counts[i] >= 2:
                self.estimations[i] = self.estimator(t, i)

        return
        
    def run(self):
        for t in range(self.T):
            if t < self.narms*2:
                arm, feedback = self.env.pull_noncombi(int(t%self.narms))
                if t != 0:
                    self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + feedback * self.nlengths
            else:
                arm, feedback = self.env.pull_noncombi(np.argmax(self.estimations))
                self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + feedback * self.nlengths

            self.update(t, arm, feedback)
            if t %10000 == 0:
                print(t)
        return self.cumulative_rewards, self.selections
            