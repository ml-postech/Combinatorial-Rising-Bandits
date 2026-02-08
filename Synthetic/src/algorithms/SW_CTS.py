import numpy as np


class SW_CTS(object):
    def __init__(self, T, env, task):
        self.T = T
        self.env = env
        self.task = task
        self.sigma = self.env.environment.sigma
        self.narms = self.env.environment.narms
        self.observations = np.zeros((self.narms, self.T), dtype=float)
        self.counts = np.zeros(self.narms, dtype=int)
        self.estimations = np.zeros(self.narms, dtype=float)
        self.cumulative_rewards = np.zeros(self.T, dtype=float)
        self.window_size = np.sqrt(self.T)
        self.selections = np.zeros((self.narms, self.T), dtype=int)
        
    def estimator(self, t, k):
        cnt = self.counts[k]
        h = int(min(self.window_size, cnt))
        
        reward_sum = np.sum(self.observations[k,cnt-h:cnt])
        
        reward_alpha = 1 + reward_sum
        reward_beta = 1 + h - reward_sum
        
        mu = np.random.beta(reward_alpha, reward_beta)
        
        return mu
        
    def update(self, t, arms, feedback):
        for i in arms:
            self.selections[i,t] += 1
            self.observations[i,self.counts[i]] = feedback[i]
            self.counts[i] += 1
        
        for i in range(self.narms):
            if self.counts[i] >= 2:
                self.estimations[i] = self.estimator(t, i)
        
        self.env.environment.update(self.estimations)
        return
        
    def run(self):
        for t in range(self.T):
            if t < self.narms*2:
                arms, feedback = self.env.pull(self.env.environment.random_i(int(t%self.narms)))
                if t != 0:
                    if self.task == 'Max':
                        self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + np.max(feedback)
                    else:
                        self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + np.sum(feedback)
                           
            else:
                arms, feedback = self.env.pull(self.env.environment.best())
                if self.task == 'Max':
                    self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + np.max(feedback)
                else:
                    self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + np.sum(feedback)
                
            self.update(t, arms, feedback)
            if t %10000 == 0:
                print(t)
        return self.cumulative_rewards, self.selections
            