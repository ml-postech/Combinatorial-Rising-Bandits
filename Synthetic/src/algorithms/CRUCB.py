import numpy as np


class CRUCB(object):
    def __init__(self, T, env, task, epsilon = 0.125):
        self.T = T
        self.env = env
        self.task = task
        self.sigma = self.env.environment.sigma
        self.narms = self.env.environment.narms
        self.observations = np.zeros((self.narms, self.T), dtype=float)
        self.counts = np.zeros(self.narms, dtype=int)
        self.estimations = np.zeros(self.narms, dtype=float)
        self.cumulative_rewards = np.zeros(self.T, dtype=float)
        self.mu = np.zeros(self.narms, dtype=float)
        self.slope = np.zeros(self.narms, dtype=float)
        self.selections = np.zeros((self.narms, self.T), dtype=int)
        self.epsilon = epsilon
        
    def estimator(self, t, k, pulled):
        cnt = self.counts[k]
        if self.epsilon > 0.33:
            h = np.floor(cnt*self.epsilon).astype(int)
            if h == 0:
                h = 1
        else:
            h = np.ceil(cnt*self.epsilon).astype(int)
        
        if not pulled:
            self.mu[k] = self.mu[k] + self.slope[k]
        else:
            observation = self.observations[k, cnt-2*h:cnt]
            slope = observation[h:] - observation[:h]
            self.slope[k] = np.sum(slope) / (h**2)
            self.mu[k] = np.sum(observation[h:] + np.arange(t-cnt+h, t-cnt, -1) * slope / h)/h
            
        
        beta = self.sigma * (t - cnt + h) * np.sqrt(4*np.log(t+1)/(h**3))

        return self.mu[k] + beta
        
    def update(self, t, arms, feedback):
        for i in arms:
            self.selections[i,t] += 1
            self.observations[i,self.counts[i]] = feedback[i]
            self.counts[i] += 1
        
        for i in range(self.narms):
            if self.counts[i] >= 2:
                if np.isin(i, arms):
                    self.estimations[i] = self.estimator(t, i, True)
                else:
                    self.estimations[i] = self.estimator(t, i, False)
        
        self.env.environment.update(self.estimations)
        return
        
    def run(self):
        for t in range(self.T):
            if t %10000 == 0:
                print(t)
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
        return self.cumulative_rewards, self.selections