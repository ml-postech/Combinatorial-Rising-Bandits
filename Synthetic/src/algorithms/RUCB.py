import numpy as np


class RUCB(object):
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
        self.mu = np.zeros(self.narms, dtype=float)
        self.slope = np.zeros(self.narms, dtype=float)
        self.selections = np.zeros((self.narms, self.T), dtype=int)
        
    def estimator(self, t, k, pulled):
        cnt = self.counts[k]
        h = np.ceil(cnt/8).astype(int)
        
        if not pulled:
            self.mu[k] = self.mu[k] + self.slope[k]
            
        else:
            observation = self.observations[k, cnt-2*h:cnt]
            slope = observation[h:] - observation[:h]
            self.slope[k] = np.sum(slope) / (h**2)
            self.mu[k] = np.sum(observation[h:] + np.arange(t-cnt+h, t-cnt, -1) * slope / h)/h

        beta = self.sigma*((t+1) - cnt + h - 1) * np.sqrt(10*np.log(t+1)/(h**3))

        return self.mu[k] + beta
        
    def update(self, t, arm, feedback):
        self.selections[arm,t] += 1
        
        self.observations[arm,self.counts[arm]] = feedback
        self.counts[arm] += 1
        
        for i in range(self.narms):
            if self.counts[i] >= 2:
                if i == arm:
                    self.estimations[i] = self.estimator(t, i, True)
                else:
                    self.estimations[i] = self.estimator(t, i, False)
         
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
            