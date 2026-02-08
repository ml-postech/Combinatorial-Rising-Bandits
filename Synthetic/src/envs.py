import numpy as np
from tasks.matching import Matching
import matplotlib.pyplot as plt
from tasks.path import Path
from tasks.toy import Toy
from tasks.spanning import Spanning
import pandas as pd
from tasks.max import Max


class Environment(object):
    def __init__(self, task, T, seed, easy):
        self.task = task
        self.T = T
        self.seed = seed
        self.easy = easy
        
        self.environment = globals()[self.task](self.seed, self.T, self.easy)
        self.nlengths = self.environment.nlengths
        self.counts = np.zeros(self.environment.narms, dtype=int)
        self.rewards = self.environment.rewards()
        self.environment.set_superarm()
    
    def get_optimal_cumulative_rewards(self):
        m_candi = self.environment.get_optimal_action_candi()
        optimal_cumulative_rewards = np.zeros((m_candi.shape[0], self.T), dtype=float)

        for i in range(1, self.T):
            for j in range(m_candi.shape[0]):
                if self.task == 'Max':
                    optimal_cumulative_rewards[j,i] = optimal_cumulative_rewards[j,i-1] + np.max(self.rewards[m_candi[j], i])
                else:
                    optimal_cumulative_rewards[j, i] = optimal_cumulative_rewards[j, i-1] + np.sum(self.rewards[m_candi[j], i])
        
        if self.task == 'Max':
            return optimal_cumulative_rewards[0]
        else:
            return np.max(optimal_cumulative_rewards, axis=0)
        
    def reset(self):
        self.counts = np.zeros(self.environment.narms, dtype=int)
    
    def pull(self, arms):
        rewards = np.zeros(self.environment.narms, dtype=float)
        
        for i in arms:
            rewards[int(i)] = self.rewards[int(i),self.counts[int(i)]]
            self.counts[int(i)]+=1
            
        return arms, rewards

    def pull_noncombi(self, arm):
        
        _, rewards = self.pull(self.environment.superarm[arm])
        
        if self.task == 'Max':
            return arm, np.max(rewards)/self.nlengths
        else:
            return arm, np.sum(rewards)/self.nlengths