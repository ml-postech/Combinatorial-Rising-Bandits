import numpy as np
import networkx as nx
from networkx import SpanningTreeIterator

class Spanning(object):
    def __init__(self, seed, T, easy = False):
        self.seed = seed
        self.T = T
        self.easy = easy

        if self.easy:
            self.nnodes = 4
            self.narms = 6
            self.nsuperarms = 8
            self.nlengths = 3
            self.sigma = 0.01
            self.initial =      np.array([0, 1, 0,
                                             1, 0,
                                                1,], dtype=float)

            const = 250
            satu = 5000
            self.indices = np.repeat(np.arange(const+1, self.T+2, const),const)
            self.inverse = self.indices ** -1.2
            self.inverse[satu:] = 0
            self.mu = np.zeros((self.narms, self.T), dtype=float)
            for i in range(self.narms):
                if self.initial[i] == 1:
                    for j in range(1, self.T):
                        self.mu[i][j] = self.mu[i][j-1] + self.inverse[j-1]
                    if self.mu[i][self.T-1] > 1:
                        self.mu[i] = self.mu[i] / self.mu[i][self.T-1]    
                else:
                    self.mu[i] += 0.9

        else:
        
            self.nnodes = 6
            self.narms = 15
            self.nsuperarms = 1296
            self.nlengths = 5
            
            self.sigma = 0.01
            self.initial =      np.array([0, 1, 0, 1, 0,
                                             1, 1, 0, 1,
                                                0, 0, 0, 
                                                   0, 0, 
                                                      0
                                                        ], dtype=float)
        
            const = 1000
            satu = 20000
            self.indices = np.repeat(np.arange(const+1, self.T+2, const),const)
            self.inverse = self.indices ** -1.2
            self.inverse[satu:] = 0
            self.mu = np.zeros((self.narms, self.T), dtype=float)
            for i in range(self.narms):
                if self.initial[i] == 1.0:
                    for j in range(1, self.T):
                        self.mu[i][j] = self.mu[i][j-1] + self.inverse[j-1]
                    if self.mu[i][self.T-1] > 1:
                        self.mu[i] = self.mu[i] / self.mu[i][self.T-1]    
                else:
                    self.mu[i] += 0.6
        
        self.G = nx.Graph()
        self._from_to = np.zeros((self.narms,2), dtype=int)
        
        cnt = 0
        
        for i in range(self.nnodes-1):
            for j in range(i + 1, self.nnodes):
                self._from_to[cnt][0] = i
                self._from_to[cnt][1] = j                
                cnt += 1
        
        for i in range(self.narms):
            self.G.add_edge(self._from_to[i][0], self._from_to[i][1], weight = 0)
                    
        self.superarm = np.zeros((self.nsuperarms, self.nlengths), dtype=int)
                
    def set_superarm(self):
        Generator = SpanningTreeIterator(self.G)
        Generator.__iter__()
        for i in range(self.nsuperarms):
            tree = Generator.__next__()
            
            superarm = np.array([])
            for edge in tree.edges(data=True):
                superarm = np.concatenate((superarm, np.where((self._from_to == edge[0:2]).all(axis=1))[0]),axis=0)
            self.superarm[i] = superarm
            
        return
    
    def rewards(self):
        np.random.seed(self.seed)
        return np.clip(self.mu + self.sigma * np.random.randn(self.narms, self.T), 0, 1)
    
    def update(self, rewards):
        for i in range(self.narms):
            self.G[self._from_to[i][0]][self._from_to[i][1]]['weight'] = -rewards[i]
        return
    
    def get_optimal_action_candi(self):
        if self.easy:
            a = np.array([[0, 1, 2]])
            b = np.array([[1, 3, 5]])
        else:
            a = np.array([[0, 2, 4, 7, 10]])
            b = np.array([[1, 3, 5, 6, 8]])
        return np.concatenate((a, b))
    
    def random_i(self, i):       
        return self.superarm[np.random.choice(np.where(self.superarm == i)[0])]
    
    def best(self):
        tree = nx.minimum_spanning_tree(self.G, weight='weight')
        
        best = np.array([])
        for edge in tree.edges(data=True):
            best = np.concatenate((best, np.where((self._from_to == edge[0:2]).all(axis=1))[0]), axis=0)
            
        return best.astype(int)
        
    