import numpy as np
import networkx as nx

class Matching(object):
    def __init__(self, seed, T, easy = False):
        self.seed = seed
        self.T = T
        self.easy = easy
        if self.easy:
            self.nmans = 2
            self.njobs = 4
            self.narms = 8
            self.nsuperarms = 12
            self.nlengths = 2
            self.sigma = 0.01
            self.initial =      np.array([1.0, 0.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0, 0.0], dtype=float)
            
            self._from = ['A', 'B']
            self._to = [0, 1, 2, 3]

            const = 250
            satu = 5000
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
                    self.mu[i] += 0.9
        else:
            self.nmans = 4
            self.njobs = 7
            self.narms = 28
            self.nsuperarms = 840
            self.nlengths = 4
            self.sigma = 0.01
            self.initial =      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
            

            self._from = ['A', 'B', 'C', 'D']
            self._to = [0, 1, 2, 3, 4, 5, 6]

            
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

        self.G.add_nodes_from(self._from, bipartite=0)
        self.G.add_nodes_from(self._to, bipartite=1)
        
        for i in range(self.nmans):
            for j in range(self.njobs):
                self.G.add_edge(self._from[i], self._to[j], weight = 0)

        self.superarm = np.zeros((self.nsuperarms, self.nmans), dtype=int)
                
    
    def set_superarm(self):
        if self.easy:
            nums = [0, 1, 2, 3]
            c = np.array(np.meshgrid(nums, nums), dtype=int).T.reshape(-1, 2)

            self.superarm = np.delete(c, np.where((c[:,0]==c[:,1]))[0], axis=0)
        else:
            nums = [0, 1, 2, 3, 4, 5, 6]
            c = np.array(np.meshgrid(nums, nums, nums, nums), dtype=int).T.reshape(-1, 4)

            self.superarm = np.delete(c, np.where((c[:,0]==c[:,1])|(c[:,0]==c[:,2])|(c[:,0]==c[:,3])|(c[:,1]==c[:,2])|(c[:,1]==c[:,3])|(c[:,2]==c[:,3]))[0], axis=0)
        return
    
    def rewards(self):
        np.random.seed(self.seed)
        return np.clip(self.mu + self.sigma * np.random.randn(self.narms, self.T), 0, 1)
    
    def update(self, rewards):
        for i in range(self.narms):
            self.G[self._from[int(i/self.njobs)]][self._to[int(i%self.njobs)]]['weight'] = -rewards[i]
        return
    
    def get_optimal_action_candi(self):
        if self.easy:
            a = np.array([[1, 6]])
            b = np.array([[0, 5]])
        else:
            a = np.array([[1, 9, 17, 25]])
            b = np.array([[0, 8, 16, 24]])
        return np.concatenate((a, b))
    
    def random_i(self, i):
        man = int(i/self.njobs)
        job = int(i%self.njobs)
        
        mans = np.delete(np.array(range(self.nmans)), man)
        jobs = np.random.choice(np.delete(np.array(range(self.njobs)), job), self.nmans-1, replace=False)
        
        return np.sort(np.append(mans*self.njobs + jobs, i))
    
    def best(self):
        _max = nx.bipartite.minimum_weight_full_matching(self.G, weight='weight')
        cnt = 0
        best = []
        
        for i in self._from:
            best.append(cnt+_max[i])
            cnt += self.njobs
            
        return best
        
    