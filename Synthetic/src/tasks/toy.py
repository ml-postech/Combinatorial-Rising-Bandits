import numpy as np
import networkx as nx

class Toy(object):
    def __init__(self, seed, T, easy = False):
        self.seed = seed
        self.T = T

        self.nnodes = 5
        self.narms = 5
        self.nsuperarms = 2
        self.nlengths = 3
        self.sigma = 0.00
        self.l_initial = 1.0
        self.e_initial = 0.0

        const = 1000
        satu = 500000
        self.indices = np.repeat(np.arange(const+1, self.T+2, const),const)
        self.inverse = self.indices ** -1.2
        self.inverse[satu:] = 0
        self.mu = np.zeros((self.narms, self.T), dtype=float)
        
        const = 0.9
        
        self.mu[2][0] = const
        
        for j in range(1, self.T):
            self.mu[0][j] = self.mu[0][j-1] + self.inverse[j-1]
            if j < 5000:
                self.mu[1][j] = j/5000.0 * 1.0
            else:
                self.mu[1][j] = 1.0
            self.mu[2][j] = const
        
        if self.mu[0][self.T-1] > 1:
            self.mu[0] = self.mu[0] / self.mu[0][self.T-1] 

        self.G = nx.DiGraph()

        self._from_to = np.zeros((self.narms, 2), dtype=int)
        
        self._from_to[0][0] = 0
        self._from_to[0][1] = 1
        self._from_to[1][0] = 1
        self._from_to[1][1] = 2
        self._from_to[2][0] = 1
        self._from_to[2][1] = 3
        self._from_to[3][0] = 2
        self._from_to[3][1] = 4
        self._from_to[4][0] = 3
        self._from_to[4][1] = 4

        for i in range(self.narms):
            self.G.add_edge(self._from_to[i][0], self._from_to[i][1], weight = 0)

        self.superarm = np.zeros((self.nsuperarms, self.nlengths), dtype=int)
    
    def set_superarm(self):
        cnt = 0    
        for path in nx.all_simple_paths(self.G, source=0, target=self.nnodes-1):
            path = np.array(path, dtype=int)
            superarm = np.where((self._from_to == path[0:2]).all(axis=1))[0]
            for i in range(1, self.nlengths):
                superarm = np.concatenate((superarm, np.where((self._from_to == path[i: i+2]).all(axis=1))[0]), axis=0)
            self.superarm[cnt] = superarm
            cnt += 1
        return
    
    def rewards(self):
        np.random.seed(self.seed)
        return np.clip(self.mu + self.sigma * np.random.randn(self.narms, self.T), 0, 1)
    
    def update(self, rewards):
        for i in range(self.narms):
            self.G[self._from_to[i][0]][self._from_to[i][1]]['weight'] = -rewards[i]
        return
    
    def get_optimal_action_candi(self):
        a = np.array([[0, 1, 3]])
        b = np.array([[0, 2, 4]])
        return np.concatenate((a, b))
    
    def random_i(self, i):
        return self.superarm[np.random.choice(np.where(self.superarm == i)[0])]
    
    def best(self):
        path = nx.shortest_path(self.G, 0, self.nnodes-1, weight='weight')
        best = np.where((self._from_to == path[0:2]).all(axis=1))[0]
        for i in range(1, self.nlengths):
            best = np.concatenate((best, np.where((self._from_to == path[i: i+2]).all(axis=1))[0]), axis=0)
        
        return best.astype(int)
    