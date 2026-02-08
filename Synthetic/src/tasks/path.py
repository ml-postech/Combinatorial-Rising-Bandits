import numpy as np
import networkx as nx

class Path(object):
    def __init__(self, seed, T, easy = False):
        self.seed = seed
        self.T = T
        self.easy = easy

        if self.easy:
            # 0 - 1 - 2
            # |   |   |
            # 3 - 4 - 5
            # |   |   |
            # 6 - 7 - 8

            
            # @ 0 @ 1 @
            # 2   3   4
            # @ 5 @ 6 @
            # 7   8   9
            # @ 10@ 11@ 
            # >>VV
            # 0, 1, 4, 9
            # 2, 7, 10, 11
            self.size = 3
            self.nnodes = 9
            self.narms = 12
            self.nsuperarms = 6
            self.nlengths = 4
            self.sigma = 0.01
            self.l_initial = 1.0
            self.e_initial = 0.0

            self.initial =      np.array([  self.e_initial,    self.e_initial,      
                                        self.l_initial,    self.e_initial,    self.e_initial,     
                                            self.l_initial,    self.l_initial,      
                                        self.e_initial,    self.e_initial,    self.l_initial,  
                                            self.e_initial,    self.e_initial,     
                                        ], dtype=float)

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
            # 0 - 1 - 2 - 3 - 4 - 5
            # |   |   |   |   |   |
            # 6 - 7 - 8 - 9 -10 -11
            # |   |   |   |   |   |
            #12 -13 -14 -15 -16 -17
            # |   |   |   |   |   |
            #18 -19 -20 -21 -22 -23
            # |   |   |   |   |   |
            #24 -25 -26 -27 -28 -29
            # |   |   |   |   |   |
            #30 -31 -32 -33 -34 -35

            # @ 0 @ 1 @ 2 @ 3 @ 4 @
            # 5   6   7   8   9  10
            # @11 @12 @13 @14 @15 @ 
            #16  17  18  19  20  21
            # @22 @23 @24 @25 @26 @
            #27  28  29  30  31  32
            # @33 @34 @35 @36 @37 @
            #38  38  40  41  42  43
            # @44 @45 @46 @47 @48 @
            #49  50  51  52  53  54
            # @55 @56 @57 @58 @59 @
            # VVV>>>>VV>
            # 5, 16, 27, 33, 34, 35, 36, 42, 53, 59 
            self.size = 6
            self.nnodes = 36
            self.narms = 60
            self.nsuperarms = 252
            self.nlengths = 10
            self.sigma = 0.01
            
            self.l_initial = 1.0
            self.e_initial = 0.0

            self.initial =      np.array([    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,
                                        self.l_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,
                                            self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                        self.l_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                            self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                        self.l_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                            self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                        self.l_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                            self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,
                                        self.l_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial,    self.e_initial, 
                                            self.l_initial,    self.l_initial,    self.l_initial,    self.l_initial,    self.l_initial,    
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

        self.G = nx.DiGraph()

        self._from_to = np.zeros((self.narms, 2), dtype=int)
        
        cnt = 0
        for i in range(self.size):
            for j in range(self.size-1):
                self._from_to[cnt][0] = self.size*i+j
                self._from_to[cnt][1] = self.size*i+j+1
                cnt += 1
            
            if i != self.size-1:
                for j in range(self.size):
                    self._from_to[cnt][0] = self.size*i+j
                    self._from_to[cnt][1] = self.size*(i+1)+j
                    cnt += 1

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
        if self.easy:
            a = np.array([[2, 5, 6, 9]])
            b = np.array([[0, 3, 8, 11]])
        else:
            a = np.array([[0, 1, 2, 3, 4, 10, 21, 32, 43, 54]])
            b = np.array([[5, 16, 27, 38, 49, 55, 56, 57, 58, 59]])
        return np.concatenate((a, b))
    
    def random_i(self, i):
        return self.superarm[np.random.choice(np.where(self.superarm == i)[0])]
    
    def best(self):
        path = nx.shortest_path(self.G, 0, self.nnodes-1, weight='weight')
        best = np.where((self._from_to == path[0:2]).all(axis=1))[0]
        for i in range(1, self.nlengths):
            best = np.concatenate((best, np.where((self._from_to == path[i: i+2]).all(axis=1))[0]), axis=0)
        
        return best.astype(int)
    