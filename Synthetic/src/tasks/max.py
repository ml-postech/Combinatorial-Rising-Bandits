import numpy as np

class Max(object):
    def __init__(self, seed, T, easy = False):
        self.seed = seed
        self.T = T
        self.easy = easy

        if self.easy:
            self.nmans = 1
            self.njobs = 5
            self.narms = 5
            self.nsuperarms = 10
            self.nlengths = 2
            self.sigma = 0.01

        const = 250
        satu = 5000
        self.indices = np.repeat(np.arange(const+1, self.T+2, const),const)
        self.inverse = self.indices ** -1.2
        self.inverse[satu:] = 0
        self.mu = np.zeros((self.narms, self.T), dtype=float)


        for j in range(1, self.T):
            self.mu[0][j] = self.mu[0][j-1] + self.inverse[j-1]
        self.mu[1][0] = 0.3
        for j in range(1, self.T):
            self.mu[1][j] = self.mu[1][j-1] + self.inverse[j-1] * 0.5
        self.mu[2][0] = 0.5
        for j in range(1, self.T):
            self.mu[2][j] = self.mu[2][j-1] + self.inverse[j-1] * 0.1
        self.mu[3][0] = 0.7
        for j in range(1, self.T):
            self.mu[3][j] = self.mu[3][j-1]
        self.mu[4][0] = 0.9
        for j in range(1, self.T):
            self.mu[4][j] = self.mu[4][j-1]

        
        self._mu = np.zeros((self.narms), dtype=float)

    def set_superarm(self):
        if self.easy:
            self.superarm = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
        return
    
    def rewards(self):
        np.random.seed(self.seed)
        return np.clip(self.mu + self.sigma * np.random.randn(self.narms, self.T), 0, 1)
    
    def update(self, rewards):
        for i in range(self.narms):
            self._mu[int(i)] = rewards[i]
        return
    
    def get_optimal_action_candi(self):
        if self.easy:
            a = np.array([[0, 4]])
        return a
    
    def random_i(self, i):
        ran = np.random.choice(np.delete(np.array(range(self.narms)), i), 1, replace = False)
        
        return np.append(ran, i)
    
    def best(self):
        # top-k
        _max = np.argsort(self._mu)[-2:][::-1]
        
        
        # # max+rand
        # _max = np.argsort(self._mu)[-1:][::-1]
        # _max = _max[0] 

        # ran = np.random.choice(np.delete(np.array(range(self.narms)), _max), 1, replace=False)
        # _max = np.concatenate((np.array([_max]), ran))

        return _max.astype(int)
        
     