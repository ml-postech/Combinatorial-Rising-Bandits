import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)

def get_tensor(z):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))

class RndPredictor(nn.Module):
    def __init__(self, state_dim, hidden_dim=300, output_dim=128):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1
    
class RandomNetworkDistillation(object):
    def __init__(self, input_dim, output_dim, lr, use_ag_as_input=False):
        self.predictor = RndPredictor(input_dim, output_dim)
        self.predictor_target = RndPredictor(input_dim, output_dim)

        if torch.cuda.is_available():
            self.predictor = self.predictor.to(device)
            self.predictor_target = self.predictor_target.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.use_ag_as_input = use_ag_as_input

    def get_novelty(self, obs):
        obs = get_tensor(obs)
        with torch.no_grad():
            target_feature = self.predictor_target(obs)
            feature = self.predictor(obs)
            novelty = (feature - target_feature).pow(2).sum(1).unsqueeze(1) / 2
        return novelty

    def train(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer
            batch = replay_buffer.sample(batch_size)
            
            #input = batch['ob']
            input = batch['o2'] if not self.use_ag_as_input else batch['ag']
            input = get_tensor(input)

            with torch.no_grad():
                target_feature = self.predictor_target(input)
            feature = self.predictor(input)
            loss = (feature - target_feature).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss