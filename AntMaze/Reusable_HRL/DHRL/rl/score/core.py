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
    def __init__(self, input_dim, output_dim, lr, args, use_ag_as_input=False):
        self.predictor = RndPredictor(input_dim, output_dim)
        self.predictor_target = RndPredictor(input_dim, output_dim)
        self.cuda_num = args.cuda_num

        if torch.cuda.is_available():
            self.predictor = self.predictor.to(device=self.cuda_num)
            self.predictor_target = self.predictor_target.to(device=self.cuda_num)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.use_ag_as_input = use_ag_as_input

    def get_novelty(self, input):

        with torch.no_grad():
            target_feature = self.predictor_target(input)
            feature = self.predictor(input)
            novelty = (feature - target_feature).pow(2).sum(1).unsqueeze(1) / 2

        return novelty

    def train(self, replay_buffer, iterations, batch_size=128, rms=None, goal_rms = None, action_rms=None):
        for it in range(iterations):
            # Sample replay buffer
            batch = replay_buffer.sample_regular_batch(batch_size)

            input = batch['ob'] if not self.use_ag_as_input else batch['ag']
            goal_input = batch['bg']
            action_input = batch['a']
            if rms is not None:
                input = (input - rms.mean) / np.sqrt(rms.var)
            if goal_rms is not None:
                goal_input = (goal_input - goal_rms.mean) / np.sqrt(goal_rms.var)
            if action_rms is not None:
                action_input = (action_input - action_rms.mean) / np.sqrt(action_rms.var)

            input = get_tensor(input)
            goal_input = get_tensor(goal_input)
            action_input = get_tensor(action_input)

            input = torch.cat([input, goal_input, action_input], dim=-1)

            with torch.no_grad():
                target_feature = self.predictor_target(input)
            feature = self.predictor(input)
            loss = (feature - target_feature).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
    
    def rnd_loss(self, input):
        with torch.no_grad():
            target_feature = self.predictor_target(input)
        feature = self.predictor(input)
        loss = (feature - target_feature).pow(2).mean()
        # loss = (feature - target_feature).pow(2).mean(-1)
        # mask = torch.rand(len(loss)).to(device=self.cuda_num)
        # mask = (mask < 0.5).type(torch.FloatTensor).to(device=self.cuda_num)
        # loss = (loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device=self.cuda_num))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count