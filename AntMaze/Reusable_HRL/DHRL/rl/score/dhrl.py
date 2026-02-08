import numpy as np
import torch 
import os.path as osp

from rl.score.core import RandomNetworkDistillation, RunningMeanStd


class LowScore:
    def __init__(self, env_params, args, agent, monitor, name='low_score'):
        self.method = args.low_score                   
        self.args = args
        self.agent = agent
        self.monitor = monitor
        self.env_params = env_params
        goal_input_dim = env_params['goal']
        obs_input_dim = env_params['obs']
        action_input_dim = env_params['l_action_dim']
        self._save_file = str(name) + '.pt'
        #For low-level policy score
        self.action_rms = RunningMeanStd(shape=(action_input_dim))
        self.goal_rms = RunningMeanStd(shape=(goal_input_dim))
        self.score_rms = RunningMeanStd()
        if args.low_input == 'goal':
            self.rms = RunningMeanStd(shape=(goal_input_dim))
            score_input_dim = goal_input_dim + goal_input_dim + action_input_dim
            self.use_ag_as_input = True
        else:
            self.rms = RunningMeanStd(shape=(obs_input_dim))
            score_input_dim = obs_input_dim + goal_input_dim + action_input_dim
            self.use_ag_as_input = False
            
        if self.method == 'RND':
            self.RND_score = RandomNetworkDistillation(input_dim=score_input_dim, output_dim=128, lr=1e-3, use_ag_as_input=self.use_ag_as_input)

    def get_batch_score(self, batch, eval=False):
        obs = batch['wp']
        goal = batch['bg']
        action = self.agent.get_actions(obs, goal)

        if self.method == 'RND':
            if self.args.low_input == 'goal':
                obs = batch['wp'][:,:self.args.subgoal_dim]
            if self.args.input_normalization:
                score = self.RND_score.get_novelty(obs, goal, action, self.rms, self.goal_rms, self.action_rms)
            else:
                score = self.RND_score.get_novelty(obs, goal, action)

        elif self.method == 'MC_Dropout':
            self.agent.train()
            predictions = torch.zeros((30, obs.shape[0]))
            for i in range(30):
                value = self.agent.get_qs(obs, goal, action)
                predictions[i,:] = value.squeeze()
            score = predictions.var(dim=0)

        final_score = score.detach().cpu().numpy()
        if self.args.score_normalization and not eval:
            self.score_rms.update(final_score)
        final_score = (final_score - self.score_rms.mean) / np.sqrt(self.score_rms.var)
        return final_score
    
    def update_RND(self, batch):
        input = batch['ob'] if not self.use_ag_as_input else batch['ag']
        goal_input = batch['bg']
        action_input = batch['a']
        if self.args.input_normalization:
            self.rms.update(input)
            self.goal_rms.update(goal_input)
            self.action_rms.update(action_input)
            input = (input - self.rms.mean) / np.sqrt(self.rms.var)
            goal_input = (goal_input - self.goal_rms.mean) / np.sqrt(self.goal_rms.var)
            action_input = (action_input - self.action_rms.mean) / np.sqrt(self.action_rms.var)

        input = self.agent.to_tensor(input)
        goal_input = self.agent.to_tensor(goal_input)
        action_input = self.agent.to_tensor(action_input)
        input = torch.cat([input, goal_input, action_input], dim=-1)

        RND_loss = self.RND_score.rnd_loss(input)
        self.monitor.store(rnd_loss_low=RND_loss)

    def train(self, replay_buffer, iterations, batch_size):
        low_rnd_loss = self.RND_score.train(replay_buffer, iterations, batch_size)
        return low_rnd_loss

    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )

    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']

class HighScore:
    def __init__(self, env_params, args, agent, monitor, name='high_score'):
        self.method = args.high_score            
        self.args = args
        self.agent = agent
        self.env_params = env_params
        self.monitor = monitor
        goal_input_dim = env_params['goal']
        obs_input_dim = env_params['obs']
        self._save_file = str(name) + '.pt'
        # For high-level policy score
        self.action_rms = RunningMeanStd(shape=(env_params['h_action_dim']))
        self.goal_rms = RunningMeanStd(shape=(goal_input_dim))
        self.score_rms = RunningMeanStd()
        if args.high_score_input == 'goal':
            self.rms = RunningMeanStd(shape=(goal_input_dim))
            score_input_dim = goal_input_dim + goal_input_dim + goal_input_dim
            #score_input_dim = goal_input_dim + goal_input_dim
            self.use_ag_as_input = True
        else:
            self.rms = RunningMeanStd(shape=(obs_input_dim))
            score_input_dim = obs_input_dim + goal_input_dim + goal_input_dim
            #score_input_dim = obs_input_dim + goal_input_dim
            self.use_ag_as_input = False
            
        if self.method == 'RND':
            self.RND_score = RandomNetworkDistillation(input_dim=score_input_dim, output_dim=128, lr=5e-5, args=args, use_ag_as_input=self.use_ag_as_input)

    def get_batch_score(self, batch, eval=False):

        input = batch['ob'] if not self.args.high_score_input == 'goal' else batch['ag']
        goal_input = batch['bg']
        action_input = batch['a']
        #action = batch['a']
        if self.args.input_normalization:
            self.rms.update(input)
            self.goal_rms.update(goal_input)
            self.action_rms.update(action_input)
            input = (input - self.rms.mean) / np.sqrt(self.rms.var)
            goal_input = (goal_input - self.goal_rms.mean) / np.sqrt(self.goal_rms.var)
            action_input = (action_input - self.action_rms.mean) / np.sqrt(self.action_rms.var)

        input = self.agent.to_tensor(input)
        goal_input = self.agent.to_tensor(goal_input)
        action_input = self.agent.to_tensor(action_input)
        input = torch.cat([input, goal_input, action_input], dim=-1)
        #input = torch.cat([input, goal_input], dim=-1)
                          
        if self.method == 'RND':
            score=self.RND_score.get_novelty(input)

        final_score = score.detach().cpu().numpy()
        if self.args.score_normalization:
            self.score_rms.update(final_score)
            #final_score = (final_score - self.score_rms.mean) / np.sqrt(self.score_rms.var)
            final_score = final_score / np.sqrt(self.score_rms.var)
        return final_score
    
    def update_RND(self, batch):
        input = batch['ob'] if not self.use_ag_as_input else batch['ag']
        goal_input = batch['bg']
        action_input = batch['a']
        if self.args.input_normalization:
            self.rms.update(input)
            self.goal_rms.update(goal_input)
            self.action_rms.update(action_input)
            input = (input - self.rms.mean) / np.sqrt(self.rms.var)
            goal_input = (goal_input - self.goal_rms.mean) / np.sqrt(self.goal_rms.var)
            action_input = (action_input - self.action_rms.mean) / np.sqrt(self.action_rms.var)

        input = self.agent.to_tensor(input)
        goal_input = self.agent.to_tensor(goal_input)
        action_input = self.agent.to_tensor(action_input)
        input = torch.cat([input, goal_input, action_input], dim=-1)
        #input = torch.cat([input, goal_input], dim=-1)

        loss=self.RND_score.rnd_loss(input)
        self.monitor.store(rnd_loss_high=loss)

    def train(self, replay_buffer, iterations, batch_size):
        if self.args.input_normalization:
            high_rnd_loss = self.RND_score.train(replay_buffer, iterations, batch_size, self.rms, self.goal_rms, self.action_rms)
        else:
            high_rnd_loss = self.RND_score.train(replay_buffer, iterations, batch_size)
        return high_rnd_loss
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
        
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']

    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)