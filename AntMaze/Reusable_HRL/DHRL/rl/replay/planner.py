import threading
import math
import torch
import os.path as osp
import numpy as np
from sklearn import mixture
from scipy.stats import rankdata
from rl.replay.her_algo import sample_bher_transitions, sample_archer_transitions, sample_cher_transitions, sample_her_transitions_with_subgoaltesting_high, sample_mep_transitions, sample_her_transitions_with_subgoaltesting_original, sample_her_transitions_with_subgoaltesting_gbphrl


def sample_her_transitions(buffer, reward_func, batch_size, future_step, future_p=1.0):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    not_her_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    
    future_t = (t_samples + 1 + future_offset)
    
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    
    batch['offset'] = future_offset.copy()
    
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_her_transitions_grid(buffer, reward_func, batch_size, future_step, future_p = 1.0, movement_pen = 1.0, movement_threshold = 0.5, future_offset_threshold = 30.):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    not_her_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    
    future_t = (t_samples + 1 + future_offset)
    
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    
    batch['offset'] = future_offset.copy()
    
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    
    dist = batch['ag'][not_her_indexes] - buffer['ag'][ep_idxes[not_her_indexes], future_t[not_her_indexes]]
    dist2 = batch['bg'][not_her_indexes] - buffer['ag'][ep_idxes[not_her_indexes], future_t[not_her_indexes]]
    future_offset_test = future_t[not_her_indexes]
    
    movement_failure = not_her_indexes[0][np.where((dist < movement_threshold) & (dist2 > movement_threshold) & (future_offset_test > future_offset_threshold))]
    batch['r'][movement_failure] -= movement_pen
    
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def sample_her_transitions_with_subgoaltesting(buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, subgoaltest_threshold, monitor, gradual_pen):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    original_batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['origin_bg'] = batch['bg'].copy()
    batch['origin_a'] = batch['a'].copy()

    batch['bg'] = buffer['ag'][ep_idxes, future_t]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    dist = batch['a'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes]
    batch['a'][not_subgoaltesting_indexes] = batch['ag2'][not_subgoaltesting_indexes]
    
    dist = np.linalg.norm(dist, axis=1)
    subgoaltesting_failure = subgoaltesting_indexes[0][np.where(dist>subgoaltest_threshold)]


    penalty = 1.3
    batch['r'][subgoaltesting_failure] = - penalty
    
    if graphplanner.graph is not None:
        dist_2 = graphplanner.dist_from_graph_to_goal(batch['a'][subgoaltesting_indexes])
        monitor.store(distance_from_graph = np.mean(dist_2))
        subgoaltesting_failure_2 = subgoaltesting_indexes[0][np.where(dist_2>(cutoff*3))]
        batch['r'][subgoaltesting_failure_2] = - gradual_pen
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch



def sample_transitions(buffer, batch_size):
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    return batch


class LowReplay:
    def __init__(self, env_params, args, low_reward_func, agent=None, name='low_replay'):
        self.env_params = env_params
        self.args = args
        self.low_reward_func = low_reward_func
        self.agent = agent
        self.horizon = env_params['max_timesteps']
        self.size = args.buffer_size // self.horizon
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['sub_goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['sub_goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['l_action_dim'])))
        
        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'
    
    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             future_p=self.args.low_future_p)
        return transitions

    def sample_g(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             future_p=self.args.low_future_p_g)
        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
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




class HighReplay:
    def __init__(self, env_params, args, high_reward_func, monitor, low_score, high_score, agent=None, name='high_replay'):
        self.env_params = env_params
        self.args = args
        self.high_reward_func = high_reward_func
        self.monitor = monitor
        self.horizon = math.ceil(env_params['max_timesteps'] / args.subgoal_freq)
        self.size = args.buffer_size // self.horizon
        self.agent = agent
        self.low_score = low_score
        self.high_score = high_score
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['h_action_dim'])))
        
        self.buffers_for_adahind = dict(wp=np.zeros((self.size, self.horizon, self.env_params['obs'])))

        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'


    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            if self.args.ada_hindsight or self.args.add_loss:
                self.buffers_for_adahind['wp'][idxs] = episodes['wp'].copy()
            if self.args.high_hindsight == 'mep':
                if not isinstance(self.clf, int):
                    ag = self.buffers['ag']
                    X = ag.reshape(-1, ag.shape[1]*ag.shape[2])
                    pred = -self.clf.score_samples(X)

                    pred = pred - self.pred_min
                    pred = np.clip(pred, 0, None)
                    pred = pred / self.pred_sum

                    self.buffers['e'] = pred.reshape(-1,1)

                    entropy_transition_total = self.buffers_for_mep['e'][:self.current_size]
                    rank_method = 'dense'
                    
                    entropy_rank = rankdata(entropy_transition_total, method=rank_method)
                    entropy_rank = entropy_rank - 1
                    entropy_rank = entropy_rank.reshape(-1, 1)
                    # print(entropy_rank)
                    self.buffers_for_mep['p'][:self.current_size] = entropy_rank.copy()
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size, graphplanner):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
            if self.args.ada_hindsight or self.args.add_loss:
                for key2 in self.buffers_for_adahind.keys():
                    temp_buffers[key2] = self.buffers_for_adahind[key2][:self.current_size]
            if self.args.high_hindsight =='mep':
                for key3 in self.buffers_for_mep.keys():
                    temp_buffers[key3] = self.buffers_for_mep[key3][:self.current_size]
        if self.args.ada_hindsight:
            transitions = sample_her_transitions_with_subgoaltesting_high(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                                future_step=self.args.high_future_step,
                                                cutoff = self.args.cutoff,
                                                subgoaltest_p=self.args.subgoaltest_p,
                                                subgoaltest_threshold = self.args.subgoaltest_threshold,
                                                monitor = self.monitor,
                                                gradual_pen= self.args.gradual_pen,
                                                score=self.high_score,
                                                agent=self.agent,
                                                method = self.args.high_score,
                                                epsilon = self.args.epsilon
            )
        elif self.args.high_hindsight == 'bher':
            transitions = sample_bher_transitions(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                                future_step=self.args.high_future_step,
                                                agent=self.agent,
                                                monitor=self.monitor
            )
        elif self.args.high_hindsight == 'archer':
            transitions = sample_archer_transitions(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                                future_step=self.args.high_future_step,
                                                agent=self.agent,
                                                monitor=self.monitor
            )
        elif self.args.high_hindsight == 'cher':
            transitions = sample_cher_transitions(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                                future_step=self.args.high_future_step,
                                                agent=self.agent,
                                                monitor=self.monitor
            )
        elif self.args.high_hindsight == 'mep':
            transitions = sample_mep_transitions(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                                future_step=self.args.high_future_step,
                                                monitor=self.monitor)

        else:
            if (self.args.method == 'dhrl') or (self.args.method == 'grid') or (self.args.method == 'grid8') or (self.args.method == 'grid_complex') or (self.args.method == 'custom') or (self.args.method == 'custom_complex'):
                transitions = sample_her_transitions_with_subgoaltesting(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                        future_step=self.args.high_future_step,
                                        cutoff = self.args.cutoff,
                                        subgoaltest_p=self.args.subgoaltest_p,
                                        subgoaltest_threshold = self.args.subgoaltest_threshold,
                                        monitor = self.monitor,
                                        gradual_pen= self.args.gradual_pen)
                
            elif (self.args.method == 'gbphrl') or (self.args.method == 'value'):
                if self.args.ada_score:
                    transitions = sample_her_transitions_with_subgoaltesting_original(temp_buffers, self.high_reward_func, batch_size,
                                        future_step=self.args.high_future_step,
                                        subgoaltest_p=self.args.subgoaltest_p,
                                        future_p=self.args.high_future_p,
                                        subgoaltest_threshold = self.args.subgoaltest_threshold,
                                        monitor = self.monitor,
                                        high_penalty=self.args.high_penalty,
                                        high_score=self.high_score)
                else:
                    transitions = sample_her_transitions_with_subgoaltesting_gbphrl(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                        future_step=self.args.high_future_step,
                                        cutoff = self.args.cutoff,
                                        subgoaltest_p=self.args.subgoaltest_p,
                                        future_p=self.args.high_future_p,
                                        subgoaltest_threshold = self.args.subgoaltest_threshold,
                                        high_penalty=self.args.high_penalty)


        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
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
