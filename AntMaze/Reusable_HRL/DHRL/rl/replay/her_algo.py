import io
import threading
import math
import copy
import torch
import random
import rl.replay.cher_config as config_cur
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

def sample_her_transitions_with_subgoaltesting_gbphrl(buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, future_p, high_penalty, subgoaltest_threshold, high_score=None, movement_penalty = 1.0, movement_threshold = 0.5):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 

    batch['origin_bg'] = batch['bg'].copy()
    batch['origin_a'] = batch['a'].copy()

    uni = np.random.uniform(size = batch_size)
    
    her_indexes = np.where(uni < future_p)
    not_her_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    # subgoaltesting_indexes = np.where(uni < subgoaltest_p)
    # not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    dist = batch['a'][not_her_indexes] - batch['ag2'][not_her_indexes]
    batch['a'][her_indexes] = batch['ag2'][her_indexes]
    
    dist = np.linalg.norm(dist, axis=1)
    
    subgoaltesting_failure = not_her_indexes[np.where(dist>subgoaltest_threshold)]
    penalty = high_penalty
    batch['r'][subgoaltesting_failure] -= penalty
    
    # dist2 = batch['ag'][not_her_indexes] - batch['ag2'][not_her_indexes]
    
    # movement_failure = not_her_indexes[np.where(dist2 < movement_threshold)[0]]
    
    # dist3 = batch['ag'][movement_failure] - batch['bg'][movement_failure]
    
    # movement_failure = movement_failure[np.where(dist3 > movement_threshold)[0]]
    
    # batch['r'][movement_failure] -= movement_penalty
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def sample_her_transitions_with_subgoaltesting_original(buffer, reward_func, batch_size, future_step, subgoaltest_p, future_p, subgoaltest_threshold, monitor, high_penalty, high_score=None):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    original_batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['origin_bg'] = batch['bg'].copy()
    batch['origin_a'] = batch['a'].copy()
    
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    dist = batch['a'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes]
    batch['a'][not_subgoaltesting_indexes] = batch['ag2'][not_subgoaltesting_indexes]
    dist = np.linalg.norm(dist, axis=1)
    subgoaltesting_failure = subgoaltesting_indexes[0][np.where(dist>subgoaltest_threshold)]

    dist2 = batch['ag'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes]
    dist2 = np.linalg.norm(dist2, axis=1)
    not_moving = subgoaltesting_indexes[0][np.where(dist2<1)]
    not_arriving = subgoaltesting_indexes[0][np.where(batch['r'][subgoaltesting_indexes].reshape(-1)==-1)]
    moving_failure = np.intersect1d(not_moving, not_arriving)

    penalty = high_penalty
    batch['r'][moving_failure] -= penalty
    batch['r'][subgoaltesting_failure] = - penalty
    
    if high_score is not None:
        bonus = high_score.get_batch_score(batch)
        monitor.store(bonus_mean = np.mean(bonus))
        clipped_bonus = np.clip(bonus, 0, 1)
        monitor.store(cliped_bonus_mean = np.mean(clipped_bonus))
        #batch['r'] += np.clip(bonus, 0, 1)
        #batch['r'][subgoaltesting_indexes] += bonus[subgoaltesting_indexes]
        batch['r'][subgoaltesting_indexes] += clipped_bonus[subgoaltesting_indexes]

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_bher_transitions(buffer, reward_func, batch_size, graphplanner, future_step, agent, monitor):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]

    # Select which episodes and time steps to use.
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    original_batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    # lky
    batch['origin_g'] = batch['bg'].copy()

    future_p = 1.1
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    other_indexes = np.delete(np.arange(batch_size), her_indexes)

    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    batch['offset'] = future_offset.copy()
    batch['a'][her_indexes] = batch['ag2'][her_indexes]

    action = agent.get_actions(batch['ob'][her_indexes], batch['origin_g'][her_indexes])
    action_her = agent.get_actions(batch['ob'][her_indexes], batch['bg'][her_indexes])
    bias = 0.002 
    reward_bias = bias * (np.square(np.linalg.norm(action - batch['a'][her_indexes], axis=1)) - np.square(np.linalg.norm(action_her - batch['a'][her_indexes], axis=1)))
    exp_reward_bias = np.exp(reward_bias)
    exp_reward_bias = np.clip(exp_reward_bias, 0, 10)
    print(np.mean(exp_reward_bias))
    if future_p > 1:
        batch['r'][her_indexes] *= np.expand_dims(exp_reward_bias, axis=1)
    else:
        exp_reward_bias_mean = np.mean(exp_reward_bias)
        batch['r'][other_indexes] /= exp_reward_bias_mean

    return batch


def sample_archer_transitions(buffer, reward_func, batch_size, graphplanner, future_step, agent, monitor):
    # do not apply goal hindsight when subgoal testing
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    future_p = 0.8
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    other_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    batch['offset'] = future_offset.copy()
    batch['a'][her_indexes] = batch['ag2'][her_indexes]
    
    batch['r'][her_indexes] *= 0.5
    batch['r'][other_indexes] *= 2

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def curriculum(transitions, batch_size_in_transitions):
    sel_list = lazier_and_goals_sample_kg(
        transitions['bg'], transitions['ag'], transitions['ob'],
        batch_size_in_transitions)
    transitions = {
        key: transitions[key][sel_list].copy()
        for key in transitions.keys()
    }
    config_cur.learning_step += 1
    return transitions

def fa(k, a_set, v_set, sim, row, col):
    if len(a_set) == 0:
        init_a_set = []
        marginal_v = 0
        for i in v_set:
            max_ki = 0
            if k == col[i]:
                max_ki = sim[i]
            init_a_set.append(max_ki)
            marginal_v += max_ki
        return marginal_v, init_a_set

    new_a_set = []
    marginal_v = 0
    for i in v_set:
        sim_ik = 0
        if k == col[i]:
            sim_ik = sim[i]

        if sim_ik > a_set[i]:
            max_ki = sim_ik
            new_a_set.append(max_ki)
            marginal_v += max_ki - a_set[i]
        else:
            new_a_set.append(a_set[i])
    return marginal_v, new_a_set

def lazier_and_goals_sample_kg(goals, ac_goals, obs,
                                batch_size_in_transitions):
    if config_cur.goal_type == "ROTATION":
        goals, ac_goals = goals[..., 3:], ac_goals[..., 3:]

    num_neighbor = 1
    kgraph = NearestNeighbors(
        n_neighbors=num_neighbor, algorithm='kd_tree',
        metric='euclidean').fit(goals).kneighbors_graph(
            mode='distance').tocoo(copy=False)
    row = kgraph.row
    col = kgraph.col
    sim = np.exp(
        -np.divide(np.power(kgraph.data, 2),
                    np.mean(kgraph.data)**2))
    delta = np.mean(kgraph.data)

    sel_idx_set = []
    idx_set = [i for i in range(len(goals))]
    balance = config_cur.fixed_lambda
    if int(balance) == -1:
        balance = math.pow(
            1 + config_cur.learning_rate,
            config_cur.learning_step) * config_cur.lambda_starter
    v_set = [i for i in range(len(goals))]
    max_set = []
    for i in range(0, batch_size_in_transitions):
        sub_size = 3
        sub_set = random.sample(idx_set, sub_size)
        sel_idx = -1
        max_marginal = float("-inf")  #-1 may have an issue
        for j in range(sub_size):
            k_idx = sub_set[j]
            marginal_v, new_a_set = fa(k_idx, max_set, v_set, sim, row,
                                        col)
            euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])
            marginal_v = marginal_v - balance * euc
            if marginal_v > max_marginal:
                sel_idx = k_idx
                max_marginal = marginal_v
                max_set = new_a_set

        idx_set.remove(sel_idx)
        sel_idx_set.append(sel_idx)
    return np.array(sel_idx_set)


# does not use it
def get_distance(p, init_set):
    dist = 0.
    for i in range(len(init_set)):
        dist += np.linalg.norm(p - init_set[i])
    return dist

def sample_cher_transitions(buffer, reward_func, batch_size, graphplanner, future_step, agent, monitor):
    """episode_batch is {key: array(buffer_size x T x dim_key)}
    """
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    future_p = 0.8
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    other_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['a'][her_indexes] = batch['ag2'][her_indexes]

    #assert batch_size_in_transitions == 64
    if batch_size != config_cur.learning_selected:
        batch_size = config_cur.learning_selected

    # curriculum learning process
    batch = curriculum(batch, batch_size)

    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def sample_her_transitions_with_subgoaltesting_high(buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, subgoaltest_threshold, score, agent, method, epsilon, monitor, gradual_pen):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]

    epsilon = epsilon
    
    if batch_size > n_trajs*horizon:
        batch_size = n_trajs*horizon

    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    
    # if method == 'RND':
    #     RND_batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    #     highscore = score.get_batch_score(RND_batch).reshape(-1)

    # elif method == 'Dist':
    #     highscore = agent._get_p2p_dist(buffer['ob'][:,:-1,:], buffer['a'][:,:,:]).reshape(-1)

    # candi = np.argpartition(highscore,-(int(batch_size*epsilon)))[-(int(batch_size*epsilon)):]

    # #t_samples = np.random.randint(0, horizon, size=batch_size)
    # ep_idxes[:(int(batch_size*epsilon))] = candi // horizon
    # t_samples[:(int(batch_size*epsilon))]= candi %  horizon
    
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)
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

def sample_mep_transitions(buffer, reward_func, batch_size, graphplanner, future_step, monitor, rank_method='none', temperature=1.0, update_stats=False):

    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)

    if not update_stats:

        if rank_method == 'none':
            entropy_trajectory = buffer['e']
        else:
            entropy_trajectory = buffer['p']
        # print('e',entropy_trajectory) 
        p_trajectory = np.power(entropy_trajectory, 1/(temperature+1e-2))
        p_trajectory = p_trajectory / p_trajectory.sum()
        # print('p',p_trajectory.flatten())
        episode_idxs_entropy = np.random.choice(n_trajs, size=batch_size, replace=True, p=p_trajectory.flatten())
        ep_idxes = episode_idxs_entropy


    batch = {}
    for key in buffer.keys():
        if not key =='p' and not key == 's' and not key == 'e':
            batch[key] = buffer[key][ep_idxes, t_samples].copy()

    # Select future time indexes proportional with probability future_p. These
    # will be used for HER replay by substituting in future goals.
    future_p = 0.8
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    other_indexes = np.delete(np.arange(batch_size), her_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    batch['a'][her_indexes] = batch['ag2'][her_indexes]

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def visualize_transition_data(original_batch, batch, graphplanner):
    map_size = [-4, 20]
    if graphplanner.env.env_name == 'AntMaze':
        # -4~ 20
        map_size = [-4, 20]
        Map_x, Map_y = (24, 24)
        start_x, start_y = (4, 4)
    elif graphplanner.env.env_name == 'AntMazeSmall-v0':
        # -2 ~ 12
        map_size = [-2, 12]
        Map_x, Map_y = (12, 12)
        start_x, start_y = (2,2)
    elif graphplanner.env.env_name == 'AntMazeS':
            map_size = [-4, 36]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 4)
    elif graphplanner.env.env_name == 'AntMazeW':
            map_size_x = [-4, 36]
            map_size_y = [-12, 28]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 12)
    elif graphplanner.env.env_name == 'AntMazeP':
            map_size_x = [-12, 28]
            map_size_y = [-4, 36]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (12, 4)
    elif graphplanner.env.env_name == 'AntMazeBottleneck':
        map_size = [-4, 20]
        Map_x, Map_y = (24, 24)
        start_x, start_y = (4, 4)       
    elif graphplanner.env.env_name == 'AntMazeComplex-v0':
        # -4 ~ 52
        map_size = [-4, 52]
        Map_x, Map_y = (56, 56)
        start_x, start_y = (4, 4)
    fig5, ax5 = plt.subplots()

    if graphplanner.env_name == 'AntMazeP' or graphplanner.env_name == 'AntMazeW':
        ax5.set_xlim(map_size_x)
        ax5.set_ylim(map_size_y)
    else:
        ax5.set_xlim(map_size)
        ax5.set_ylim(map_size)
    visual_num = 3
    her_ag = batch['ag'][:visual_num]
    her_bg = batch['bg'][:visual_num]
    her_sub = batch['a'][:visual_num]
    bg = original_batch['bg'][:visual_num]
    sub = original_batch['a'][:visual_num]

    ob_x, bg_x, sub_x = [], [], []
    ob_y, bg_y, sub_y = [], [], []
    her_bg_x, her_sub_x = [], []
    her_bg_y, her_sub_y = [], []

    for idx, val in enumerate(zip(her_ag, her_bg, her_sub, bg, sub)):
        cur_ag, her_goal, her_subgoal, goal, subgoal = val
        ob_x.append(cur_ag[0])
        ob_y.append(cur_ag[1])
        her_bg_x.append(her_goal[0])
        her_bg_y.append(her_goal[1])
        her_sub_x.append(her_subgoal[0])
        her_sub_y.append(her_subgoal[1])
        bg_x.append(goal[0])
        bg_y.append(goal[1])
        sub_x.append(subgoal[0])
        sub_y.append(subgoal[1])
        ax5.annotate(str(idx), (cur_ag[0], cur_ag[1]))
        ax5.annotate(str(idx), (her_goal[0], her_goal[1]))
        ax5.annotate(str(idx), (her_subgoal[0], her_subgoal[1]))
        ax5.annotate(str(idx), (goal[0], goal[1]))
        ax5.annotate(str(idx), (subgoal[0], subgoal[1]))

    ax5.scatter(bg_x, bg_y, c='r', marker='s', alpha=1)
    ax5.scatter(sub_x, sub_y, c='b', marker='s', alpha=1)
    ax5.scatter(her_bg_x, her_bg_y, c='r', marker='o', alpha=1)
    ax5.scatter(her_sub_x, her_sub_y, c='b', marker='o', alpha=1)
    ax5.scatter(ob_x, ob_y, c='k', marker='o', alpha=1)

    buf5 = io.BytesIO()
    fig5.savefig(buf5, format='png')
    buf5.seek(0)
    image5 = Image.open(buf5)
    hindsight_array = np.array(image5)
    plt.close()
    return hindsight_array