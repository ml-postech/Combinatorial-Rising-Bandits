import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import io
import networkx as nx
import torch
import random
from PIL import Image
from math import floor, log, sqrt

class GraphPlanner:
    def __init__(self, args, low_replay, low_agent, high_agent, score, env):
        self.low_replay = low_replay
        self.low_agent = low_agent
        self.high_agent = high_agent
        self.env = env
        self.dim = args.subgoal_dim
        self.args = args
        self.score = score
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.graph = None
        self.deleted_node = []
        self.init_dist = 0
        self.n_graph_node = 0
        self.cutoff = args.cutoff
        self.wp_candi = None
        self.landmarks = None
        self.pos = None
        self.states = None
        self.waypoint_vec = None
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0
        self.edge_lengths = None
        self.edge_visit_counts = None
        self.initial_sample = args.initial_sample
        self.waypoint_bef_obs = None
        self.disconnected = []
        self.current = None
        self.n_succeeded_node = 0
        self.waypoint_chase_step_threshold = args.waypoint_chase_step_threshold
        self.waypoint_count_threshold = args.waypoint_count_threshold
        self.time = 0
        self.superarms = []
        
        if self.env.env_name == 'Reacher3D-v0':
            self.n_arms = 66
            self.n_superarms = 4467
            self.previous_timesteps = np.zeros((66,2001), dtype=int)
            self.previous_supertimesteps = np.zeros((4467, 2001), dtype=int)
            self.pull_counts = np.zeros(66, dtype=int)
            self.super_pull_counts = np.zeros(4467, dtype=int)
            self.selected_arm = np.zeros(2001, dtype=int)
            self.selected_edges = np.zeros((66, 2001), dtype=int)
            self.sigma = 0.0001 # hyperparameter
            random.seed(self.args.seed)
            self.from_to = np.array([[0, 1, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 16, 17],
                                     [1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 10, 11, 12, 13, 14, 13, 14, 15, 16, 17, 16, 17, 0, 1, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16]]
                        ).T
            self.cnt_edges = np.zeros(66, dtype=int)
            self.window_size = 325
            self.start_node = 15
            self.end_node = 5
            self.achieve_threshold = 0.1
        
        elif self.env.env_name == 'AntMazeCB':
            self.n_arms = 7
            self.n_superarms = 3
            self.previous_timesteps = np.zeros((7, 5000), dtype=int)
            self.previous_supertimesteps = np.zeros((3, 5000), dtype=int)
            self.pull_counts = np.zeros(7, dtype=int)
            self.super_pull_counts = np.zeros(3, dtype=int)
            self.selected_arm = np.zeros(5000, dtype=int)
            self.selected_edges = np.zeros((7, 5000), dtype=int)
            self.sigma = 0.0001 # hyperparameter
            random.seed(self.args.seed)
            self.from_to = np.array([[0, 0, 1, 1, 2, 5, 4], [1, 3, 2, 4, 5, 4, 3]]).T
            self.cnt_edges = np.zeros(7, dtype=int)
            self.window_size = 325
            self.start_node = 0
            self.end_node = 3   
            self.achieve_threshold = 0.5
            
        elif self.env.env_name == 'AntMazeCBcomplex':
            self.n_arms = 48
            self.n_superarms = 178
            self.previous_timesteps = np.zeros((48, 5000), dtype=int)
            self.previous_supertimesteps = np.zeros((178, 5000), dtype=int)
            self.pull_counts = np.zeros(48, dtype=int)
            self.super_pull_counts = np.zeros(178, dtype=int)
            self.selected_arm = np.zeros(5000, dtype=int)
            self.selected_edges = np.zeros((48, 5000), dtype=int)
            self.sigma = 0.0001 # hyperparameter
            random.seed(self.args.seed)
            self.from_to = np.array([[0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 5, 6 ,7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15],
                                     [1, 2, 3, 4, 5, 6, 7, 5, 6 ,7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 13, 14]]
                        ).T
            self.cnt_edges = np.zeros(48, dtype=int)
            self.window_size = 325
            self.start_node = 0
            self.end_node = 12
            self.achieve_threshold = 0.5
        
        
    def fps_selection(
            self,
            landmarks,
            states,
            n_select: int,
            inf_value=1e6,
            low_bound_epsilon=10, early_stop=True,
    ):
        n_states = landmarks.shape[0]
        dists = np.zeros(n_states) + inf_value
        chosen = []
        while len(chosen) < n_select:
            if (np.max(dists) < low_bound_epsilon) and early_stop and (len(chosen) > self.args.n_graph_node/10):
                break
            idx = np.argmax(dists)  # farthest point idx
            farthest_state = states[idx]
            chosen.append(idx)
            # distance from the chosen point to all other pts
            if self.args.use_oracle_G:
                new_dists = self._get_dist_from_start_oracle(farthest_state, landmarks)
            else:
                new_dists = self.low_agent._get_dist_from_start(farthest_state, landmarks)
            new_dists[idx] = 0
            dists = np.minimum(dists, new_dists)
        return chosen
        
    def graph_construct(self, iter):
        if self.args.method == 'grid8':
            self.current = None
            self.init_dist = self.args.init_dist
            if self.graph is None:
                # if self.env.env_name == 'Reacher3D-v0':
                #     replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                #     landmarks = replay_data['ag']
                #     self.xmin = -1.0
                #     self.xmax = 1.1
                #     self.ymin = -1.0
                #     self.ymax = 1.1
                #     self.zmin = -1.0
                #     self.zmax = 1.1
                #     x = np.arange(self.xmin + self.init_dist/2, self.xmax, self.init_dist)
                #     y = np.arange(self.ymin + self.init_dist/2, self.ymax, self.init_dist)
                #     z = np.arange(self.zmin, self.zmax, self.init_dist)
                #     X,Y,Z = np.meshgrid(x, y, z)
                #     self.landmarks = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
                #     self.n_graph_node = self.landmarks.shape[0]
                #     self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                    
                #     self.graph = nx.DiGraph()
                #     for i in range(self.n_graph_node):
                #         for j in range(self.n_graph_node):
                #             if i != j:
                #                 if (np.linalg.norm(self.landmarks[i][0] - self.landmarks[j][0]) <= self.init_dist * 1.01) and (np.linalg.norm(self.landmarks[i][1] - self.landmarks[j][1]) <= self.init_dist * 1.01) and (np.linalg.norm(self.landmarks[i][2] - self.landmarks[j][2]) <= self.init_dist * 1.01):
                #                     self.graph.add_edge(i, j, weight = np.linalg.norm(self.landmarks[i] - self.landmarks[j]))
                    
                #     nx.set_node_attributes(self.graph, self.init_dist, 'distance')                
                #     nx.set_node_attributes(self.graph, 0, 'attempt_count')
                #     nx.set_node_attributes(self.graph, 0, 'success_count')
                #     nx.set_node_attributes(self.graph, 0, 'before')
                #     nx.set_edge_attributes(self.graph, 0, 'visit_count')
                #     l = landmarks.shape[0]
                #     for i in range(l):
                #         for j in range(self.n_graph_node):
                #             dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                #             if dist < 0.1:
                #                 self.graph.nodes[j]['success_count'] += 1       
                                
                #     return self.landmarks, self.states
                if self.env.env_name == 'Reacher3D-v0':
                    x = np.arange(1, 0.5, -0.2)
                    y = np.arange(-0.2, 0.3, 0.2)
                    z = np.arange(0, -0.3, -0.2)
                    X, Y, Z = np.meshgrid(x, y, z)
                    
                    self.landmarks = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]  
                    self.graph = nx.DiGraph()
                    # 6- 7- 8            15-16-17
                    # |  |  |             |  |  |
                    # 3- 4- 5            12-13-14
                    # |  |  |             |  |  |
                    # 0- 1- 2             9-10-11
                    # 
                    # .10.11.  18 19 20  .31.32.
                    # 7  8  9           28 29 30
                    # .5 .6 .  15 16 17  .26.27.
                    # 2  3  4           23 24 25
                    # .0 .1 .  12 13 14  .21.22.
                    #
                    # 15, 12, 9, 10, 11, 14, 5
                    self.graph.add_edge(0, 1, weight = 0)#0
                    self.graph.add_edge(1, 2, weight = 0)
                    self.graph.add_edge(0, 3, weight = 0)
                    self.graph.add_edge(1, 4, weight = 0)
                    self.graph.add_edge(2, 5, weight = 0)
                    self.graph.add_edge(3, 4, weight = 0)#5
                    self.graph.add_edge(4, 5, weight = 0)
                    self.graph.add_edge(3, 6, weight = 0)
                    self.graph.add_edge(4, 7, weight = 0)
                    self.graph.add_edge(5, 8, weight = 0)
                    self.graph.add_edge(6, 7, weight = 0)#10
                    self.graph.add_edge(7, 8, weight = 0)
                    self.graph.add_edge(0, 9, weight = 0)
                    self.graph.add_edge(1, 10, weight = 0)
                    self.graph.add_edge(2, 11, weight = 0)
                    self.graph.add_edge(3, 12, weight = 0)#15
                    self.graph.add_edge(4, 13, weight = 0)
                    self.graph.add_edge(5, 14, weight = 0)
                    self.graph.add_edge(6, 15, weight = 0)
                    self.graph.add_edge(7, 16, weight = 0)
                    self.graph.add_edge(8, 17, weight = 0)#20
                    self.graph.add_edge(9, 10, weight = 0)
                    self.graph.add_edge(10, 11, weight = 0)
                    self.graph.add_edge(9, 12, weight = 0)
                    self.graph.add_edge(10, 13, weight = 0)
                    self.graph.add_edge(11, 14, weight = 0)#25
                    self.graph.add_edge(12, 13, weight = 0)
                    self.graph.add_edge(13, 14, weight = 0)
                    self.graph.add_edge(12, 15, weight = 0)
                    self.graph.add_edge(13, 16, weight = 0)
                    self.graph.add_edge(14, 17, weight = 0)#30
                    self.graph.add_edge(15, 16, weight = 0)
                    self.graph.add_edge(16, 17, weight = 0)
                    self.graph.add_edge(1, 0, weight = 0)
                    self.graph.add_edge(2, 1, weight = 0)
                    self.graph.add_edge(3, 0, weight = 0)#35
                    self.graph.add_edge(4, 1, weight = 0)
                    self.graph.add_edge(5, 2, weight = 0)
                    self.graph.add_edge(4, 3, weight = 0)
                    self.graph.add_edge(5, 4, weight = 0)
                    self.graph.add_edge(6, 3, weight = 0)#40
                    self.graph.add_edge(7, 4, weight = 0)
                    self.graph.add_edge(8, 5, weight = 0)
                    self.graph.add_edge(7, 6, weight = 0)
                    self.graph.add_edge(8, 7, weight = 0)
                    self.graph.add_edge(9, 0, weight = 0)#45
                    self.graph.add_edge(10, 1, weight = 0)
                    self.graph.add_edge(11, 2, weight = 0)
                    self.graph.add_edge(12, 3, weight = 0)
                    self.graph.add_edge(13, 4, weight = 0)
                    self.graph.add_edge(14, 5, weight = 0)#50
                    self.graph.add_edge(15, 6, weight = 0)
                    self.graph.add_edge(16, 7, weight = 0)
                    self.graph.add_edge(17, 8, weight = 0)
                    self.graph.add_edge(10, 9, weight = 0)
                    self.graph.add_edge(11, 10, weight = 0)#55
                    self.graph.add_edge(12, 9, weight = 0)
                    self.graph.add_edge(13, 10, weight = 0)
                    self.graph.add_edge(14, 11, weight = 0)
                    self.graph.add_edge(13, 12, weight = 0)
                    self.graph.add_edge(14, 13, weight = 0)#60
                    self.graph.add_edge(15, 12, weight = 0)
                    self.graph.add_edge(16, 13, weight = 0)
                    self.graph.add_edge(17, 14, weight = 0)
                    self.graph.add_edge(16, 15, weight = 0)
                    self.graph.add_edge(17, 16, weight = 0)#65
                    self.superarms = list(nx.all_simple_paths(self.graph, source=self.start_node, target=self.end_node))
                    
                    return self.landmarks, self.states
                elif self.env.env_name == 'AntMazeCB':
                    
                    x = np.arange(0, 18, 8)
                    y = np.arange(0, 10, 8)
                    X,Y = np.meshgrid(x, y)
                    
                    self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]    
                    
                    self.graph = nx.DiGraph()
                    # 3-4-5
                    # | | |
                    # 0-1-2
                    # .6.5. 
                    # 1 3 4
                    # .0.2.
                    self.graph.add_edge(0, 1, weight = 0)
                    self.graph.add_edge(0, 3, weight = 0)
                    self.graph.add_edge(1, 2, weight = 0)
                    self.graph.add_edge(1, 4, weight = 0)
                    self.graph.add_edge(2, 5, weight = 0)
                    self.graph.add_edge(5, 4, weight = 0)
                    self.graph.add_edge(4, 3, weight = 0)            
                    
                    self.superarms = list(nx.all_simple_paths(self.graph, source=self.start_node, target=self.end_node))
                    
                    return self.landmarks, self.states
                elif self.env.env_name == 'AntMazeCBcomplex':
                    x = np.arange(0, 26, 8)
                    y = np.arange(0, 26, 8)
                    X,Y = np.meshgrid(x, y)
                    
                    self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]    
                    
                    self.graph = nx.DiGraph()
                    #12-13-14-15
                    # |  |  |  |
                    # 8- 9-10-11
                    # |  |  |  |
                    # 4- 5- 6- 7
                    # |  |  |  |
                    # 0- 1- 2- 3
                    # .21.22.23.
                    #17 18 19 20
                    # .14.15.16.
                    #10 11 12 13
                    # . 7. 8. 9.
                    # 3  4  5  6
                    # . 0. 1. 2.
                    self.graph.add_edge(0, 1, weight = 0)#0
                    self.graph.add_edge(1, 2, weight = 0)
                    self.graph.add_edge(2, 3, weight = 0)
                    self.graph.add_edge(0, 4, weight = 0)
                    self.graph.add_edge(1, 5, weight = 0)
                    self.graph.add_edge(2, 6, weight = 0)#5
                    self.graph.add_edge(3, 7, weight = 0)
                    self.graph.add_edge(4, 5, weight = 0)
                    self.graph.add_edge(5, 6, weight = 0)
                    self.graph.add_edge(6, 7, weight = 0)
                    self.graph.add_edge(4, 8, weight = 0)#10
                    self.graph.add_edge(5, 9, weight = 0)
                    self.graph.add_edge(6,10, weight = 0)
                    self.graph.add_edge(7,11, weight = 0)
                    self.graph.add_edge(8, 9, weight = 0)
                    self.graph.add_edge(9, 10, weight = 0)#15
                    self.graph.add_edge(10, 11, weight = 0)
                    self.graph.add_edge(8, 12, weight = 0)
                    self.graph.add_edge(9, 13, weight = 0)
                    self.graph.add_edge(10, 14, weight = 0)
                    self.graph.add_edge(11, 15, weight = 0)#20
                    self.graph.add_edge(12, 13, weight = 0)
                    self.graph.add_edge(13, 14, weight = 0)
                    self.graph.add_edge(14, 15, weight = 0)    
                    self.graph.add_edge(1, 0, weight = 0)#0
                    self.graph.add_edge(2, 1, weight = 0)
                    self.graph.add_edge(3, 2, weight = 0)
                    self.graph.add_edge(4, 0, weight = 0)
                    self.graph.add_edge(5, 1, weight = 0)
                    self.graph.add_edge(6, 2, weight = 0)#5
                    self.graph.add_edge(7, 3, weight = 0)
                    self.graph.add_edge(5, 4, weight = 0)
                    self.graph.add_edge(6, 5, weight = 0)
                    self.graph.add_edge(7, 6, weight = 0)
                    self.graph.add_edge(8, 4, weight = 0)#10
                    self.graph.add_edge(9, 5, weight = 0)
                    self.graph.add_edge(10, 6, weight = 0)
                    self.graph.add_edge(11, 7, weight = 0)
                    self.graph.add_edge(9, 8, weight = 0)
                    self.graph.add_edge(10, 9, weight = 0)#15
                    self.graph.add_edge(11, 10, weight = 0)
                    self.graph.add_edge(12, 8, weight = 0)
                    self.graph.add_edge(13, 9, weight = 0)
                    self.graph.add_edge(14, 10, weight = 0)
                    self.graph.add_edge(15, 11, weight = 0)#20
                    self.graph.add_edge(13, 12, weight = 0)
                    self.graph.add_edge(14, 13, weight = 0)
                    self.graph.add_edge(15, 14, weight = 0)       
                    
                    self.superarms = list(nx.all_simple_paths(self.graph, source=self.start_node, target=self.end_node))
                    
                    return self.landmarks, self.states
                elif self.env.env_name == 'AntMaze' or self.env.env_name == 'AntMazeBottleneck':
                    self.xmin = -4
                    self.xmax = 20
                    self.ymin = -4
                    self.ymax = 20
                elif self.env.env_name == 'AntMazeP':
                    self.xmin = -12
                    self.xmax = 28
                    self.ymin = -4
                    self.ymax = 36
                elif self.env.env_name == 'AntMazeS':
                    self.xmin = -4
                    self.xmax = 36
                    self.ymin = -4
                    self.ymax = 36
                elif self.env.env_name == 'AntMazeW':
                    self.xmin = -4
                    self.xmax = 36
                    self.ymin = -12
                    self.ymax = 28
                elif self.env.env_name == 'AntMazeComplex-v0':
                    self.xmin = -4
                    self.xmax = 52
                    self.ymin = -4
                    self.ymax = 52
                
                replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                landmarks = replay_data['ag']
                
                x = np.arange(self.xmin + self.init_dist/2, self.xmax, self.init_dist)
                y = np.arange(self.ymin + self.init_dist/2, self.ymax, self.init_dist)
                X,Y = np.meshgrid(x, y)
                self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                
                # self.landmarks = np.random.uniform(low=-4, high=20, size=(200,2))
                
                self.n_graph_node = self.landmarks.shape[0]
                self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                
                self.states = np.zeros((self.landmarks.shape[0], 29))
                random_state = replay_data['ob'][0,2:29]
                self.pos = np.expand_dims(random_state, axis = 0)
                self.states[:,2:29] = random_state
                self.states[:,:2] = self.landmarks
                
                self.graph = nx.DiGraph()
                for i in range(self.n_graph_node):
                    # self.graph.add_node(i)
                    for j in range(self.n_graph_node):
                        if i != j:
                            if (np.linalg.norm(self.landmarks[i][0] - self.landmarks[j][0]) <= self.init_dist * 1.01) and (np.linalg.norm(self.landmarks[i][1] - self.landmarks[j][1]) <= self.init_dist * 1.01):
                                self.graph.add_edge(i, j, weight = self.init_dist)
                                
                nx.set_node_attributes(self.graph, self.init_dist, 'distance')
                nx.set_node_attributes(self.graph, 0, 'attempt_count')
                nx.set_node_attributes(self.graph, 0, 'success_count')
                nx.set_node_attributes(self.graph, 0, 'before')
                nx.set_edge_attributes(self.graph, 0, 'visit_count')
                
                l = landmarks.shape[0]
                for i in range(l):
                    for j in range(self.n_graph_node):
                        dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                        if dist < 0.5:
                            self.graph.nodes[j]['attempt_count'] += 1
                            self.graph.nodes[j]['success_count'] += 1       
                for i in range(self.n_graph_node):
                    if self.graph.nodes[i]['success_count'] > 0:
                        self.n_succeeded_node += 1
                        
                return self.landmarks, self.states
            else:
                return self.landmarks, self.states

    
    def dense(self, dg):
        is_dense = True
        dist = self.graph.nodes[dg]['distance']
        
        if self.env.env_name=='Reacher3D-v0':
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        exist = False
                        candi = np.array([0., 0., 0.])
                        candi[0] = self.landmarks[dg][0] + dist * i
                        candi[1] = self.landmarks[dg][1] + dist * j
                        candi[2] = self.landmarks[dg][2] + dist * k
                    for k in range(self.n_graph_node):
                        if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                            exist = True
                    if exist == False:
                        is_dense = False
        else:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    exist = False
                    candi = np.array([0., 0.])
                    candi[0] = self.landmarks[dg][0] + dist * i
                    candi[1] = self.landmarks[dg][1] + dist * j
                    for k in range(self.n_graph_node):
                        if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                            exist = True
                    if exist == False:
                        is_dense = False
        
        edges = self.graph.edges(data=True)
        
        if is_dense:
            remove_edges = []
            for edge in edges:
                if edge[0] == dg or edge[1] == dg:
                    remove_edges.append(edge)
            for edge in remove_edges:
                self.graph.remove_edge(*edge[:2])
        
        self.graph.nodes[dg]['attempt_count'] = 0
        self.graph.nodes[dg]['success_count'] = 0
        if is_dense:
            dist = self.graph.nodes[dg]['distance'] / 2.
            self.graph.nodes[dg]['distance'] = dist
        
        
        if self.env.env_name=='Reacher3D-v0':
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for n in range(-2, 3):
                        exist = False
                        candi = np.array([0., 0., 0.])
                        candi[0] = self.landmarks[dg][0] + dist * i
                        candi[1] = self.landmarks[dg][1] + dist * j
                        candi[2] = self.landmarks[dg][2] + dist * n
                        for k in range(self.n_graph_node):
                            if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                                exist = True
                                self.graph.nodes[k]['attempt_count'] = 0
                                self.graph.nodes[k]['success_count'] = 0
                                for l in range(self.n_graph_node):
                                    if l != k:
                                        d = np.min([dist, self.graph.nodes[l]['distance']])
                                        if (np.linalg.norm(self.landmarks[k][0] - self.landmarks[l][0]) < 1.01 * d) and (np.linalg.norm(self.landmarks[k][1] - self.landmarks[l][1]) < 1.01 * d) and (np.linalg.norm(self.landmarks[k][2] - self.landmarks[l][2]) < 1.01 * d):
                                            if self.graph.has_edge(k, l):
                                                self.graph[k][l]['weight'] = np.linalg.norm(self.landmarks[k] - self.landmarks[l])
                                                self.graph[l][k]['weight'] = np.linalg.norm(self.landmarks[k] - self.landmarks[l])
                                                self.graph.nodes[k]['distance'] = dist
                                if k in self.disconnected:
                                    self.disconnected.remove(k)
                            
                    if not exist:
                        candi = np.expand_dims(candi, axis = 0)
                        self.landmarks = np.concatenate((self.landmarks, candi))
                        self.graph.add_node(self.n_graph_node)
                        self.graph.nodes[self.n_graph_node]['attempt_count'] = 0
                        self.graph.nodes[self.n_graph_node]['success_count'] = 0
                        self.graph.nodes[self.n_graph_node]['distance'] = dist
                        for m in range(self.n_graph_node):
                            d = np.min([dist,self.graph.nodes[m]['distance']])
                            if (np.linalg.norm(self.landmarks[self.n_graph_node][0] - self.landmarks[m][0]) < 1.01 * d) and (np.linalg.norm(self.landmarks[self.n_graph_node][1] - self.landmarks[m][1]) < 1.01 * d) and (np.linalg.norm(self.landmarks[self.n_graph_node][2] - self.landmarks[m][2]) < 1.01 * d):
                                self.graph.add_edge(m, self.n_graph_node, weight = np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]))
                                self.graph.add_edge(self.n_graph_node, m, weight = np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]))
                        
                        self.n_graph_node += 1
             
        else:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    exist = False
                    candi = np.array([0., 0.])
                    candi[0] = self.landmarks[dg][0] + dist * i
                    candi[1] = self.landmarks[dg][1] + dist * j
                    for k in range(self.n_graph_node):
                        if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                            exist = True
                            self.graph.nodes[k]['attempt_count'] = 0
                            self.graph.nodes[k]['success_count'] = 0
                            for l in range(self.n_graph_node):
                                if l != k:
                                    d = np.min([dist, self.graph.nodes[l]['distance']])
                                    if (np.linalg.norm(self.landmarks[k][0] - self.landmarks[l][0]) < 1.01 * d) and (np.linalg.norm(self.landmarks[k][1] - self.landmarks[l][1]) < 1.01 * d):
                                        if self.graph.has_edge(k, l):
                                            self.graph[k][l]['weight'] = np.linalg.norm(self.landmarks[k] - self.landmarks[l])
                                            self.graph[l][k]['weight'] = np.linalg.norm(self.landmarks[k] - self.landmarks[l])
                                            self.graph.nodes[k]['distance'] = dist
                            if k in self.disconnected:
                                self.disconnected.remove(k)
                            
                    if not exist:
                        candi = np.expand_dims(candi, axis = 0)
                        self.landmarks = np.concatenate((self.landmarks, candi))
                        candi_state = np.concatenate((candi, self.pos), axis=1)
                        self.states = np.concatenate((self.states, candi_state))
                        self.graph.add_node(self.n_graph_node)
                        self.graph.nodes[self.n_graph_node]['attempt_count'] = 0
                        self.graph.nodes[self.n_graph_node]['success_count'] = 0
                        self.graph.nodes[self.n_graph_node]['distance'] = dist
                        for m in range(self.n_graph_node):
                            d = np.min([dist,self.graph.nodes[m]['distance']])
                            if (np.linalg.norm(self.landmarks[self.n_graph_node][0] - self.landmarks[m][0]) < 1.01 * d) and (np.linalg.norm(self.landmarks[self.n_graph_node][1] - self.landmarks[m][1]) < 1.01 * d):
                                self.graph.add_edge(m, self.n_graph_node, weight = np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]))
                                self.graph.add_edge(self.n_graph_node, m, weight = np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]))
                        
                        self.n_graph_node += 1
                        
    
    def densify(self):
        failed = []
        cnt = 0
        for i in range(self.n_graph_node):
            if(self.graph.nodes[i]['success_count'] > 0):
                cnt += 1
        for i in range(self.n_graph_node):
            if((self.graph.nodes[i]['attempt_count'] > self.waypoint_count_threshold) and (self.graph.nodes[i]['success_count'] == 0)):
                failed.append(self.graph.nodes[i]['distance'])
            else:
                failed.append(0)
                
        failed = np.array(failed)
        max_dist = np.max(failed)
        
        if cnt > self.n_succeeded_node:
            self.n_succeeded_node = cnt
            return
        if max_dist == 0:
            return
        candi = np.where(failed == max_dist)
        
        self.dense(candi[0][random.choices(range(len(candi[0])))][0])
        
        return 

    def superarm_index(self, target):
        for i, array in enumerate(self.superarms):
            if array == target:
                return i
        return -1
    
    def _reward(self, s):
        return 0 if s >=1e3 else 1/s

    def pulling(self):
        # 3-4-5
        # | | |
        # 0-1-2
        # .6.5. 
        # 1 3 4
        # .0.2.
        #12-13-14-15
        # |  |  |  |
        # 8- 9-10-11
        # |  |  |  |
        # 4- 5- 6- 7
        # |  |  |  |
        # 0- 1- 2- 3
        # .21.22.23.
        #17 18 19 20
        # .14.15.16.
        #10 11 12 13
        # . 7. 8. 9.
        # 3  4  5  6
        # . 0. 1. 2.
        edges = self.graph.edges(data=True)
        if self.args.setting == 'optimal':
            if self.env.env_name == 'AntMazeCB':
                path = [0, 1, 4, 3]
            if self.env.env_name == 'AntMazeCBcomplex':
                path = [0, 1, 5, 4, 8, 12]
            if self.env.env_name == 'Reacher3D-v0':
                path = [15, 12, 9, 10, 11, 14, 5]
        else:  
            minimum_index = 0
            minimum_time_step = 1e6
            if self.args.setting == 'CRUCB':
                print(self.args.setting)
                for i in range(self.n_arms):
                    h = floor(self.pull_counts[i]/4)
                    if h > 0:
                        delta = self.time ** -4
                        reward = 0
                        reward_slope = 0
                        for j in range(self.pull_counts[i]-h, self.pull_counts[i]):
                            reward += self._reward(self.previous_timesteps[i][j]) 
                            reward_slope = (self.time-j)*(self._reward(self.previous_timesteps[i][j])-self._reward(self.previous_timesteps[i][j-h]))/h
                        reward = reward + reward_slope if reward_slope > 0 else reward
                        reward = reward / h
                        beta = self.sigma * (self.time - self.pull_counts[i] + h - 1) * sqrt(10*log(1/delta)/(h**3))
                        expected_reward = max(reward+beta, 1e-6)
                        expected_time_step = 1/(expected_reward)
                        self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = expected_time_step
                        print(f'{i}:{beta}, {reward}, {expected_time_step}')
                        
            elif self.args.setting == 'RUCB':
                print(self.args.setting)
                for i in range(self.n_superarms):
                    h = floor(self.super_pull_counts[i]/4)
                    if h > 0:
                        delta = self.time ** -4
                        reward = 0
                        reward_slope = 0
                        for j in range(self.super_pull_counts[i]-h, self.super_pull_counts[i]):
                            reward += self._reward(self.previous_supertimesteps[i][j]) 
                            reward_slope = (self.time-j)*(self._reward(self.previous_supertimesteps[i][j])-self._reward(self.previous_supertimesteps[i][j-h]))/h
                        reward = reward + reward_slope if reward_slope > 0 else reward
                        reward = reward / h
                        beta = self.sigma * (self.time - self.super_pull_counts[i] + h - 1) * sqrt(10*log(1/delta)/(h**3))
                        expected_reward = max(reward+beta, 1e-6)
                        expected_time_step = 1/(expected_reward)
                        # self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = expected_time_step
                        if expected_time_step < minimum_time_step:
                            minimum_index = i
                            minimum_time_step = expected_time_step
                    else:
                        minimum_index = i
                        minimum_time_step = -1
                        break
                        
            elif self.args.setting == 'CSWUCB':
                print(self.args.setting)
                for i in range(self.n_arms):
                    h = floor(self.pull_counts[i]/4)
                    if h > 0:
                        reward = 0
                        for j in range(self.pull_counts[i]-h, self.pull_counts[i]):
                            reward += self._reward(self.previous_timesteps[i][j])
                        reward = reward/h
                        reward += sqrt(2*log(self.time)/self.pull_counts[i])
                        expected_reward = max(reward, 1e-6)
                        expected_time_step = 1/(expected_reward)
                        self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = expected_time_step
                        print(f'{i}:{reward}, {expected_time_step}')
                        
            elif self.args.setting == 'SWUCB':
                print(self.args.setting)
                for i in range(self.n_superarms):
                    h = floor(self.super_pull_counts[i]/4)
                    if h > 0:
                        reward = 0
                        for j in range(self.super_pull_counts[i]-h, self.super_pull_counts[i]):
                            reward += self._reward(self.previous_supertimesteps[i][j])
                        reward = reward/h
                        reward += sqrt(2*log(self.time)/self.super_pull_counts[i])
                        expected_reward = max(reward, 1e-6)
                        expected_time_step = 1/(expected_reward)
                        if expected_time_step < minimum_time_step:
                            minimum_index = i
                            minimum_time_step = expected_time_step
                    else:
                        minimum_index = i
                        minimum_time_step = -1
                        break
                        
            
            # elif self.args.setting == 'TCSWUCB' or self.args.setting == 'TSWUCB':
            #     print(self.args.setting)
            #     # h = 325 # 4 sqrt(T log T)
                
            #     for i in range(self.n_arms):
            #         h = self.cnt_edges[i]
            #         reward = 0
            #         if h > 0:
            #             for j in range(self.pull_counts[i]-h, self.pull_counts[i]):
            #                 reward += self._reward(self.previous_timesteps[i][j])
            #             reward = reward/h
            #             reward += sqrt(2*log(np.max((np.min((self.time, self.window_size)),1)))/h)
            #             expected_reward = max(reward, 1e-6)
            #             expected_time_step = 1/(expected_reward)
            #             self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = expected_time_step
            #         else:
            #             self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = 0
            
            elif self.args.setting == 'CSWTS':
                print(self.args.setting)
                for i in range(self.n_arms):
                    h = floor(self.pull_counts[i]/4)
                    if h > 0:
                        reward_alpha = 1
                        reward_beta = 1
                        for j in range(self.pull_counts[i]-h, self.pull_counts[i]):
                            reward = self._reward(self.previous_timesteps[i][j])
                            reward_alpha += reward
                            reward_beta += (1-reward)
                        reward = np.random.beta(reward_alpha, reward_beta)
                        expected_reward = max(reward, 1e-6)
                        expected_time_step = 1/(expected_reward)
                        self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = expected_time_step
                        print(f'{i}:{reward}, {expected_time_step}')
                        
            elif self.args.setting == 'SWTS':
                print(self.args.setting)
                for i in range(self.n_superarms):
                    h = floor(self.super_pull_counts[i]/4)
                    if h > 0:
                        reward_alpha = 1
                        reward_beta = 1
                        for j in range(self.super_pull_counts[i]-h, self.super_pull_counts[i]):
                            reward = self._reward(self.previous_supertimesteps[i][j])
                            reward_alpha += reward
                            reward_beta += (1-reward)
                        reward = np.random.beta(reward_alpha, reward_beta)
                        expected_reward = max(reward, 1e-6)
                        expected_time_step = 1/(expected_reward)
                        if expected_time_step < minimum_time_step:
                            minimum_index = i
                            minimum_time_step = expected_time_step
                    else:
                        minimum_index = i
                        minimum_time_step = -1
                        break
                        
            # elif self.args.setting == 'TCSWTS' or self.args.setting == 'TSWTS':
                # print(self.args.setting)
                # for i in range(self.n_arms):
                #     h = self.cnt_edges[i]
                #     reward_alpha = 1
                #     reward_beta = 1
                #     for j in range(self.pull_counts[i]-h, self.pull_counts[i]):
                #         reward = self._reward(self.previous_timesteps[i][j])
                #         reward_alpha += reward
                #         reward_beta += (1-reward)
                #     reward = np.random.beta(reward_alpha, reward_beta)
                #     expected_reward = max(reward, 1e-6)
                #     expected_time_step = 1/(expected_reward)
                #     self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = expected_time_step
                #     self.graph[self.from_to[i][0]][self.from_to[i][1]]['weight'] = 0
            
            if (self.args.setting == 'CRUCB') or (self.args.setting == 'CSWUCB') or (self.args.setting == 'CSWTS'):   
                path = nx.shortest_path(self.graph, self.start_node, self.end_node, weight='weight')
                
            elif (self.args.setting == 'RUCB') or (self.args.setting == 'SWUCB') or (self.args.setting == 'SWTS'):
                path = self.superarms[minimum_index]
            
        if self.env.env_name == 'AntMazeCB':
            self.selected_arm[self.time] = self.superarm_index(path)
            print(self.selected_arm[self.time]) 
            
            # if self.time >= self.window_size:
            #     for i in range(7):
            #         if self.selected_edges[i][self.time - self.window_size] == 1:
            #             self.cnt_edges[i] -= 1
        
        if self.env.env_name == 'AntMazeCBcomplex' or self.env.env_name == 'Reacher3D-v0':
            self.selected_arm[self.time] = self.superarm_index(path)
            print(self.selected_arm[self.time]) 
            # if self.time >= self.window_size:
            #     for i in range(48):
            #         if self.selected_edges[i][self.time - self.window_size] == 1:
            #             self.cnt_edges[i] -= 1
        
        self.time += 1
        
        return path
    
    def update_experience(self, is_success, final = False):
        if (self.args.setting == 'CRUCB') or (self.args.setting == 'CSWUCB') or (self.args.setting == 'CSWTS') or (self.args.setting == 'TCSWUCB') or (self.args.setting == 'TCSWTS') or (self.args.setting == 'optimal'):
            i = np.where(np.all(self.from_to == [self.waypoint_vec[self.waypoint_idx-1], self.waypoint_vec[self.waypoint_idx]], axis=1))[0][0]
            self.selected_edges[i][self.time-1] = 1
            self.cnt_edges[i] += 1
            self.previous_timesteps[i][self.pull_counts[i]] = self.waypoint_chase_step if is_success else 1e6
            self.pull_counts[i] += 1
            
        elif (self.args.setting == 'RUCB') or (self.args.setting == 'SWUCB') or (self.args.setting == 'SWTS') or (self.args.setting == 'TSWUCB') or (self.args.setting == 'TSWTS'):
            i = np.where(np.all(self.from_to == [self.waypoint_vec[self.waypoint_idx-1], self.waypoint_vec[self.waypoint_idx]], axis=1))[0][0]
            j = self.selected_arm[self.time-1]
            self.selected_edges[i][self.time-1] = 1
            self.cnt_edges[i] += 1
            self.previous_timesteps[i][self.pull_counts[i]] = self.waypoint_chase_step if is_success else 1e6
            self.pull_counts[i] += 1
            
            if is_success:
                self.previous_supertimesteps[j][self.super_pull_counts[j]] += self.waypoint_chase_step
                if final:
                    self.super_pull_counts[j] += 1
            else:
                self.previous_supertimesteps[j][self.super_pull_counts[j]] = 1e6
                self.super_pull_counts[j] += 1
        
        return
            
    def find_path(self, ob, subgoal, ag, bg, inf_value=1e6, train = False, first = False):
        self.edge_lengths = []
        if self.args.nosubgoal:
            subgoal = bg
        else:
            subgoal = subgoal[:self.dim]
        self.wp_candi = None
        
        if self.env.env_name == 'AntMazeCB' or self.env.env_name == 'AntMazeCBcomplex' or self.env.env_name == 'Reacher3D-v0':
            path = self.pulling()
            
            self.wp_candi = subgoal
            self.waypoint_vec = list(path)
            self.waypoint_idx = 1
            self.waypoint_chase_step = 0
            return self.wp_candi
        
        elif self.args.method == 'grid8':
            expanded_graph = self.graph.copy()
            if first:
                self.deleted_node = []
            if self.deleted_node:
                for i in self.deleted_node:
                    for j in range(self.n_graph_node):
                        if i != j:
                            if expanded_graph.has_edge(i, j):
                                expanded_graph[i][j]['weight'] = 10000.
                                expanded_graph[j][i]['weight'] = 10000.
            
            if self.env.env_name != 'Reacher3D-v0':
                edges = expanded_graph.edges(data=True)
                pdist = self.low_agent._get_pairwise_dist(self.states, self.landmarks)
                for edge in edges:
                    if expanded_graph[edge[0]][edge[1]]['weight'] < 500.:
                        expanded_graph[edge[0]][edge[1]]['weight'] = pdist[edge[0]][edge[1]]
            
            start_to_goal_length = np.linalg.norm(ag - bg)
            
            if start_to_goal_length < self.init_dist:
                if self.env.env_name != 'Reacher3D-v0':
                    start_to_goal_length = np.squeeze(self.low_agent._get_point_to_point(ob, subgoal))
                expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
            
            start_edge_length = self.dist_to_graph(ag, self.landmarks)
            goal_edge_length = self.dist_to_graph(bg, self.landmarks)
            if self.env.env_name != 'Reacher3D-v0':    
                start_edge_length_ = self.low_agent._get_dist_from_start(ob, self.landmarks)
                goal_edge_length_ = self.low_agent._get_dist_to_goal(self.states, bg)
            else:
                start_edge_length_ = start_edge_length
                goal_edge_length_ = goal_edge_length
            self.edge_lengths = [] 
            
            for i in range(self.n_graph_node):
                if start_edge_length[i] < self.init_dist:
                    if i not in self.disconnected:
                        expanded_graph.add_edge('start', i, weight = start_edge_length_[i])
                    else:
                        expanded_graph.add_edge('start', i, weight = 10000.)
                if goal_edge_length[i] < self.init_dist:
                    if i not in self.disconnected:
                        expanded_graph.add_edge(i, 'goal', weight = goal_edge_length_[i])
                    else:
                        expanded_graph.add_edge(i, 'goal', weight = 10000.)
            if (not expanded_graph.has_node('start')):
                added = False
                adjusted = 1.5
                while True:
                    adjusted_cutoff = self.init_dist * adjusted
                    for i in range(self.n_graph_node):
                        if(start_edge_length[i] < adjusted_cutoff):
                            if i not in self.disconnected:
                                expanded_graph.add_edge('start', i, weight = start_edge_length_[i])
                                added = True
                    if added:
                        break
                    adjusted += 0.5           
            
            if(not expanded_graph.has_node('goal')):
                adjusted_cutoff = 2.0 * self.init_dist
                for i in range(self.n_graph_node):
                    if(goal_edge_length[i] < adjusted_cutoff):
                        if i not in self.disconnected:
                            expanded_graph.add_edge(i, 'goal', weight = goal_edge_length_[i])
            
            if(not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal')):
                while True:
                    nearestnode = np.argmin(goal_edge_length) #nearest point from the goal
                    if goal_edge_length_[nearestnode] > start_to_goal_length:
                        expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
                        break
                    if(expanded_graph.has_node(nearestnode)) and (nx.has_path(expanded_graph, 'start', nearestnode)):
                        expanded_graph.add_edge(nearestnode, 'goal', weight = goal_edge_length_[nearestnode])
                        break
                    else:
                        goal_edge_length[nearestnode] = inf_value
                        
            path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
            for (i, j) in zip(path[:-1], path[1:]):
                self.edge_lengths.append(expanded_graph[i][j]['weight'])
                
            self.waypoint_vec = list(path)[1:-1]
            self.waypoint_idx = 0
            self.waypoint_chase_step = 0
            self.wp_candi = subgoal
            
            return self.wp_candi
    
    def get_goal_candi(self, expanded_graph):
        start_edge_length = []
        exist = False
        for i in range(self.n_graph_node):
            if self.graph.nodes[i]['attempt_count'] == 0:
                if nx.has_path(expanded_graph, 'start', i):
                    start_edge_length.append(nx.shortest_path_length(expanded_graph, source='start', target=i, weight='weight'))
                    exist = True
                else:
                    start_edge_length.append(100000)
            else:
                start_edge_length.append(100000)
        if exist:
            return self.landmarks[np.argmin(start_edge_length)]
        return None
            
    def check_easy_goal(self, ob, ag, subgoal):
        expanded_graph = self.graph.copy()
        subgoal = subgoal[:self.dim]
        if self.args.method == 'dhrl':
            if self.args.use_oracle_G:
                start_edge_length = self._get_dist_from_start_oracle(ob, self.landmarks)
            else:
                start_edge_length = self.low_agent._get_dist_from_start(ob, self.landmarks)
            if self.args.use_oracle_G:
                goal_edge_length = self._get_dist_to_goal_oracle(self.states, subgoal)
            else:
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, subgoal)
            for i in range(self.n_graph_node):
                if(start_edge_length[i] < self.cutoff):
                    expanded_graph.add_edge('start', i, weight = start_edge_length[i])
                if(goal_edge_length[i] < self.cutoff):
                    expanded_graph.add_edge(i, 'goal', weight = goal_edge_length[i])

            if self.args.use_oracle_G:
                start_to_goal_length = np.squeeze(self._get_point_to_point_oracle(ob, subgoal))
            else:
                start_to_goal_length = np.squeeze(self.low_agent._get_point_to_point(ob, subgoal))
            if start_to_goal_length < self.cutoff:
                expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
            
            # 1. DHRL
            
            if((not expanded_graph.has_node('start')) or (not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal'))):
                return None
            elif(nx.has_path(expanded_graph, 'start', 'goal')):
                start_edge_length = []
                for i in range (self.n_graph_node):
                    if expanded_graph.has_node(i) and nx.has_path(expanded_graph, 'start', i):
                        start_edge_length.append(nx.shortest_path_length(expanded_graph, source='start', target=i, weight='weight'))
                    else:
                        start_edge_length.append(5e3)
                start_edge_length = np.array(start_edge_length)
                farthest = random.choices(range(len(start_edge_length)), weights=start_edge_length)[0]
                farthest = np.argmax(start_edge_length)
                return self.landmarks[farthest,:self.dim] + np.random.uniform(low=-3, high=3, size=self.args.subgoal_dim)
        else:
            goal_edge_length = self.dist_to_graph(subgoal, self.landmarks)
            for i in range(self.n_graph_node):
                if goal_edge_length[i] < self.init_dist * 1.01:
                    if self.graph.nodes[i]['success_count'] > 0:
                        if self.deleted_node:
                            for i in self.deleted_node:
                                for j in range(self.n_graph_node):
                                    if i != j:
                                        threshold = np.max([expanded_graph.nodes[i]['distance'], expanded_graph.nodes[j]['distance']])
                                        if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) < threshold * 1.01:
                                            if expanded_graph.has_edge(i, j):
                                                expanded_graph[i][j]['weight'] = 10000.
                                                expanded_graph[j][i]['weight'] = 10000.
                        start_to_goal_length = np.linalg.norm(ag - subgoal)
                        if start_to_goal_length < 4.0:
                            expanded_graph.add_edge('start', 'goal', weight = 1.)
                            
                        start_edge_length = self.dist_to_graph(ag, self.landmarks)
                        goal_edge_length = self.dist_to_graph(subgoal, self.landmarks)
                        
                        self.edge_lengths = [] 
                        
                        for i in range(self.n_graph_node):
                            if start_edge_length[i] < self.init_dist * 1.01:
                                if i not in self.disconnected:
                                    expanded_graph.add_edge('start', i, weight = start_edge_length[i])
                                # else:
                                #     expanded_graph.add_edge('start', i, weight = 10000.)
                            if goal_edge_length[i] < self.init_dist * 1.01:
                                if i not in self.disconnected:
                                    expanded_graph.add_edge(i, 'goal', weight = goal_edge_length[i])
                                # else:
                                #     expanded_graph.add_edge(i, 'goal', weight = 10000.)
                        if (not expanded_graph.has_node('start')):
                            added = False
                            adjusted = 1.5
                            while True:
                                adjusted_cutoff = self.init_dist * adjusted
                                for i in range(self.n_graph_node):
                                    if(start_edge_length[i] < adjusted_cutoff):
                                        if i not in self.disconnected:
                                            expanded_graph.add_edge('start', i, weight = start_edge_length[i])
                                            added = True
                                if added:
                                    break
                                adjusted += 0.5
                        return self.get_goal_candi(expanded_graph)
            return None
    
    def dist_from_graph_to_goal(self, subgoal):
        dist_list=[]
        for i in range(subgoal.shape[0]):  
            curr_subgoal = subgoal[i,:self.dim]
            if self.args.use_oracle_G:
                goal_edge_length = self._get_dist_to_goal_oracle(self.states, curr_subgoal)
            else:
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, curr_subgoal)
            dist_list.append(min(goal_edge_length))
        return np.array(dist_list)
    
    def dist_to_graph(self, node, landmarks):
        return np.linalg.norm(node[:self.dim]-landmarks, axis = 1)
            
    
    def get_waypoint(self, ob, ag, subgoal, bg, train=False):
        if self.graph is not None:
            if self.env.env_name == 'AntMazeCB' or self.env.env_name == 'AntMazeCBcomplex' or self.env.env_name == 'Reacher3D-v0':
                self.waypoint_chase_step += 1

                if (self.waypoint_idx >= len(self.waypoint_vec)):
                    waypoint_subgoal = ag[:self.dim]
                    return waypoint_subgoal
                elif self.waypoint_idx > 100:
                    waypoint_subgoal = ag[:self.dim]
                    return waypoint_subgoal
                
                i = self.waypoint_vec[self.waypoint_idx]
                if ((np.linalg.norm(ag[:self.dim] - self.landmarks[i][:self.dim]) < self.achieve_threshold)):
                    
                    if (self.waypoint_idx >= len(self.waypoint_vec)-1):
                        self.update_experience(is_success = True, final = True)
                    else:
                        self.update_experience(is_success = True, final = False)
                        
                    self.waypoint_idx += 1
                    self.waypoint_chase_step = 0
                    if (self.waypoint_idx >= len(self.waypoint_vec)):
                        waypoint_subgoal = ag[:self.dim]
                    else:
                        waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
                elif ((self.waypoint_chase_step > self.waypoint_chase_step_threshold)):
                    self.update_experience(is_success = False)
                    self.waypoint_idx = 1000
                    self.waypoint_chase_step = 0
                    waypoint_subgoal = ag[:self.dim]
                else:
                    waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
                return waypoint_subgoal
            
            elif self.args.method == 'grid8':
                self.waypoint_chase_step += 1
                if(self.waypoint_idx >= len(self.waypoint_vec)):
                    waypoint_subgoal = bg
                else:
                    i = self.waypoint_vec[self.waypoint_idx]
                
                    if self.env.env_name == 'Reacher3D-v0':
                        dist_threshold = 0.1
                    else:
                        dist_threshold = 0.5
                    
                    if((np.linalg.norm(ag[:self.dim]-self.landmarks[i][:self.dim]) < dist_threshold)):
                        if train:
                            self.graph.nodes[i]['attempt_count'] += 1
                            self.graph.nodes[i]['success_count'] += 1
                            self.graph.nodes[i]['before'] = 0
                        
                        self.waypoint_idx += 1
                        self.waypoint_chase_step = 0
                    
                    elif((self.waypoint_chase_step > self.waypoint_chase_step_threshold)):
                        if train:
                            self.graph.nodes[i]['attempt_count'] += 1
                        if (self.graph.nodes[i]['success_count'] == 0) or (self.graph.nodes[i]['success_count'] * 50 <= self.graph.nodes[i]['attempt_count']):
                            if train:
                                if self.graph.nodes[i]['attempt_count'] > self.waypoint_count_threshold:
                                    self.disconnected.append(i)
                                    for j in range(self.n_graph_node):
                                        if i != j:
                                            if self.graph.has_edge(i, j):
                                                self.graph[i][j]['weight'] = 10000.
                                                self.graph[j][i]['weight'] = 10000.
                                                    
                                    self.find_path(ob, subgoal, ag, bg)
                                else:
                                    self.deleted_node.append(i)
                                    self.find_path(ob, subgoal, ag, bg)
                            else:
                                self.deleted_node.append(i)
                                self.find_path(ob, subgoal, ag, bg)
                        else:
                            self.deleted_node.append(i)
                            self.find_path(ob, subgoal, ag, bg)
                            
                    if(self.waypoint_idx >= len(self.waypoint_vec)):
                        waypoint_subgoal = bg
                    else:
                        waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
        else:
            waypoint_subgoal = subgoal
        return waypoint_subgoal

    
    def draw_edge_graph(self):
        plt.cla()
        new_graph = nx.Graph()
        new_graph.add_edges_from((u,v,d) for u,v,d in self.graph.edges(data=True) if d['visit_count'] > 0)
        # edge_colors = [self.graph[edge[0]][edge[1]]['visit_count'] for edge in self.graph.edges()]
        edge_colors = [new_graph[edge[0]][edge[1]]['visit_count'] for edge in new_graph.edges()]
        # print(np.max(edge_colors))
        pos = {idx: (landmark[0], landmark[1]) for idx, landmark in enumerate(self.landmarks)}
        nx.draw_networkx_nodes(self.graph, pos, node_size=5, node_color='k')
        edge_colloection = nx.draw_networkx_edges(new_graph, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=1, arrows=False, style='-')
        plt.colorbar(edge_colloection)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    def draw_graph(self, start=None, subgoal=None, goal=None):
        if self.env.env_name == 'Reacher3D-v0':
            map_size = [-1.5, 1.5]
            
            cube = np.array([
                np.array([[0.35, -1.1, -0.55], [1.25, -1.1, -0.55], [1.25, 1.1, -0.55], [0.35, 1.1, -0.55], [0.35, -1.1, -0.55]]),
                np.array([[0.35, -1.1, -0.25], [1.25, -1.1, -0.25], [1.25, 1.1, -0.25], [0.35, 1.1, -0.25], [0.35, -1.1, -0.25]]), 
                np.array([[0.35, -1.1, -0.55], [0.35, -1.1, -0.25]]),
                np.array([[1.25, -1.1, -0.55], [1.25, -1.1, -0.25]]),
                np.array([[0.35, 1.1, -0.55], [0.35, 1.1, -0.25]]),
                np.array([[1.25, 1.1, -0.55], [1.25, 1.1, -0.25]])
            ], dtype=object)
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.set_xlim([1.5, -1.5])
            ax1.set_ylim(map_size)
            ax1.set_zlim(map_size)
            
            
            x_vertex = []
            y_vertex = []
            z_vertex = []
            for landmark in self.landmarks:
                if self.env.env_name == 'Reacher3D-v0':
                    x_vertex.append(landmark[0])
                    y_vertex.append(landmark[1])
                    z_vertex.append(landmark[2])
            for surface in cube:
                ax1.plot(surface[:, 0], surface[:, 1], surface[:, 2], c='b')
            
            if len(self.disconnected) != 0:
                x_disconnected = []
                y_disconnected = []
                z_disconnected = []
                for landmark in self.disconnected:
                    if self.env.env_name == 'Reacher3D-v0':
                        x_disconnected.append(self.landmarks[landmark][0])
                        y_disconnected.append(self.landmarks[landmark][1])
                        z_disconnected.append(self.landmarks[landmark][2])
                    print(f'{self.landmarks[landmark][0]}, {self.landmarks[landmark][1]}, {self.landmarks[landmark][2]}')
                ax1.scatter(x_disconnected, y_disconnected, z_disconnected, c='k', marker='o', alpha=1)
            else:
                edges = self.graph.edges(data=True)
                x_edges = []
                y_edges = []
                z_edges = []
                for edge in edges:
                    node1, node2 = edge[0], edge[1]
                    x1 = self.landmarks[node1][0]
                    x2 = self.landmarks[node2][0]
                    y1 = self.landmarks[node1][1]
                    y2 = self.landmarks[node2][1]
                    z1 = self.landmarks[node1][2]
                    z2 = self.landmarks[node2][2]
                    if self.graph[node1][node2]['weight'] <= 1.0:
                        x_edges.append((x1, x2))
                        y_edges.append((y1, y2))
                        z_edges.append((z1, z2))

                x_edges = np.array(x_edges)
                y_edges = np.array(y_edges)
                z_edges = np.array(z_edges)
                
                ax1.scatter(x_vertex, y_vertex, z_vertex, c='k', marker='o', alpha=1)
                
                for i in range(x_edges.shape[0]):                
                    ax1.plot(x_edges[i], y_edges[i], z_edges[i], c='k', alpha=0.2)
                
                # for cedge in cube:
                #     ax1.plot(cedge[0], cedge[1], cedge[2], c='r')

            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png')
            buf1.seek(0)
            image1 = Image.open(buf1)
            numpy_array1= np.array(image1)
            plt.close()
            # Second Graph
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([1.5, -1.5])
            ax.set_ylim(map_size)
            ax.set_zlim(map_size)
            x_waypoint = []
            y_waypoint = []
            z_waypoint = []
            x_waypoint_edges = []
            y_waypoint_edges = []
            z_waypoint_edges = []
    
            if start is not None and subgoal is not None:
                bef_x_waypoint = start[0]
                bef_y_waypoint = start[1]
                bef_z_waypoint = start[2]
                    
                for waypoint_idx in self.waypoint_vec:
                    if waypoint_idx < self.n_graph_node:
                        waypoint_subgoal = self.landmarks[waypoint_idx][:self.dim]
                        
                        x_waypoint.append(waypoint_subgoal[0])
                        y_waypoint.append(waypoint_subgoal[1])
                        z_waypoint.append(waypoint_subgoal[2])
                        x_waypoint_edges.append((bef_x_waypoint, waypoint_subgoal[0]))
                        y_waypoint_edges.append((bef_y_waypoint, waypoint_subgoal[1]))
                        z_waypoint_edges.append((bef_z_waypoint, waypoint_subgoal[2]))
                        
                        bef_x_waypoint = waypoint_subgoal[0]
                        bef_y_waypoint = waypoint_subgoal[1]
                        bef_z_waypoint = waypoint_subgoal[2]
                
                x_waypoint_edges.append((bef_x_waypoint, subgoal[0]))
                y_waypoint_edges.append((bef_y_waypoint, subgoal[1]))
                z_waypoint_edges.append((bef_z_waypoint, subgoal[2]))
                
                x_waypoint_edges = np.array(x_waypoint_edges)
                y_waypoint_edges = np.array(y_waypoint_edges)
                z_waypoint_edges = np.array(z_waypoint_edges)
                
                
                for i in range(x_waypoint_edges.shape[0]):                
                    ax.plot(x_waypoint_edges[i], y_waypoint_edges[i], z_waypoint_edges[i], c='k', alpha=1)
                ax.scatter(x_waypoint, y_waypoint, z_waypoint, c='g', marker='o')   
                
                ax.scatter(start[0], start[1], start[2], c='r', marker='o')
                ax.scatter(subgoal[0], subgoal[1], subgoal[2], c='b', marker='o')

                for surface in cube:
                    ax.plot(surface[:, 0], surface[:, 1], surface[:, 2], c='#A32929')
                
            if goal is not None:
                x_goal, y_goal, z_goal = goal[0] , goal[1], goal[2]
                ax.scatter([x_goal], [y_goal], [z_goal], c='k', marker='o', alpha=1)
            ax.scatter(x_vertex, y_vertex, z_vertex, c='k', marker='o', alpha=0.05)
            # ax.plot(x_edges.T, y_edges.T, c='k', alpha=0.1)
            # if wall_x is not None:
            #     ax.plot(wall_x, wall_y, c ='g')
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            numpy_array = np.array(image)
            plt.close()
            return numpy_array, numpy_array1
        else:
            map_size = [-4, 20]
            wall_x = None
            wall_y = None
            if self.env.env_name == 'AntMaze':
                # -4~ 20
                map_size = [-6, 22]
                wall_x = [-4, 20, 20, -4, -4, 12, 12, -4, -4]
                wall_y = [-4, -4, 20, 20, 12, 12, 4, 4 , -4]
                Map_x, Map_y = (24, 24)
                start_x, start_y = (4, 4)
            elif self.env.env_name == 'AntMazeBottleneck':
                map_size = [-8, 24]
                wall_x = [-4, 20, 20, 17, 17, 20, 20, -4, -4, 12, 12, 15, 15, 12, 12, -4, -4]
                wall_y = [-4, -4,  7,  7,  9,  9, 20, 20, 12, 12,  9,  9,  7,  7,  4,  4, -4]
                Map_x, Map_y = (24, 24)
                start_x, start_y = (4, 4)
            elif self.env.env_name == 'AntMazeMultiPathBottleneck':
                map_size = [-8, 24]
                wall_x = [ 4,  4, 15, 15, 4]
                wall_y = [ 4, 12, 12, 4, 4]
                wall_x2 = [17, 20, 20, -4, -4, 20, 20, 17, 17]
                wall_y2 = [12, 12, 20, 20, -4, -4, 4, 4, 12]
            elif self.env.env_name == 'AntMazeSmall-v0':
                # -2 ~ 12
                map_size = [-2, 12]
                Map_x, Map_y = (12, 12)
                start_x, start_y = (2,2)
            elif self.env.env_name == 'AntMazeS':
                map_size = [-6, 38]
                wall_x = [-4, 36, 36,  4,  4, 36, 36, -4, -4, 28, 28, -4, -4]
                wall_y = [-4, -4, 20, 20, 28, 28, 36, 36, 12, 12,  4,  4, -4]
                Map_x, Map_y = (40, 40)
                start_x, start_y = (4, 4)
            elif self.env.env_name == 'AntMazeW':
                map_size_x = [-6, 38]
                map_size_y = [-14, 30]
                wall_x = [ -4,  36, 36, -4, -4,  4,  4, 28, 28, 12, 12, 28, 28,  4,  4, -4, -4]
                wall_y = [-12, -12, 28, 28, 12, 12, 20, 20, 12, 12,  4,  4, -4, -4,  4,  4, -12]
                Map_x, Map_y = (40, 40)
                start_x, start_y = (4, 12)
            elif self.env.env_name == 'AntMazeP':
                map_size_x = [-16, 32]
                map_size_y = [-8, 40]
                wall_x = [-12,  4,  4, -4, -4,  4,  4, 12, 12, 20, 20, 12, 12, 28, 28, 20, 20, 28, 28, -12, -12, -4, -4, -12, -12]
                wall_y = [ -4, -4,  4,  4, 12, 12, 28, 28, 12, 12,  4,  4, -4, -4, 20, 20, 28, 28, 36,  36,  28, 28, 20,  20,  -4]
                Map_x, Map_y = (40, 40)
                start_x, start_y = (12, 4)
            elif self.env.env_name == 'AntMazeMultiPath-v0':
                # -2 ~ 12
                map_size = [-2, 12]
                Map_x, Map_y = (12, 12)
                start_x, start_y = (6,2)    
            elif self.env.env_name == 'AntMazeCB' or self.env.env_name == 'AntMazeCBcomplex':
                map_size_x = [-6, 38]
                map_size_y = [-6, 22]
                wall_x = [17, 17, 28, 28, 17]
                wall_y = [ 4, 12, 12,  4,  4]
                wall_x2 = [-4, -4, 15, 15, -4, -4, 36, 36, -4]
                wall_y2 = [-4,  4,  4, 12, 12, 20, 20, -4, -4]
                Map_x, Map_y = (40, 24)
                start_x, start_y = (4, 4)
            elif self.env.env_name == 'AntMazeComplex-v0':
                # -4 ~ 52
                map_size = [-4, 52]
                wall_x = [-4, -4, 12, 12, -4, -4, 4, 4, 12, 12, 20, 20, 28, 28, 52, 52, 44, 44, 36, 36, 44, 44, 52, 52, 28, 28, 36, 36, 28, 28, 20, 20, 12, 12, 4, 4, 20, 20, -4]
                wall_y = [-4, 4, 4, 12, 12, 52, 52, 44, 44, 52, 52, 44, 44, 52, 52, 36, 36, 44, 44, 28, 28, 12, 12, -4, -4, 12, 12, 20, 20, 36, 36, 28, 28, 36, 36, 20, 20, -4, -4]
                Map_x, Map_y = (56, 56)
                start_x, start_y = (4, 4)
            # First Graph
            fig1, ax1 = plt.subplots()
            if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW' or self.env.env_name == 'AntMazeCB'or self.env.env_name == 'AntMazeCBcomplex':
                ax1.set_xlim(map_size_x)
                ax1.set_ylim(map_size_y)
            else:
                ax1.set_xlim(map_size)
                ax1.set_ylim(map_size)
            x_vertex = []
            y_vertex = []
            for landmark in self.landmarks:
                x_vertex.append(landmark[0])
                y_vertex.append(landmark[1])

            edges = self.graph.edges(data=True)
            x_edges = []
            y_edges = []
            for edge in edges:
                node1, node2 = edge[0], edge[1]
                if self.graph[node1][node2]['weight'] < 100.:
                    # if node1 < node2:
                    x_edges.append((self.landmarks[node1][0], self.landmarks[node2][0]))
                    y_edges.append((self.landmarks[node1][1], self.landmarks[node2][1]))
            
            x_edges = np.array(x_edges)
            y_edges = np.array(y_edges)
            
            ax1.scatter(x_vertex, y_vertex, c='k', marker='o', alpha=1)
            ax1.plot(x_edges.T, y_edges.T, c='k', alpha=0.2)
            
            ax1.plot(wall_x, wall_y, c ='k')
            
            if self.env.env_name == 'AntMazeMultiPathBottleneck' or self.env.env_name == 'AntMazeCB'or self.env.env_name == 'AntMazeCBcomplex':
                ax1.plot(wall_x2, wall_y2, c='k')
            
            buf1 = io.BytesIO()
            
            fig1.savefig(buf1, format='png')
            buf1.seek(0)
            image1 = Image.open(buf1)
            numpy_array1= np.array(image1)
            plt.close()
            # Second Graph
            fig, ax = plt.subplots()
            if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW' or self.env.env_name == 'AntMazeCB'or self.env.env_name == 'AntMazeCBcomplex':
                ax.set_xlim(map_size_x)
                ax.set_ylim(map_size_y)
            else:
                ax.set_xlim(map_size)
                ax.set_ylim(map_size)
            x_waypoint = []
            y_waypoint = []
            x_waypoint_edges = []
            y_waypoint_edges = []
    
            if start is not None and subgoal is not None:
                bef_x_waypoint = start[0]
                bef_y_waypoint = start[1]

                for waypoint_idx in self.waypoint_vec:
                    if waypoint_idx < self.n_graph_node:
                        waypoint_subgoal = self.landmarks[waypoint_idx][:self.dim]
                        x_waypoint.append(waypoint_subgoal[0])
                        y_waypoint.append(waypoint_subgoal[1])
                        x_waypoint_edges.append((bef_x_waypoint, waypoint_subgoal[0]))
                        y_waypoint_edges.append((bef_y_waypoint, waypoint_subgoal[1]))
                        bef_x_waypoint = waypoint_subgoal[0]
                        bef_y_waypoint = waypoint_subgoal[1]

                x_waypoint_edges.append((bef_x_waypoint, subgoal[0]))
                y_waypoint_edges.append((bef_y_waypoint, subgoal[1]))
                ax.plot(x_waypoint_edges, y_waypoint_edges, c='k', alpha=1)
                ax.scatter(x_waypoint, y_waypoint, c='g', marker='o')   
                ax.scatter(start[0], start[1], c='r', marker='o')
                ax.scatter(subgoal[0], subgoal[1], c='b', marker='o')

            if goal is not None:
                x_goal, y_goal = goal[0], goal[1]
                ax.scatter([x_goal], [y_goal], c='k', marker='o', alpha=1)
            ax.scatter(x_vertex, y_vertex, c='k', marker='o', alpha=0.1)
            ax.plot(x_edges.T, y_edges.T, c='k', alpha=0.1)
            if wall_x is not None:
                ax.plot(wall_x, wall_y, c ='k')
            if self.env.env_name == 'AntMazeMultiPathBottleneck' or self.env.env_name == 'AntMazeCB'or self.env.env_name == 'AntMazeCBcomplex':
                ax.plot(wall_x2, wall_y2, c='k')
                
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            numpy_array = np.array(image)
            plt.close()
            return numpy_array, numpy_array1

    #####################oracle graph#########################
    def _get_dist_to_goal_oracle(self, obs_tensor, goal):
        goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
            * np.expand_dims(goal[:self.args.subgoal_dim], axis=0)
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - goal_repeat, axis=1)
        return dist

    def _get_dist_from_start_oracle(self, start, obs_tensor):
        start_repeat = np.ones_like(obs_tensor) * np.expand_dims(start, axis=0)
        start_repeat = start_repeat[:, :self.args.subgoal_dim]
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - start_repeat, axis=1)
        return dist

    def _get_point_to_point_oracle(self, point1, point2):
        point1 = point1[:self.args.subgoal_dim]
        point2 = point2[:self.args.subgoal_dim]
        dist = np.linalg.norm(point1-point2)
        return dist

    def _get_pairwise_dist_oracle(self, obs_tensor):
        goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = np.ones_like(goal_tensor) * np.expand_dims(obs, axis=0)
            dist = np.linalg.norm(obs_repeat_tensor[:, :self.args.subgoal_dim] - goal_tensor[:, :self.args.subgoal_dim], axis=1)
            dist_matrix.append(np.squeeze(dist))
        pairwise_dist = np.array(dist_matrix) #pairwise_dist[i][j] is dist from i to j
        return pairwise_dist