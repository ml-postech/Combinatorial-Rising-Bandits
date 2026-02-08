import numpy as np
import argparse
from collections import deque
from gym import spaces

#import create_maze_env



def get_reward_fn(env_name): # we don't use this function
    if env_name in ['AntMaze', 'AntMazeBottleneck', 'AntMazeMultiPathBottleneck', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeS', 'AntMazeP', 'AntMazeW', 'AntPush', 'AntMazeCB']:
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntMazeSparse':
        return lambda obs, goal: float(np.sum(np.square(obs[:2] - goal)) ** 0.5 < 1)
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'



def get_success_fn(env_name): # we don't use this function
    if env_name in ['AntMaze', 'AntMazeBottleneck', 'AntMazeMultiPathBottleneck', 'AntMazeComplex-v0', 'AntMazeS', 'AntMazeP', 'AntMazeW', 'AntPush', 'AntFall']:
        return lambda reward: reward > -5.0
    elif env_name in ['AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeCB']:
        return lambda reward: reward > -2.5
    elif env_name == 'AntMazeSparse':
        return lambda reward: reward > 1e-6
    else:
        assert False, 'Unknown env'


class GatherEnv(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.count = 0

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        obs = self.base_env.reset()
        self.count = 0
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }

    def step(self, a):
        obs, reward, done, info = self.base_env.step(a)
        self.count += 1
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }
        return next_obs, reward, done or self.count >= 500, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def goal_distance(self, achieved_goal, goal):
            if(achieved_goal.ndim == 1):
                dist = np.linalg.norm(goal - achieved_goal)
            else:
                dist = np.linalg.norm(goal - achieved_goal, axis=1)
                dist = np.expand_dims(dist, axis=1)
            return dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    @property
    def action_space(self):
        return self.base_env.action_space


class EnvWithGoal(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.coverage = False
        self.curriculum = False
        self.setting = 'FIFG'
        self.curriculum_goal = np.array([2., 0.])
        self.failure_count = 0
        self.success_curriculum = False
        self.success_fn = get_success_fn(env_name)
        self.goal = None
        self.distance_threshold = 0.5 if env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeP', 'AntMazeW','AntMazeS', 'AntPush', 'AntFall', 'AntMazeCB'] else 1
        self.count = 0
        self.early_stop = False if env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeP', 'AntMazeW','AntMazeS', 'AntPush', 'AntFall', 'AntMazeCB'] else True
        self.early_stop_flag = False


    def seed(self, seed):
        self.base_env.seed(seed)
        
    def rand_init(self):
        if self.env_name == 'AntMaze':
            while True:
                obs = np.random.uniform(low=-4., high=20., size=2)
                if not ((obs[0] < 12) and (obs[1] > 4) and (obs[1] < 12)):
                    break
        elif self.env_name == 'AntMazeSmall-v0':
            while True:
                obs = np.random.uniform(low=-2., high=10., size=2)
                if not ((obs[0] < 6) and (obs[1] > 2) and (obs[1] < 6)):
                    break
        elif self.env_name == 'AntMazeMultiPath-v0':
            while True:
                obs = np.random.uniform(low=-2., high=10., size=2)
                if not ((obs[0] > 2) and (obs[0] < 6) and (obs[1] > 2) and (obs[1] < 6)):
                    break
        elif self.env_name == 'AntMazeS':
            while True:
                obs = np.random.uniform(low=-4., high=36., size = 2)
                x = obs[0]
                y = obs[1]
                if not ((((x > 4) and (x < 36)) and ((y > 20) and (y < 28))) or (((x > -4) and (x < 28)) and ((y > 4) and (y < 12)))):
                    break
        elif self.env_name == 'AntMazeW':
            while True:
                obs = np.random.uniform(low=-4., high=36., size = 2)
                obs[1] = obs[1]-8.
                x = obs[0]
                y = obs[1]
                if not ((((x > -5) and (x < 13)) and ((y > 3) and (y < 13))) or\
                        (((x > 3) and (x < 29)) and ((y > -5) and (y < 5))) or\
                        (((x > 3) and (x < 29)) and ((y > 11) and (y < 21)))):
                    break
        elif self.env_name == 'AntMazeP':
            while True:
                obs = np.random.uniform(low=-4., high=36., size = 2)
                obs[0] = obs[0]-8.
                x = obs[0]
                y = obs[1]
                if not ((((x > -12) and (x < -4)) and ((y > 20) and (y < 28))) or\
                        (((x > 20) and (x < 28)) and ((y > 20) and (y < 28))) or\
                        (((x > -4) and (x < 20)) and ((y > 4) and (y < 12))) or\
                        (((x > 4) and (x < 12)) and ((y > -4) and (y < 20)))):
                    break
        elif self.env_name == 'AntMazeComplex-v0':
            while True:
                obs = np.random.uniform(low=[-4, -4], high=[52, 52], size=2)
                x = obs[0]
                y = obs[1]
                if (\
                ((((-4<x)and(x<20))or((28<x)and(x<52))) and (y<4)) or\
                ((((12<x)and(x<20))or((28<x)and(x<52))) and ((4<y)and(y<12))) or\
                ((((-4<x)and(x<20))or((36<x)and(x<44))) and ((12<y)and(y<20))) or\
                ((((-4<x)and(x<4))or((28<x)and(x<44))) and ((20<y)and(y<28))) or\
                ((((-4<x)and(x<4))or((12<x)and(x<20))or((28<x)and(x<36))) and ((28<y)and(y<36))) or\
                ((((-4<x)and(x<36))or((44<x)and(x<52))) and ((36<y)and(y<44))) or\
                ((((-4<x)and(x<4))or((12<x)and(x<20))or((28<x)and(x<52))) and ((44<y)and(y<52)))):
                    break

        else:
            raise NameError('rand goal error')
        
        return obs
        
        
    def rand_goal(self):
        if self.env_name == 'AntMaze':
            while True:
                self.goal = np.random.uniform(low=-4., high=20., size=2)
                if not ((self.goal[0] < 12) and (self.goal[1] > 4) and (self.goal[1] < 12)):
                    break
        elif self.env_name == 'AntMazeSmall-v0':
            while True:
                self.goal = np.random.uniform(low=-2., high=10., size=2)
                if not ((self.goal[0] < 6) and (self.goal[1] > 2) and (self.goal[1] < 6)):
                    break
        elif self.env_name == 'AntMazeMultiPath-v0':
            while True:
                self.goal = np.random.uniform(low=-2., high=10., size=2)
                if not ((self.goal[0] > 2) and (self.goal[0] < 6) and (self.goal[1] > 2) and (self.goal[1] < 6)):
                    break
        elif self.env_name == 'AntMazeS':
            while True:
                self.goal = np.random.uniform(low=-4., high=36., size = 2)
                x = self.goal[0]
                y = self.goal[1]
                if not ((((x > 4) and (x < 36)) and ((y > 20) and (y < 28))) or (((x > -4) and (x < 28)) and ((y > 4) and (y < 12)))):
                    break
        elif self.env_name == 'AntMazeW':
            while True:
                self.goal = np.random.uniform(low=-4., high=36., size = 2)
                self.goal[1] = self.goal[1]-8.
                x = self.goal[0]
                y = self.goal[1]
                if not ((((x > -4) and (x < 12)) and ((y > 4) and (y < 12))) or\
                        (((x > 4) and (x < 28)) and ((y > -4) and (y < 4))) or\
                        (((x > 4) and (x < 28)) and ((y > 12) and (y < 20)))):
                    break
        elif self.env_name == 'AntMazeP':
            while True:
                self.goal = np.random.uniform(low=-4., high=36., size = 2)
                self.goal[0] = self.goal[0]-8.
                x = self.goal[0]
                y = self.goal[1]
                if not ((((x > -12) and (x < -4)) and ((y > 20) and (y < 28))) or\
                        (((x > 20) and (x < 28)) and ((y > 20) and (y < 28))) or\
                        (((x > -4) and (x < 20)) and ((y > 4) and (y < 12))) or\
                        (((x > 4) and (x < 12)) and ((y > -4) and (y < 20)))):
                    break
                
                
        elif self.env_name == 'AntMazeComplex-v0':
            while True:
                self.goal = np.random.uniform(low=[-4, -4], high=[52, 52], size=2)
                x = self.goal[0]
                y = self.goal[1]
                if (\
                ((((-4<x)and(x<20))or((28<x)and(x<52))) and (y<4)) or\
                ((((12<x)and(x<20))or((28<x)and(x<52))) and ((4<y)and(y<12))) or\
                ((((-4<x)and(x<20))or((36<x)and(x<44))) and ((12<y)and(y<20))) or\
                ((((-4<x)and(x<4))or((28<x)and(x<44))) and ((20<y)and(y<28))) or\
                ((((-4<x)and(x<4))or((12<x)and(x<20))or((28<x)and(x<36))) and ((28<y)and(y<36))) or\
                ((((-4<x)and(x<36))or((44<x)and(x<52))) and ((36<y)and(y<44))) or\
                ((((-4<x)and(x<4))or((12<x)and(x<20))or((28<x)and(x<52))) and ((44<y)and(y<52)))):
                    break

        else:
            raise NameError('rand goal error')

    def reset(self, inference=False, init_position=None, goal_position=None, xi = 0, yi = 0, init=False, xg = 0, yg = 0, size = 4):
        self.early_stop_flag = False
        obs = self.base_env.reset()
        self.count = 0
        
        if self.env_name == 'AntMazeCB':
            self.base_env.wrapped_env.set_xy(np.array([0., 0.]))
            obs[:2] = np.array([0., 0.])
            self.goal = np.array([0., 8.])
            if self.setting == 'CB1':
                self.base_env.wrapped_env.set_xy(np.array([0., 0.]))
                obs[:2] = np.array([0., 0.])
                self.goal = np.array([0., 16.])
                
            if self.setting == 'CB2':
                self.base_env.wrapped_env.set_xy(np.array([16., 0.]))
                obs[:2] = np.array([16., 0.])
                self.goal = np.array([16., 16.])
                
            if self.setting == 'CB3':
                self.base_env.wrapped_env.set_xy(np.array([32., 0.]))
                obs[:2] = np.array([32., 0.])
                self.goal = np.array([32., 16.])
        elif inference:
            self.base_env.wrapped_env.set_xy(init_position)
            obs[:2] = init_position
            self.goal = goal_position
        elif self.evaluate:
            if self.coverage:
                if init:
                    self.base_env.wrapped_env.set_xy(np.array([xi, yi]))
                    obs[:2] = np.array([xi, yi])
                self.goal = np.array([xg, yg]) + np.random.uniform(low=-size, high=size, size=2)                
            else:
                if self.env_name == 'AntMaze':
                    self.goal = np.array([0., 16.])
                elif self.env_name == 'AntMazeSmall-v0':
                    self.goal = np.array([0., 8.])
                elif self.env_name == 'AntMazeMultiPath-v0':
                    self.goal = np.array([0., 8.])
                elif self.env_name == 'AntMazeS':
                    self.goal = np.array([32., 32.])
                elif self.env_name == 'AntMazeP':
                    self.goal = np.array([16., 0.])
                elif self.env_name == 'AntMazeW':
                    self.goal = np.array([0., 16.])
                elif self.env_name == 'AntMazeComplex-v0':
                    goal_seed = np.random.randint(4)
                    goal_seed = 0
                    if goal_seed == 0:
                        self.goal = np.array([40., 8.])
                    elif goal_seed == 1:
                        self.goal = np.array([16., 48.])
                    elif goal_seed == 2:
                        self.goal = np.array([40., 48.])
                    else:
                        self.goal = np.array([16., 32.])
                else:
                    raise NameError('rand goal error')
        else:
            if self.setting == 'FIFG':
                if self.env_name == 'AntMaze':
                    self.goal = np.array([0., 16.])
                elif self.env_name == 'AntMazeSmall-v0':
                    self.goal = np.array([0., 8.])
                elif self.env_name == 'AntMazeMultiPath-v0':
                    self.goal = np.array([0., 8.])
                elif self.env_name == 'AntMazeS':
                    self.goal = np.array([32., 32.])
                elif self.env_name == 'AntMazeP':
                    self.goal = np.array([16., 0.])
                elif self.env_name == 'AntMazeW':
                    self.goal = np.array([0., 16.])
                elif self.env_name == 'AntMazeComplex-v0':
                    goal_seed = np.random.randint(4)
                    goal_seed = 0
                    if goal_seed == 0:
                        self.goal = np.array([40., 8.])
                    elif goal_seed == 1:
                        self.goal = np.array([16., 48.])
                    elif goal_seed == 2:
                        self.goal = np.array([40., 48.])
                    else:
                        self.goal = np.array([16., 32.])
                else:
                    raise NameError('rand goal error')
            elif self.setting == 'FIRG':
                self.rand_goal()
            elif self.setting == 'RIRG':
                init_position = self.rand_init()

                self.base_env.wrapped_env.set_xy(init_position)
                obs[:2] = init_position
                
                self.rand_goal()
        self.desired_goal = self.goal
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.high_reward_func(obs[:2], self.goal, info) 
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        done = self.early_stop_flag and self.count % 10 == 0
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }
        if self.env_name in ['AntMaze', 'AntPush', 'AntFall', 'AntMazeMultiPathBottleneck']:
            info['is_success'] = (self.goal_distance(obs[:2], self.desired_goal) <= 5)
            done = done or self.count >= 600
        elif self.env_name in ['AntMazeCB']:
            info['is_success'] = (self.goal_distance(obs[:2], self.desired_goal) <= 0.5)
            done = done or self.count >= 100
        elif self.env_name in ['AntMazeS', 'AntMazeW', 'AntMazeP']:
            info['is_success'] = (self.goal_distance(obs[:2], self.desired_goal) <= 5)
            done = done or self.count >= 1000
        elif self.env_name in ['AntMazeComplex-v0']:
            info['is_success'] = (self.goal_distance(obs[:2], self.desired_goal) <= 5)
            done = done or self.count >= 2000
        elif self.env_name in ['AntMazeSmall-v0', 'AntMazeMultiPath-v0']:
            info['is_success'] = (self.goal_distance(obs[:2], self.desired_goal) <= 2.5)
            done = done or self.count >= 600
        return next_obs, reward, done, info


    def render(self):
        self.base_env.render()
    
    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('goal_point')] = np.array([goal[0], goal[1], 0])
            self.base_env.wrapped_env.sim.data.site_xpos[self.base_env.wrapped_env.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('subgoal_point')] = np.array([subgoal[0], subgoal[1], 0])
            self.base_env.wrapped_env.sim.data.site_xpos[self.base_env.wrapped_env.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            self.sim.data.site_xpos[self.model.site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
            
        return self.base_env.render(mode='rgb_array', width=500, height=500)

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False, threshold = None):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            if threshold is None:
                rs = (np.array(dist) > self.distance_threshold)
            else:
                rs = (np.array(dist) > threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def high_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False) * 0.5

    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist


    @property
    def action_space(self):
        return self.base_env.action_space
