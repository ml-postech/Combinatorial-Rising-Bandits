import gym
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from gym import spaces
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

class AntMazeBottleneckEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    xml_filename = 'ant_maze_bottleneck.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([0,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0
        self.env_name = 'AntMazeBottleneck'
        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)


        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15],
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    
    def rand_goal(self):
        while True:
            self.goal = np.random.uniform(low=-4., high=20., size=2)
            if not ((self.goal[0] < 12) and (self.goal[1] > 4) and (self.goal[1] < 12)):
                break


    def reset_model(self):
        self.rand_goal()
        self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

class AntMazeBottleneckEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle): 
    xml_filename = 'ant_maze_bottleneck.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([0,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0
        self.env_name = 'AntMazeBottleneck'

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15], 
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    

    def reset_model(self):
        self.goal = np.array([0., 16.])
        self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()
    
    def change_goal(self, x = 0, y = 0, size=2):
        self.goal = np.array([x, y]) + np.random.uniform(low=-size, high=size, size=2)
        self.set_goal("goal_point")
        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('goal_point')] = np.array([goal[0], goal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('subgoal_point')] = np.array([subgoal[0], subgoal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('way_point')] = np.array([waypoint[0], waypoint[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
        return self.render(mode='rgb_array', width=500, height=500)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance= 50
        self.viewer.cam.elevation = -90
        #self.viewer.cam.distance = self.physics.stat.extent


        
class AntMazeMultiPathBottleneckEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    xml_filename = 'ant_maze_multipath_bottleneck.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([8,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0
        self.env_name = 'AntMazeMultiPathBottleneck'
        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)


        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15],
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    
    def rand_goal(self):
        while True:
            self.goal = np.random.uniform(low=-4., high=20., size=2)
            if not ((self.goal[0] > 4) and (self.goal[1] > 4) and (self.goal[1] < 12)):
            # if not ((((self.goal[0] > 4) and (self.goal[0] < 15.2)) or (self.goal[0] > 16.8))and (self.goal[1] > 4) and (self.goal[1] < 12)):          
                break


    def reset_model(self):
        self.rand_goal()
        self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

class AntMazeMultiPathBottleneckEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle): 
    xml_filename = 'ant_maze_multipath_bottleneck.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([8,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0
        self.env_name = 'AntMazeMultiPathBottleneck'

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15], 
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    

    def reset_model(self):
        self.goal = np.array([16., 16.])
        self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('goal_point')] = np.array([goal[0], goal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('subgoal_point')] = np.array([subgoal[0], subgoal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('way_point')] = np.array([waypoint[0], waypoint[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
        return self.render(mode='rgb_array', width=500, height=500)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance= 50
        self.viewer.cam.elevation = -90
        #self.viewer.cam.distance = self.physics.stat.extent
        
        
class AntMazeCBEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    xml_filename = 'ant_maze_CB_small.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([0,0])
    
    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 100
        self.nb_step = 0
        self.env_name = 'AntMazeCB'
        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()
        self.evaluate = False
        self.coverage = False
        self.curriculum = False
        self.setting = 'CB1'
        self.failure_count = 0
        self.goal = None
        self.distance_threshold = 0.5 if self.env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeP', 'AntMazeW','AntMazeS', 'AntPush', 'AntFall', 'AntMazeCB'] else 1
        self.count = 0
        self.early_stop = False if self.env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeP', 'AntMazeW','AntMazeS', 'AntPush', 'AntFall', 'AntMazeCB'] else True
        self.early_stop_flag = False
        
    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15],
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    
    def rand_goal(self):
        while True:
            self.goal = np.random.uniform(low=-4., high=20., size=2)
            if not ((self.goal[0] > 4) and (self.goal[1] > 4) and (self.goal[1] < 12)):
            # if not ((((self.goal[0] > 4) and (self.goal[0] < 15.2)) or (self.goal[0] > 16.8))and (self.goal[1] > 4) and (self.goal[1] < 12)):          
                break

    def reset_model(self):
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        init = np.array([0., 0.])
        self.goal = np.array([0., 8.]) 
        if self.setting == 'CB1':
            init = np.array([0., 0.])
            self.goal = np.array([0., 8.])  
        elif self.setting == 'CB2':
            init = np.array([8., 0.])
            self.goal = np.array([8., 8.])
        elif self.setting == 'CB3':
            init = np.array([16., 0.])
            self.goal = np.array([16., 8.])
        else:
            self.rand_goal()
        
        self.init_qpos[:2] = init
        qpos[:2] = init        
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.set_goal("goal_point")
        # qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        # qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        # self.init_qpos[:2] = self.init_xy
        # qpos[:2] = self.init_xy

        # qpos[15:] = self.init_qpos[15:]
        # qvel[14:] = 0.
        # self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        
        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]


        waypoint = np.array([[16, 32, 16, 32], [0, 0, 16, 16]])
        for i in range(4):
            body_ids = self.model.body_name2id(f'way_point{i}')
            self.model.body_pos[body_ids][:2] = waypoint[:,i]
            self.model.body_quat[body_ids] = [0., 1., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            self.sim.data.site_xpos[self.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            self.sim.data.site_xpos[self.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('way_point')] = np.array([waypoint[0], waypoint[1], 0])
            for i in range(4):
                body_ids = self.model.body_name2id(f'way_point{i}')
                self.model.body_pos[body_ids][:2] = waypoint[:,i]
                self.model.body_quat[body_ids] = [1., 0., 0., 0.]
                # self.sim.data.site_xpos[self.model.site_name2id()] = np.array([waypoint[0][i], waypoint[1][i], 0])
        # if waypoint is not None:
        #     self.sim.data.site_xpos[self.model.site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
        return self.render(mode='rgb_array', width=7000, height=5000)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance= 40
        self.viewer.cam.elevation = -80
        current_pos = self.viewer.cam.lookat
        current_pos[0] += 8
        current_pos[1] += 4
        # current_pos[2] += 30
        self.viewer.cam.lookat[:] = current_pos
        #self.viewer.cam.distance = self.physics.stat.extent

class AntMazeCBEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle): 
    xml_filename = 'ant_maze_CB.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([8,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0
        self.env_name = 'AntMazeCB'

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15], 
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    

    def reset_model(self):
        self.base_env.wrapped_env.set_xy(np.array([0., 0.]))
        obs[:2] = np.array([0., 0.])
        self.goal = np.array([0., 16.])
        if self.setting == 'CB1':
            self.base_env.wrapped_env.set_xy(np.array([0., 0.]))
            obs[:2] = np.array([0., 0.])
            self.goal = np.array([0., 16.])  
        elif self.setting == 'CB2':
            self.base_env.wrapped_env.set_xy(np.array([32., 0.]))
            obs[:2] = np.array([16., 0.])
            self.goal = np.array([16., 16.])
        elif self.setting == 'CB3':
            self.base_env.wrapped_env.set_xy(np.array([32., 0.]))
            obs[:2] = np.array([32., 0.])
            self.goal = np.array([32., 16.])
        else:
            self.rand_goal()
        self.set_goal("goal_point")
        # qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        # qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        # self.init_qpos[:2] = self.init_xy
        # qpos[:2] = self.init_xy

        # qpos[15:] = self.init_qpos[15:]
        # qvel[14:] = 0.
        # self.set_state(qpos, qvel)
        # self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('goal_point')] = np.array([goal[0], goal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('subgoal_point')] = np.array([subgoal[0], subgoal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('way_point')] = np.array([waypoint[0], waypoint[1], 0])
            for i in range(4):
                self.sim.data.site_xpos[self.model.site_name2id(f'way_point:box{i}')] = np.array([waypoint[0][i], waypoint[1][i], 0])
        # if waypoint is not None:
        #     self.sim.data.site_xpos[self.model.site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
        return self.render(mode='rgb_array', width=500, height=500)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance= 50
        self.viewer.cam.elevation = -70
        #self.viewer.cam.distance = self.physics.stat.extent
        
        
class AntMazeCBcomplexEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    xml_filename = 'ant_maze_CB_complex.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([0,0])
    
    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 100
        self.nb_step = 0
        self.env_name = 'AntMazeCBcomplex'
        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()
        self.evaluate = False
        self.coverage = False
        self.curriculum = False
        self.setting = 'CB1'
        self.failure_count = 0
        self.goal = None
        self.distance_threshold = 0.5 if self.env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeP', 'AntMazeW','AntMazeS', 'AntPush', 'AntFall', 'AntMazeCB', 'AntMazeCBcomplex'] else 1
        self.count = 0
        self.early_stop = False if self.env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeMultiPath-v0', 'AntMazeComplex-v0', 'AntMazeP', 'AntMazeW','AntMazeS', 'AntPush', 'AntFall', 'AntMazeCB', 'AntMazeCBcomplex'] else True
        self.early_stop_flag = False
        
    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15],
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    
    def rand_goal(self):
        while True:
            self.goal = np.random.uniform(low=-4., high=20., size=2)
            if not ((self.goal[0] > 4) and (self.goal[1] > 4) and (self.goal[1] < 12)):
            # if not ((((self.goal[0] > 4) and (self.goal[0] < 15.2)) or (self.goal[0] > 16.8))and (self.goal[1] > 4) and (self.goal[1] < 12)):          
                break

    def reset_model(self):
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        init = np.array([0., 0.])
        self.goal = np.array([0., 24.]) 
        if self.setting == 'CB1':
            init = np.array([0., 0.])
            self.goal = np.array([0., 24.])  
        elif self.setting == 'CB2':
            init = np.array([8., 0.])
            self.goal = np.array([8., 8.])
        elif self.setting == 'CB3':
            init = np.array([16., 0.])
            self.goal = np.array([16., 8.])
        else:
            self.rand_goal()
        
        self.init_qpos[:2] = init
        qpos[:2] = init        
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.set_goal("goal_point")
        # qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        # qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        # self.init_qpos[:2] = self.init_xy
        # qpos[:2] = self.init_xy

        # qpos[15:] = self.init_qpos[15:]
        # qvel[14:] = 0.
        # self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        
        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            self.sim.data.site_xpos[self.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        # if subgoal is not None:
        #     self.sim.data.site_xpos[self.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            for i in range(14):
                body_ids = self.model.body_name2id(f'way_point{i}')
                self.model.body_pos[body_ids][:2] = waypoint[:,i]
                self.model.body_quat[body_ids] = [1., 0., 0., 0.]
        return self.render(mode='rgb_array', width=7000, height=5000)
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance= 70
        self.viewer.cam.elevation = -80
        current_pos = self.viewer.cam.lookat
        current_pos[0] += 12
        current_pos[1] += 12
        # current_pos[2] += 30
        self.viewer.cam.lookat[:] = current_pos
        #self.viewer.cam.distance = self.physics.stat.extent

class AntMazeCBcomplexEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle): 
    xml_filename = 'ant_maze_CB_complex.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([8,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 2000
        self.nb_step = 0
        self.env_name = 'AntMazeCB'

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15], 
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    

    def reset_model(self):
        self.base_env.wrapped_env.set_xy(np.array([0., 0.]))
        obs[:2] = np.array([0., 0.])
        self.goal = np.array([0., 24.])
        if self.setting == 'CB1':
            self.base_env.wrapped_env.set_xy(np.array([0., 0.]))
            obs[:2] = np.array([0., 0.])
            self.goal = np.array([0., 16.])  
        elif self.setting == 'CB2':
            self.base_env.wrapped_env.set_xy(np.array([32., 0.]))
            obs[:2] = np.array([16., 0.])
            self.goal = np.array([16., 16.])
        elif self.setting == 'CB3':
            self.base_env.wrapped_env.set_xy(np.array([32., 0.]))
            obs[:2] = np.array([32., 0.])
            self.goal = np.array([32., 16.])
        else:
            self.rand_goal()
        self.set_goal("goal_point")
        # qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        # qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        # self.init_qpos[:2] = self.init_xy
        # qpos[:2] = self.init_xy

        # qpos[15:] = self.init_qpos[15:]
        # qvel[14:] = 0.
        # self.set_state(qpos, qvel)
        # self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist

    def get_image(self, goal=None, subgoal=None, waypoint=None):
        if goal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('goal_point')] = np.array([goal[0], goal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('goal_point:box')] = np.array([goal[0], goal[1], 0])
        if subgoal is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('subgoal_point')] = np.array([subgoal[0], subgoal[1], 0])
            self.sim.data.site_xpos[self.model.site_name2id('subgoal_point:box')] = np.array([subgoal[0], subgoal[1], 0])
        if waypoint is not None:
            #self.base_env.wrapped_env.sim.data.body_xpos[self.base_env.wrapped_env.model.body_name2id('way_point')] = np.array([waypoint[0], waypoint[1], 0])
            for i in range(4):
                self.sim.data.site_xpos[self.model.site_name2id(f'way_point:box{i}')] = np.array([waypoint[0][i], waypoint[1][i], 0])
        # if waypoint is not None:
        #     self.sim.data.site_xpos[self.model.site_name2id('way_point:box')] = np.array([waypoint[0], waypoint[1], 0])
        return self.render(mode='rgb_array', width=500, height=500)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance= 50
        self.viewer.cam.elevation = -70
        #self.viewer.cam.distance = self.physics.stat.extent