import gym
import random
import numpy as np
import torch
import time
import os.path as osp
import wandb
from rl.utils.run_utils import Monitor
from rl.replay.planner import LowReplay, HighReplay
from rl.learn.dhrl import HighLearner, LowLearner
from rl.agent.agent import LowAgent, HighAgent
from rl.algo.dhrl import Algo
from rl.curriculum.dhrl import Curriculum
from rl.score.dhrl import LowScore, HighScore
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from envs.antenv import EnvWithGoal, GatherEnv
from envs.antenv.create_maze_env import create_maze_env
from envs.antenv.create_gather_env import create_gather_env
import matplotlib.pyplot as plt

def get_env_params(env, args):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'sub_goal': args.subgoal_dim,
              'l_action_dim': args.l_action_dim,
              'h_action_dim': args.h_action_dim,
              'action_max': args.action_max,
              'max_timesteps': args.max_steps}
    
    # env.set_goal("goal_point")
    # # env.reset()
    # # image = env.get_image(subgoal=[0, 8], waypoint=np.array([[8, 16, 8, 16], [0, 0, 8, 8]]))
    # image = env.get_image(waypoint=np.array([[8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24, 8, 16, 24], [0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 16, 24, 24, 24]]))
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # ax.axis('off')
    # fig.savefig('hihihi.pdf')   
    # plt.close(fig)
    
    # print('a')
    # b = input()
    return params


def launch(args):
    
    name = args.method
    name += args.setting
    if args.debug:
        name += '_debug'
    if args.inference:
        name += '_inference'
    if args.use_reverse_dist_func:
        name += '_reverse_dist'
    if args.curr_method is not None:
        name +=  '_' + args.curr_method
        if args.curr_method == 'RND':
            name += '_' + args.RND_method
    if args.low_replay:
        name += '_low_replay'
    if args.high_replay:
        name += '_high_replay'
    if args.low_agent:
        name += '_pre_low_agent'
    if args.high_agent:
        name += '_pre_high_agent'
    if args.load_epoch is not None:
        name += '_epoch' + str(args.load_epoch)
    if args.freeze:
        name += '_freeze'
    name += '_' + args.high_hindsight
    if args.ada_hindsight:
        high_name = '_' + args.high_score + '_epsilon' + str(args.epsilon)
        low_name = '_' + args.low_score
        if args.high_score == 'RND':
            if args.input_normalization:
                high_name += '_input'
            if args.score_normalization:
                high_name += '_score'
            high_name += '_' + args.high_input
        if args.low_score == 'MC_Dropout':
            low_name += str(args.dropout_prob)
        elif args.low_score == 'RND':
            if args.input_normalization:
                high_name += '_input'
            if args.score_normalization:
                high_name += '_score'
            high_name += '_' + args.high_input
        name += high_name + low_name
    if args.add_loss:
        name += '_additional_loss'
    if args.ada_score:
        name += '_ada_score'
    if args.go_explore:
        name += '_go_explore' + '_' + str(args.frontier_prob)
        if args.high_score == 'RND':
            if args.input_normalization:
                name += '_input'
            if args.score_normalization:
                high_name += '_score'
            name += '_' + args.high_score_input

    if args.method == 'gbphrl':
        name += '_' + str(args.high_future_p) 
        name += '_' + str(args.high_penalty)
        name += '_' + args.uncertainty
        if args.nosubgoal:
            name += '_' + 'planning_to_goal'
        name += '_' + args.exploitation
        name += '_' + str(args.alpha) + '_' + str(args.beta)
    if args.FGS:
        name += '_FGS'
    if args.debug:
         wandb.init(project=args.env_name, name=name, config=vars(args), sync_tensorboard=True, mode='disabled')
    else:
        wandb.init(project=args.env_name, name=name, config=vars(args), sync_tensorboard=True)
        
    wandb.define_metric('Total Timesteps')

    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ["AntMaze", "AntMazeSmall-v0", "AntMazeComplex-v0", "AntMazeMultiPath-v0", "AntMazeSparse", "AntMazeS", "AntMazeW", "AntMazeP", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        env.setting = args.setting
        test_env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
        test_env_coverage = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env_coverage.evaluate = True
        test_env_coverage.coverage = True
    else:
        env = gym.make(args.env_name)
        test_env = gym.make(args.test_env_name)
        if args.env_name == "AntMazeCB-v0" or args.env_name == "AntMazeCBcomplex-v0":
            env.setting = args.setting
            test_env.evaluate = True
        if args.env_name == "Reacher3D-v0" or args.env_name == "Pusher-v0":
            test_env.evaluate = True
        test_env_coverage = gym.make(args.test_env_name)
    seed = args.seed

    env.seed(seed)
    test_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    assert np.all(env.action_space.high == -env.action_space.low)
    env_params = get_env_params(env, args)
    low_reward_func = env.low_reward_func
    high_reward_func = env.high_reward_func
    monitor = Monitor(args.max_steps)


    ckpt_name = args.ckpt_name
    if len(ckpt_name) == 0:
        data_time = time.ctime().split()[1:4]
        ckpt_name = data_time[0] + '-' + data_time[1]
        time_list = np.array([float(i) for i in data_time[2].split(':')], dtype=np.float32)
        for time_ in time_list:
            ckpt_name += '-' + str(int(time_))
        args.ckpt_name = ckpt_name
    
    low_agent = LowAgent(env_params, args)
    high_agent = HighAgent(env_params, args)

    high_score = None
    low_score= None
    if args.ada_hindsight or args.ada_score or args.go_explore or (args.exploitation == 'rnd' and args.method == 'gbphrl'):
        print(1)
        high_score = HighScore(env_params, args, high_agent, monitor)
        low_score = LowScore(env_params, args, low_agent, monitor)

    low_replay = LowReplay(env_params, args, low_reward_func)
    high_replay = HighReplay(env_params, args, high_reward_func, monitor, low_score, high_score, high_agent)
    low_learner = LowLearner(low_agent, monitor, args)
    if args.add_loss and not args.nosubgoal:
        high_learner = HighLearner(high_agent, monitor, args, low_agent)
    else:
        high_learner = HighLearner(high_agent, monitor, args)

    algo = Algo(
        env=env, env_params=env_params, args=args,
        test_env=test_env, test_env_coverage=test_env_coverage, 
        low_agent=low_agent, high_agent = high_agent, low_replay=low_replay, high_replay=high_replay, monitor=monitor, 
        low_learner=low_learner, high_learner=high_learner,
        low_reward_func=low_reward_func, high_reward_func=high_reward_func, high_score=high_score, low_score=low_score
    )
    return algo