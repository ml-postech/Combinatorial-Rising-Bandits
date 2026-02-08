import sys
import numpy as np
from rl.launcher import launch
import os
import envs
def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="gbphrl", choices=["dhrl", "gbphrl", "value", "grid", "grid8", "grid_complex", "custom", "custom_complex"])
    parser.add_argument("--uncertainty", default="value", choices=['value', 'rnd'])
    parser.add_argument('--exploitation', default='both', choices=['high', 'both', 'rnd'])
    
    parser.add_argument('--env_name', type=str, default='AntMaze')
    parser.add_argument('--test_env_name', type=str, default='AntMaze')
    parser.add_argument('--setting', type=str, default='FIFG', choices=['FIFG', 'FIRG', 'RIRG', 'CB1', 'CB2', 'CB3', 'CRUCB', 'RUCB', 'SWUCB', 'CSWUCB', 'SWTS', 'CSWTS', 'TSWUCB', 'TCSWUCB', 'TSWTS', 'TCSWTS', 'optimal'])
    parser.add_argument('--action_max', type=float, default=30.) #network action_max > always 1
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--high_future_step', type=int, default=15)
    parser.add_argument('--subgoal_freq', type=int, default=40)
    parser.add_argument('--subgoal_scale', type=float, nargs='+', default=[12., 12.])
    parser.add_argument('--subgoal_offset', type=float, nargs='+', default=[8., 8.])
    parser.add_argument('--low_future_step', type=int, default=150)
    parser.add_argument('--subgoaltest_threshold', type=float, default=1)
    parser.add_argument('--no_her_high', action='store_true')
    parser.add_argument('--init_dist', type=float, default=4.)
    parser.add_argument('--waypoint_chase_step_threshold', type=float, default=100.)
    parser.add_argument('--waypoint_count_threshold', type=float, default=3.)
    
    parser.add_argument('--eval_coverage', action='store_true')
    parser.add_argument('--eval_coverage_num', type=int, default = 10)
    parser.add_argument('--eval_coverage_freq', type=int, default = 10)
    
    parser.add_argument('--eval_RIRG', action='store_true')
    parser.add_argument('--eval_RIRG_num', type=int, default = 10)
    parser.add_argument('--eval_RIRG_freq', type=int, default = 10)

    parser.add_argument('--subgoal_dim', type=int, default=2)
    parser.add_argument('--l_action_dim', type=int, default=8)
    parser.add_argument('--h_action_dim', type=int, default=2)
    parser.add_argument('--cutoff', type=float, default=30)
    parser.add_argument('--n_initial_rollouts', type=int, default=200) 

    parser.add_argument('--n_graph_node', type=int, default=300)
    parser.add_argument('--low_bound_epsilon', type=int, default=10)
    parser.add_argument('--gradual_pen', type=float, default= 5.0)
    parser.add_argument('--subgoal_noise_eps', type=float, default=2)

    ################################################################################################

    parser.add_argument('--low_future_p', type=float, default=0.8)
    parser.add_argument('--low_future_p_g', type=float, default=1.1)
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--clip_return', type=float, default=80) 
    parser.add_argument('--start_planning_epoch', type=int, default=5)
    parser.add_argument('--subgoaltest_p', type=float, default=0.2)

    parser.add_argument('--high_future_p', type=float, default=1.0)
    parser.add_argument('--high_penalty', type=float, default=1.0)
    parser.add_argument('--ada_score', action='store_true')
    parser.add_argument('--nosubgoal', action='store_true')
    
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)
    
    #score
    parser.add_argument('--ada_hindsight', action='store_true')
    parser.add_argument('--high_score', type=str, default='RND')
    parser.add_argument('--low_score', type=str, default='None')
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--high_score_input', type=str, default='goal')
    parser.add_argument('--low_input', type=str, default='goal')
    parser.add_argument('--input_normalization', action='store_true')
    parser.add_argument('--score_normalization', action='store_true')
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    # ablation
    parser.add_argument('--low_mc_dropout', action='store_true')

    #hindsight
    parser.add_argument('--high_hindsight', type=str, default='her')

    #cuda
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--cuda_num', type=int, default=0)

    #directory
    parser.add_argument('--save_dir', type=str, default='exp/')
    parser.add_argument('--ckpt_name', type=str, default='')
    parser.add_argument('--resume_ckpt', type=str, default='')
    parser.add_argument('--store_epoch', action='store_true')

    # Load model
    parser.add_argument('--load_ckpt_name', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--low_agent', action='store_true')
    parser.add_argument('--high_agent', action='store_true')
    parser.add_argument('--low_replay', action='store_true')
    parser.add_argument('--high_replay', action='store_true')
    parser.add_argument('--freeze', action='store_true')

    #network and training
    parser.add_argument('--use_reverse_dist_func', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_cycles', type=int, default=15)
    parser.add_argument('--high_optimize_freq', type=int, default=10)
    parser.add_argument('--densify_freq', type=int, default=1)

    parser.add_argument('--n_batches', type=int, default=1)
    parser.add_argument('--hid_size', type=int, default=256)
    parser.add_argument('--n_hids', type=int, default=3)
    parser.add_argument('--activ', type=str, default='relu')
    parser.add_argument('--noise_eps', type=float, default=0.1)
    
    
    parser.add_argument('--random_eps', type=float, default=0.2)
    parser.add_argument('--buffer_size', type=int, default=2500000)
    parser.add_argument('--gamma', type=float, default=0.99)
    
    parser.add_argument('--action_l2', type=float, default=0.01)
    parser.add_argument('--lr_actor', type=float, default=0.0001)
    parser.add_argument('--lr_critic', type=float, default=0.001)
    parser.add_argument('--polyak', type=float, default=0.995)

    parser.add_argument('--target_update_freq', type=int, default=10)
    parser.add_argument('--actor_update_freq', type=int, default=2)
    parser.add_argument('--grad_norm_clipping', type=float, default=-1.0)
    parser.add_argument('--grad_value_clipping', type=float, default=-1.0)
    #test
    parser.add_argument('--n_test_rollouts', type=int, default=10)
    parser.add_argument('--eval_render', type=bool, default=False)

    #graph construct
    parser.add_argument('--q_offset', action='store_true')
    parser.add_argument('--initial_sample', type=int, default=6000)
    parser.add_argument('--use_oracle_G', type=bool, default=False)
    parser.add_argument('--FGS', action="store_true")
    parser.add_argument('--AGS', action="store_true")
    parser.add_argument('--absolute_goal', action="store_true")

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inference', action='store_true')

    # Curriculum method
    # ['MEGA', 'RND']
    parser.add_argument('--curr_method', type=str, default=None)
    parser.add_argument('--RND_method', type=str, default='mixed')
    parser.add_argument('--RND_init_samples', type=int, default=512)
    parser.add_argument('--rnd_batch_size', type=int, default=128)

    # Wandb Visualization
    parser.add_argument('--train_visualization', action='store_true')
    parser.add_argument('--eval_visualization', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=5)

    parser.add_argument('--add_loss', action='store_true')
    parser.add_argument('--go_explore', action='store_true')
    parser.add_argument('--frontier_prob', type=float, default=0.3)
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    import os
    algo = launch(args)
    if args.inference:
        print('Inference Only')
        algo.inference()
    else:
        algo.run()