import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from envs import Environment
from algorithms.CRUCB import CRUCB
from algorithms.RUCB import RUCB
from algorithms.SW_CUCB import SW_CUCB
from algorithms.SW_CTS import SW_CTS
from algorithms.SW_UCB import SW_UCB
from algorithms.SW_TS import SW_TS
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.interpolate import make_interp_spline
import pandas as pd
import sys

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", nargs='+', type=str, default=['path', 'matching', 'spanning', 'max', 'toy'])
    parser.add_argument("--algos", nargs='+', type=str, default=['CRUCB','RUCB', 'SW_CTS', 'SW_TS', 'SW_UCB', 'SW_CUCB'])
    parser.add_argument("--easy", type=bool, default=False)
    parser.add_argument("--n_random", type=int, default=10)
    parser.add_argument("--T", type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    seed = args.seed
    
    np.random.seed(seed)
    easy = args.easy
    tasks = args.tasks
    algos = args.algos

    regret = {}
    plt.rcParams['font.size'] = 20
    for task in tasks:
        print(task)
        regret[task] = {}
        for algo in algos:
            regret[task][algo] = np.zeros((args.n_random, args.T), dtype=float)
        for i in range(args.n_random):
            print(i)
            env = Environment(task, args.T, seed + 100 * i, easy)
            optimal_cumulative_rewards = env.get_optimal_cumulative_rewards()
            np.savetxt(f'./results/optimal_cumulative_rewards_{task}_optimal_{easy}_b.txt', optimal_cumulative_rewards)
            
            for algo in algos:
                print(algo)
                
                env.reset()
                if '_' in algo:
                    line = algo.split('_')
                    epsilon = float(line[1])
                    print(epsilon)
                    algor = line[0]
                    runner = globals()[algor](args.T, env, epsilon)
                    # cumulative_rewards, _ = runner.run()
                    cumulative_rewards, selected_arms = runner.run()
                    
                    np.savetxt(f'./results/selected_arms_{task}_{algor}_{epsilon}_{seed+100*i}_{easy}_b.txt', selected_arms.astype(int))
                    np.savetxt(f'./results/cumulative_rewards_{task}_{algor}_{epsilon}_{seed+100*i}_{easy}_b.txt', cumulative_rewards)
                    regret[task][algo][i] += optimal_cumulative_rewards - cumulative_rewards
                
                    np.savetxt(f'./results/regrets_{task}_{algor}_{seed+100*i}_0.01_{epsilon}_{easy}_b.txt', regret[task][algo])
            
                else:
                    runner = globals()[algo](args.T, env, task)
                    cumulative_rewards, selected_arms = runner.run()
                    
                    np.savetxt(f'./results/selected_arms_{task}_{algo}_{seed+100*i}_{easy}.txt', selected_arms.astype(int))
                    np.savetxt(f'./results/cumulative_rewards_{task}_{algo}_{seed+100*i}_{easy}.txt', cumulative_rewards)
                    regret[task][algo][i] += optimal_cumulative_rewards - cumulative_rewards
                
                    np.savetxt(f'./results/regrets_{task}_{algo}_{seed+100*i}_0.01_{easy}.txt', regret[task][algo])                
                
        
        # fig, ax = plt.subplots(figsize = (10, 6))
        
        # for algo in algos:
        #     window_size = 100
        #     x = np.arange(0, args.T, 100)
        #     mean = np.mean(regret[task][algo], axis=0)
        #     mean = pd.Series(mean).rolling(window=window_size).mean()
        #     stds = np.std(regret[task][algo], axis=0)
        #     stds = pd.Series(stds).rolling(window=window_size).mean()
        #     ci95 = 1.96 * stds / np.sqrt(args.n_random)
        #     ci99 = 2.58 * stds / np.sqrt(args.n_random)
        #     mean = mean[x]
        #     ci99 = ci99[x]
        #     if algo == 'CRUCB':
        #         c = '#c80150'
        #         ax.plot(x, mean, color = c, label = f'{algo}')
        #     if algo == 'RUCB':
        #         c = '#6a1b9a'
        #         ax.plot(x, mean, color = c, label = f'{algo}')
        #     if algo == 'SW_CTS':
        #         c = '#00876c'
        #         ax.plot(x, mean, color = c, label = f'{algo}')
        #     if algo == 'SW_TS':
        #         c = '#66c2a5'
        #         ax.plot(x, mean, color = c, label = f'{algo}')
        #     if algo == 'SW_CUCB':
        #         c = '#0072b2'
        #         ax.plot(x, mean, color = c, label = f'{algo}')
        #     if algo == 'SW_UCB':
        #         c = '#00a0e3'
        #         ax.plot(x, mean, color = c, label = f'{algo}')
            
        #     ax.fill_between(x, mean-ci99, mean+ci99, color = c, alpha = 0.2)
        
        # ax.set_xlabel('episode number')
        # ax.set_ylabel('regret')
        # ax.legend(fontsize = 16)
        # plt.tight_layout()
        
        # plt.savefig(f'./results/plots/regret_{task}_{easy}_{seed}.png')