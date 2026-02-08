import numpy as np

from rl.curriculum.core import RandomNetworkDistillation
from sklearn.neighbors import KernelDensity

class Curriculum:
    def __init__(self, env_params, args, method, name='curriculum'):
        self.method = method
        self.args = args
        self.env_params = env_params
        goal_input_dim = env_params['goal']
        obs_input_dim = env_params['obs']
        self.epsilon = 0.2
        self.RND_goal = RandomNetworkDistillation(input_dim=goal_input_dim, output_dim=128, lr=1e-3, use_ag_as_input=True)
        self.RND_obs = RandomNetworkDistillation(input_dim=obs_input_dim, output_dim=128, lr=1e-3, use_ag_as_input=False)

    def generate_task(self, replay_buffer):
        if self.method == 'MEGA':
            kde = KernelDensity(kernel='gaussian', bandwidth = 0.1)
            replay_data = replay_buffer.sample_regular_batch(self.args.initial_sample)
            states = replay_data['ob']
            len_states = len(states)
            num_samples = min(1000, len_states)
            idx = np.random.randint(0, len_states, num_samples)
            kde_samples = states[idx]

            kde_sample_mean = np.mean(kde_samples, axis = 0, keepdims = True)

            kde_sample_std = np.std(kde_samples, axis=0, keepdims=True) + 1e-4

            kde_samples = (kde_samples - kde_sample_mean) / kde_sample_std

            fitted_kde = kde.fit(kde_samples)

            return states[np.argmin(fitted_kde.score_samples((states - kde_sample_mean)/kde_sample_std)),:self.args.subgoal_dim] + np.random.uniform(low=-3, high=3, size=self.args.subgoal_dim)
        
        else:
            batch = replay_buffer.sample(batch_size=self.args.RND_init_samples)
            if self.args.RND_method == 'goal':
                sampled_state = batch['ag']
                novelty_score = self.RND_goal.get_novelty(sampled_state)
            elif self.args.RND_method == 'obs':
                sampled_state = batch['o2']
                novelty_score = self.RND_obs.get_novelty(sampled_state)
            elif self.args.RND_method == 'mixed':
                p = np.random.random()
                if p >= self.epsilon:
                    sampled_state = batch['ag']
                    novelty_score = self.RND_goal.get_novelty(sampled_state)
                else:
                    sampled_state = batch['o2']
                    novelty_score = self.RND_obs.get_novelty(sampled_state)
            novel_idx = np.argmax(novelty_score.cpu())
            return sampled_state[novel_idx][:2]
    


    def train(self, replay_buffer, iterations, batch_size):
        if self.args.RND_method == 'goal':
            rnd_loss_goal = self.RND_goal.train(replay_buffer, iterations, batch_size)
        elif self.args.RND_method == 'obs':
            rnd_loss_obs = self.RND_obs.train(replay_buffer, iterations, batch_size)
        elif self.args.RND_method == 'mixed':
            rnd_loss_goal = self.RND_goal.train(replay_buffer, iterations, batch_size)
            rnd_loss_obs = self.RND_obs.train(replay_buffer, iterations, batch_size)