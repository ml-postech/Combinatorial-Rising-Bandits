import numpy as np
import sys
import torch
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from rl.algo.core import BaseAlgo
from rl.algo.graph import GraphPlanner

cmap = plt.cm.viridis

class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        test_env, test_env_coverage,
        low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
        low_reward_func, high_reward_func, high_score, low_score,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
            low_reward_func, high_reward_func, high_score, low_score,
            name=name,
        )
        self.test_env = test_env
        self.test_env_coverage = test_env_coverage
        self.fps_landmarks = None

        self.curr_subgoal = None
        self.curr_high_act = None
        self.curr_highpolicy_obs = None
        self.last_waypoint_obs = None

        self.way_to_subgoal = 0
        self.subgoal_freq = args.subgoal_freq
        self.subgoal_scale = np.array(args.subgoal_scale)
        self.subgoal_offset = np.array(args.subgoal_offset)
        self.subgoal_dim = args.subgoal_dim
        self.low_replay = low_replay
        self.high_replay = high_replay
        
        self.graphplanner = GraphPlanner(args, low_replay, low_agent, high_agent, high_score, env)
        self.waypoint_subgoal = None
        self.bef_waypoint_subgoal = None
        self.subgoal_list = []
        self.ag_list = []
        self.curriculum_goal = []
        self.train_visualization = False
        self.test_visualization = False
        if args.train_visualization:
            self.train_visualization = args.train_visualization
        if args.eval_visualization:
            self.eval_visualization = args.eval_visualization

    def get_actions(self, ob, ag, bg, a_max=1, random_goal=False, act_randomly=False, graph=False, first = False):
        #get subgoal
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)):
            self.curr_highpolicy_obs = ob
            
            if random_goal and not self.args.high_agent:
                sub_goal = np.random.uniform(low=-1, high=1, size=self.env_params['sub_goal'])
                sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
            elif self.args.nosubgoal:
                sub_goal = bg
            else:
                epsilon = np.random.uniform()
                sub_goal = self.high_agent.get_actions(ob, bg)
    
                if self.args.go_explore:
                    if self.graphplanner.graph is not None:
                        sampled_candidates = self.graphplanner.landmarks.copy()
                        batch_length = len(sampled_candidates)
                    else:
                        sampled_batch = self.low_replay.sample(batch_size=self.args.RND_init_samples)
                        batch_length = self.args.RND_init_samples
                        sampled_candidates = sampled_batch['ag'].copy()
                    batch_for_novelty = {}
                    batch_for_novelty['ob'] = np.multiply(np.ones((batch_length, 1)), ob)
                    batch_for_novelty['ag'] = np.multiply(np.ones((batch_length, 1)), ob[:self.args.subgoal_dim])
                    batch_for_novelty['bg'] = np.multiply(np.ones((batch_length, 1)), bg)
                    batch_for_novelty['a'] = sampled_candidates
                    novelty = self.high_score.get_batch_score(batch_for_novelty)
                    novel_idx = np.argmax(novelty)
                    self.monitor.store(novelty_mean=novelty.mean())
                    self.monitor.store(max_novelty=novelty[novel_idx])
                    sub_goal_novelty = batch_for_novelty['ag'][novel_idx]

                    ob_novelty_input = self.high_agent.to_tensor(np.array([ob[:self.args.subgoal_dim]]))
                    bg_novelty_input = self.high_agent.to_tensor(np.array([bg]))
                    sg_novelty_input = self.high_agent.to_tensor(np.array([sub_goal]))
                    novelty_input = torch.cat([ob_novelty_input, bg_novelty_input, sg_novelty_input], dim=-1)
                    subgoal_novelty_score = self.high_score.RND_score.get_novelty(novelty_input).detach().cpu().numpy()
                    novelty_q_value = self.high_agent.get_qs(ob, bg, np.array([sub_goal_novelty])).detach().cpu().numpy()
                    subgoal_q_value =self.high_agent.get_qs(ob, bg, np.array([sub_goal])).detach().cpu().numpy()

                    if epsilon < self.args.frontier_prob:
                        sub_goal = sub_goal_novelty
                    # print('Observation : ', ob[:self.args.subgoal_dim] , 'Desired Goal:', bg)
                    # print('Subgoal from high:', sub_goal, self.high_agent.get_qs(ob, bg, [sub_goal]))
                    # print('Subgoal from novelty:', sub_goal_novelty, self.high_agent.get_qs(ob, bg, [sub_goal_novelty]))
                    # if novelty_q_value + novelty[novel_idx] > subgoal_q_value + subgoal_novelty:
                    #     sub_goal = sub_goal_novelty

                    # if novelty_q_value > subgoal_q_value:
                    #     sub_goal = sub_goal_novelty

                if self.args.subgoal_noise_eps > 0.0:
                    subgoal_low_limit = self.subgoal_offset - self.subgoal_scale
                    subgoal_high_limit = self.subgoal_offset + self.subgoal_scale
                    sub_goal_noise = self.args.subgoal_noise_eps * np.random.randn(*sub_goal.shape)
                    sub_goal = sub_goal + sub_goal_noise
                    sub_goal = np.clip(sub_goal, subgoal_low_limit, subgoal_high_limit)

            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            self.subgoal_list.append(sub_goal)
            self.ag_list.append(ob[:self.args.subgoal_dim])

            #graph search
            if (self.graphplanner.graph is not None):
                new_sg = self.graphplanner.find_path(ob, self.curr_subgoal, ag, bg, train=True, first = first)
                if new_sg is not None:
                    self.curr_subgoal = new_sg
                # for waypoint in self.graphplanner.waypoint_vec:
                #     if waypoint in self.graphplanner.disconnected:
                #         self.graphplanner.densify()
                #         new_sg = self.graphplanner.find_path(ob, self.curr_subgoal, ag, bg, train=True, first = first)
                #         if new_sg is not None:
                #             self.curr_subgoal = new_sg

        # which waypoint to chase
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, ag, self.curr_subgoal, bg, train=True)[:self.subgoal_dim]
        if (self.bef_waypoint_subgoal != self.waypoint_subgoal).any():
            self.bef_waypoint_subgoal = self.waypoint_subgoal.copy()
            self.last_waypoint_obs = ob.copy()

        #find low level policy action
        if act_randomly and not self.args.low_agent:
            act = np.random.uniform(low=-a_max, high=a_max, size=self.env_params['l_action_dim'])
        else:
            clipped_waypoint_subgoal = ag + np.min([2, np.linalg.norm(self.waypoint_subgoal-ag)]) / (np.linalg.norm(self.waypoint_subgoal-ag)+1e-5) * (self.waypoint_subgoal-ag)
            act = self.low_agent.get_actions(ob, clipped_waypoint_subgoal)
            # act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
            if self.args.noise_eps > 0.0:
                act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
                act = np.clip(act, -a_max, a_max)
            if self.args.random_eps > 0.0:
                a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
                mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
                if self.num_envs > 1:
                    mask = np.expand_dims(mask, -1)
                act += mask * (a_rand - act)
        self.way_to_subgoal -= 1
        return act
    
    def low_agent_optimize(self):
        self.timer.start('low_train')
        for n_train in range(self.args.n_batches):
            batch = self.low_replay.sample(batch_size=self.args.batch_size)
            self.low_learner.update_critic(batch, train_embed=True)
            batch_g = self.low_replay.sample_g(batch_size=self.args.batch_size)
            self.low_learner.update_critic_g(batch_g, train_embed=True)
            if self.low_opt_steps % self.args.actor_update_freq == 0:
                self.low_learner.update_actor(batch, train_embed=True)
            self.low_opt_steps += 1
            if self.low_opt_steps % self.args.target_update_freq == 0:
                self.low_learner.target_update()
        
        self.timer.end('low_train')
        self.monitor.store(LowTimePerTrainIter=self.timer.get_time('low_train') / self.args.n_batches)


    def high_agent_optimize(self):
        self.timer.start('high_train')

        for n_train in range(self.args.n_batches):
            batch = self.high_replay.sample(batch_size=self.args.batch_size, graphplanner = self.graphplanner)
            self.high_learner.update_critic(batch, train_embed=True)
            if self.high_opt_steps % self.args.actor_update_freq == 0:
                self.high_learner.update_actor(batch, train_embed=True)
            self.high_opt_steps += 1
            if self.high_opt_steps % self.args.target_update_freq == 0:
                self.high_learner.target_update()
            if self.args.ada_hindsight or self.args.ada_score or self.args.go_explore or self.args.uncertainty == 'rnd':
                RND_batch = self.high_replay.sample_regular_batch(batch_size=self.args.rnd_batch_size)
                self.high_score.update_RND(RND_batch)
        
        self.timer.end('high_train')
        self.monitor.store(HighTimePerTrainIter=self.timer.get_time('high_train') / self.args.n_batches)


    def collect_experience(self, epoch, n_iter, random_goal= False, act_randomly=False, train_agent=True, graph=False):
        low_ob_list, low_ag_list, low_bg_list, low_a_list = [], [], [], []
        high_ob_list, high_ag_list, high_bg_list, high_a_list = [], [], [], []
        high_wp_list = []
        self.monitor.update_episode()
        observation = self.env.reset()
        first = True
        self.subgoal_list = []
        self.ag_list = []

        self.curr_subgoal = None
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']
        ag_origin = ag.copy()
        a_max = self.env_params['action_max']
        success_cnt = 0
        
        # if (self.graphplanner.graph is not None) and (self.args.FGS):
        #     FGS = self.graphplanner.check_easy_goal(ob, ag, bg)
        #     if FGS is not None:
        #         bg = FGS

        # Visualization Goal
        self.curriculum_goal.append(bg)

        # For visualization training
        bef_subgoal = self.curr_subgoal
        images = []
        graph_images = []
        graph_image = None
        edge_graph_image = None

        for timestep in range(self.env_params['max_timesteps']):
            act = self.get_actions(ob, ag, bg, a_max=a_max, random_goal= random_goal, act_randomly=act_randomly, graph=graph, first = first)
            first = False

            if train_agent:
                if self.train_visualization:
                    if epoch%10 == 0:
                        image = self.env.get_image(goal=bg, subgoal=self.curr_subgoal, waypoint=self.waypoint_subgoal)
                        images.append(image)
                        if self.graphplanner.graph is not None:
                            if (bef_subgoal != self.curr_subgoal).any():
                                bef_subgoal = self.curr_subgoal            

            low_ob_list.append(ob.copy())
            low_ag_list.append(ag.copy())
            low_bg_list.append(self.waypoint_subgoal.copy())
            low_a_list.append(act.copy())
            
            if ((self.way_to_subgoal == 0) or (timestep == self.env_params['max_timesteps'] - 1)):
                high_ob_list.append(self.curr_highpolicy_obs.copy())
                high_ag_list.append(self.curr_highpolicy_obs[:self.args.subgoal_dim].copy())
                high_bg_list.append(bg.copy())
                high_a_list.append(self.curr_subgoal.copy())
                high_wp_list.append(self.last_waypoint_obs.copy())
                self.last_waypoint_obs = None
                self.bef_waypoint_subgoal = None
            # Antmaze 0.5 / reacher 0 / pusher..?
            dist = 0.5
            if self.env.env_name == 'Reacher3D-v0':
                dist = 0.05
            if (np.linalg.norm(ob[:self.args.subgoal_dim]-bg[:self.args.subgoal_dim]) <= dist):
                if self.graphplanner.graph is not None:
                    if self.args.AGS:
                        new_goal = self.graphplanner.check_easy_goal(ob, ag, bg)
                        if new_goal is not None:
                            bg = new_goal
            observation, _, done, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            self.env_steps += 1
            for every_env_step in range(self.num_envs):
                if train_agent:
                    if not self.args.freeze:
                        if self.low_replay.n_transitions_stored > self.args.batch_size:
                            self.low_agent_optimize()
                    # if self.env_steps % self.args.high_optimize_freq == 0:
                    #     self.high_agent_optimize()
            
            self.total_timesteps += self.num_envs
            
        if train_agent:
            subgoal_image = self.draw_subgoal(bg)
            self.monitor.store_video(images, subgoal=subgoal_image, eval=False)
            
        # if self.train_visualization:
        #     if epoch % 10 == 0:
        #         image = self.env.get_image(subgoal=self.curr_subgoal, waypoint=self.waypoint_subgoal)
        #         images.append(image)
        #         self.monitor.store_video(images, graph_images, graph_image, edge_graph_image, eval=False)
            
        #print(bg, ag)
        low_ob_list.append(ob.copy())
        low_ag_list.append(ag.copy())
        high_ob_list.append(ob.copy())
        high_ag_list.append(ag.copy())

        low_experience = dict(ob=low_ob_list, ag=low_ag_list, bg=low_bg_list, a=low_a_list)
        #high_experience = dict(ob=high_ob_list, ag=high_ag_list, bg=high_bg_list, a=high_a_list)
        high_experience = dict(ob=high_ob_list, ag=high_ag_list, bg=high_bg_list, a=high_a_list, wp=high_wp_list)
        low_experience = {k: np.array(v) for k, v in low_experience.items()}
        high_experience = {k: np.array(v) for k, v in high_experience.items()}
        if low_experience['ob'].ndim == 2:
            low_experience = {k: np.expand_dims(v, 0) for k, v in low_experience.items()}
        else:
            low_experience = {k: np.swapaxes(v, 0, 1) for k, v in low_experience.items()}
        if high_experience['ob'].ndim == 2:
            high_experience = {k: np.expand_dims(v, 0) for k, v in high_experience.items()}
        else:
            high_experience = {k: np.swapaxes(v, 0, 1) for k, v in high_experience.items()}
        low_reward = self.low_reward_func(ag, self.waypoint_subgoal.copy(), None)
        high_reward = self.high_reward_func(ag, bg, None, ob)

        total_success_count = 0
        Train_Dist = self.env.goal_distance(ag, bg)
        if(self.args.env_name == "AntMazeSmall-v0"):
            if (Train_Dist <= 2.5):
                total_success_count = 1
        elif(self.args.env_name == "Reacher3D-v0"):
            if (Train_Dist <= 0.25):
                total_success_count = 1
        else:
            if (Train_Dist <= 5):
                total_success_count = 1

        self.monitor.store(LowReward=np.mean(low_reward))
        self.monitor.store(HighReward=np.mean(high_reward))
        self.monitor.store(Train_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
        self.monitor.store(Train_Dist=Train_Dist)
        self.low_replay.store(low_experience)
        self.high_replay.store(high_experience)

        return total_success_count
    

    def run(self):
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            # if n_init_rollout % 50 == 0:
            #     results = self.run_eval_coverage(0, use_test_env_coverage=True, render=self.args.eval_render)
            #     print('Coverage')
            #     print(np.mean(results[:, 2]))
            self.collect_experience(epoch = -1, n_iter = -1, random_goal= True, act_randomly=True, train_agent=False, graph=False)
            
        # results = self.run_eval_coverage(0, use_test_env_coverage=True, render=self.args.eval_render)
        # print('Coverage')
        # print(np.mean(results[:, 2]))
        
        # input()
        # self.graphplanner.graph_construct(0)
        # self.run_eval(0, use_test_env=True, render=self.args.eval_render) 
        # print('a')
        # input()
        
        for epoch in range(self.args.n_epochs):
            self.curriculum_goal = []
            total_success_count = 0
            print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
            sys.stdout.flush()
 
            if epoch >= self.args.start_planning_epoch :
                #goal_scheduling = True
                self.graphplanner.graph_construct(epoch)
                self.monitor.store(epoch=epoch)
            
            # if self.graphplanner.graph is not None:
            #     if epoch % self.args.densify_freq == 0:
            #         self.graphplanner.densify()
            
            for n_iter in range(self.args.n_cycles):
                print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                sys.stdout.flush()
                self.timer.start('rollout')

                #self.collect_experience(train_agent=True, graph=True)
                success_count = self.collect_experience(epoch, n_iter, train_agent=True, graph=True)
                total_success_count += success_count

                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout'))
                
            if epoch >= self.args.start_planning_epoch :
                self.graphplanner.graph_construct(epoch)
            
            self.monitor.store(Train_Success_Rate=total_success_count/self.args.n_cycles)
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(low_opt_steps=self.low_opt_steps)
            self.monitor.store(high_opt_steps=self.high_opt_steps)
            self.monitor.store(low_replay_size=self.low_replay.current_size)
            self.monitor.store(high_replay_size=self.high_replay.current_size)
            self.monitor.store(low_replay_fill_ratio=float(self.low_replay.current_size / self.low_replay.size))
            self.monitor.store(high_replay_fill_ratio=float(self.high_replay.current_size / self.high_replay.size))
            

            # her_success = self.run_eval(epoch, use_test_env=True, render=self.args.eval_render)   
            # print('Epoch %d her eval %.3f'%(epoch, her_success))
            # print('Log Path:', self.log_path)
            # logger.record_tabular("Epoch", epoch)
            # self.monitor.store(Success_Rate=her_success)
            # if self.args.store_epoch:
            #     if epoch > 0 and epoch % 10 == 0:
            #         self.save_all(self.model_path, epoch)
            if epoch %1000 == 0:
                np.savetxt(f'selected_superarm_{self.env.env_name}_{self.args.setting}_{self.args.seed}_{epoch}.txt', self.graphplanner.selected_arm)
                np.savetxt(f'previous_timesteps_{self.env.env_name}_{self.args.setting}_{self.args.seed}_{epoch}.txt', self.graphplanner.previous_timesteps)
                
        np.savetxt(f'selected_superarm_{self.env.env_name}_{self.args.setting}_{self.args.seed}_{epoch}.txt', self.graphplanner.selected_arm)
        np.savetxt(f'previous_timesteps_{self.env.env_name}_{self.args.setting}_{self.args.seed}_{epoch}.txt', self.graphplanner.previous_timesteps)
        
        # if not self.args.store_epoch:
        #     self.save_all(self.model_path)

    def run_eval_coverage(self, epoch, use_test_env_coverage=False, render=False):
        high_ob_list, high_ag_list, high_bg_list, high_a_list = [], [], [], []
        high_wp_list = []
        batch= {}
        env = self.env
        if use_test_env_coverage and hasattr(self, 'test_env'):
            print("use test env coverage")
            env = self.test_env_coverage
        total_success_count = 0
        total_trial_count = 0
        
        maps = []
        success_rates = []
        
        if(self.args.env_name == 'AntMazeW'):
            maps = [[ 0,  0], [ 0, -8], [ 8, -8], [16, -8], [24, -8],
                    [32, -8], [32,  0], [32,  8], [24,  8], [16,  8],
                    [32, 16], [32, 24], [24, 24], [16, 24], [ 8, 24],
                    [ 0, 24], [ 0, 16]]
        elif(self.args.env_name == 'AntMazeBottleneck-v0'):
            maps = [[ 0,  0], [ 8,  0], [16,  0], [16,  8], [16, 16],
                    [ 8, 16], [ 0, 16]]  
        elif(self.args.env_name == 'AntMaze'):
            maps = [[ 0,  0], [ 8,  0], [16,  0], [16,  8], [16, 16],
                    [ 8, 16], [ 0, 16]]
        elif(self.args.env_name == 'AntMazeP'):
            maps = [[ 0,  0], [-8,  0], [-8,  8], [-8, 16], [ 0, 16],
                    [ 0, 24], [ 0, 32], [-8, 32], [ 8, 32], [16, 32],
                    [24, 32], [16, 24], [16, 16], [24, 16], [24,  8],
                    [24,  0], [16,  0]]
        elif(self.args.env_name == 'AntMazeS'):
            maps = [[ 0,  0], [ 8,  0], [16,  0], [24,  0], [32,  0],
                    [32,  8], [32, 16], [24, 16], [16, 16], [ 8, 16],
                    [ 0, 16], [ 0, 24], [ 0, 32], [ 8, 32], [16, 32],
                    [24, 32], [32, 32]]
        elif(self.args.env_name == 'AntMazeComplex-v0'):
            maps = [[ 0,  0], [ 8,  0], [16,  0], [16,  8], [16, 16],
                    [ 8, 16], [ 0, 16], [ 0, 24], [ 0, 32], [ 0, 40],
                    [ 0, 48], [ 8, 40], [16, 32], [16, 40], [16, 48],
                    [24, 40], [32, 40], [32, 48], [40, 48], [48, 48],
                    [48, 40], [32, 32], [32, 24], [40, 24], [40, 16],
                    [32,  8], [40,  8], [48,  8], [32,  0], [40,  0],
                    [48,  0]]
            
        
        for [x, y] in maps:
            total_success_count = 0
            total_trial_count = 0
            
            for _ in range(self.args.eval_coverage_num):
                self.curr_subgoal = None
                if(self.args.env_name == 'AntMazeBottleneck-v0'):
                    observation = env.reset()
                    observation = env.change_goal(x = x, y = y, size = 4)
                    ob = observation['observation']
                    bg = observation['desired_goal']
                    ag = observation['achieved_goal']
                else:
                    observation = env.reset(xg = x, yg = y, size = 4)
                    ob = observation['observation']
                    bg = observation['desired_goal']
                    ag = observation['achieved_goal']
                first = True
                for _ in range(self.env_params['max_timesteps']):
                    act = self.eval_get_actions(ob, ag, bg, first = first)                 
                    first = False
                    observation, _, done, info = env.step(act)
                    ob = observation['observation']
                    ag = observation['achieved_goal']
                    
                TestEvn_Dist = env.goal_distance(ag, bg)
                
                total_trial_count += 1
                if(self.args.env_name == "AntMazeSmall-v0"):
                    if (TestEvn_Dist <= 2.5):
                        total_success_count += 1
                elif(self.args.env_name == "AntMazeMultiPath-v0"):
                    if (TestEvn_Dist <= 2.5):
                        total_success_count += 1
                elif(self.args.env_name == "Reacher3D-v0"):
                    if (TestEvn_Dist <= 0.25):
                        total_success_count += 1
                else:
                    if (TestEvn_Dist <= 5):
                        total_success_count += 1
            
            success_rate = total_success_count / total_trial_count
            success_rates.append(success_rate)    
            
        maps = np.array(maps)
        success_rates = np.array(success_rates)
        success_rates = np.expand_dims(success_rates, axis = -1)
        results = np.concatenate((maps, success_rates), axis = 1)
                
        return results

    def run_eval_RIRG(self, epoch, use_test_env_coverage=False, render=False):
        high_ob_list, high_ag_list, high_bg_list, high_a_list = [], [], [], []
        high_wp_list = []
        batch= {}
        env = self.env
        if use_test_env_coverage and hasattr(self, 'test_env'):
            print("use test env coverage")
            env = self.test_env_coverage
        total_success_count = 0
        total_trial_count = 0
        
        inits = []
        goals = []
        scenario_num = 0
        success_rates = []
        
        if(self.args.env_name == 'AntMazeW'):
            scenario_num = 6
            inits = [[0, 0], [0, 0], [16, 8], [16, 8], [0, 16], [0, 16]]
            goals = [[16, 8], [0, 16], [0, 0], [0, 16], [0, 0], [16, 8]]
        elif(self.args.env_name == 'AntMazeP'):
            scenario_num = 12
            inits = [[0, 0], [0, 0], [0, 0], [-8, 32], [-8, 32], [-8, 32], [24, 32], [24, 32], [24, 32], [16, 0], [16, 0], [16, 0]]
            goals = [[-8, 32], [24, 32], [16, 0], [0, 0], [24, 32], [16, 0], [0, 0], [-8, 32], [16, 0], [0, 0], [-8, 32], [24, 32]]
            
        for i in range(scenario_num):
            total_success_count = 0
            total_trial_count = 0
            for _ in range(self.args.eval_RIRG_num):
                self.curr_subgoal = None
                observation = env.reset(xi = inits[i][0], yi = inits[i][1], init = True, xg = goals[i][0], yg = goals[i][1], size = 0)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
                first = True
                for _ in range(self.env_params['max_timesteps']):
                    act = self.eval_get_actions(ob, ag, bg, first = first)                 
                    first = False
                    observation, _, done, info = env.step(act)
                    ob = observation['observation']
                    ag = observation['achieved_goal']
                    
                TestEvn_Dist = env.goal_distance(ag, bg)
                
                total_trial_count += 1
                if(self.args.env_name == "AntMazeSmall-v0"):
                    if (TestEvn_Dist <= 2.5):
                        total_success_count += 1
                elif(self.args.env_name == "AntMazeMultiPath-v0"):
                    if (TestEvn_Dist <= 2.5):
                        total_success_count += 1
                elif(self.args.env_name == "Reacher3D-v0"):
                    if (TestEvn_Dist <= 0.25):
                        total_success_count += 1
                else:
                    if (TestEvn_Dist <= 5):
                        total_success_count += 1
            
            success_rate = total_success_count / total_trial_count
            success_rates.append(success_rate)    
            
        success_rates = np.array(success_rates)
                
        return success_rates

    def run_eval(self, epoch, use_test_env=False, render=False):
        high_ob_list, high_ag_list, high_bg_list, high_a_list = [], [], [], []
        high_wp_list = []
        batch= {}
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            print("use test env")
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        images = []
        graph_images = []
        graph_image = None
        edge_graph_image = None
        curriculum_goal = None
        high_dist_from_point = None
        low_dist_from_point = None
        high_images = []
        low_images = []
        high_images2 = []
        low_images2 = []
        value_images = []
        high_uncertainty_images = []
        low_uncertainty_images = []
        success_timestep = 0
        success_first = False
        for n_test in range(self.args.n_test_rollouts):
            success_timestep = self.env_params['max_timesteps']
            success_first = False
            success_cnt = 0
            self.curr_subgoal = None
            bef_subgoal = self.curr_subgoal
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            # if n_test == 0:
            #     high_dist_from_point = self.plot_dist_from_point(ob=ob, bg=bg, sg=None, agent=self.high_agent)
            #     low_dist_from_point = self.plot_dist_from_point(ob=ob, bg=bg, sg=None, agent=self.low_agent)
            first = True
            for timestep in range(self.env_params['max_timesteps']):
                act = self.eval_get_actions(ob, ag, bg, first = first)
                first = False
                if n_test==0:
                    if epoch % self.args.eval_interval == 0 and self.args.eval_visualization:
                        if self.env.env_name == 'Reacher3D-v0':
                            image = env.get_image()
                        else:
                            image = env.get_image(goal=bg, subgoal=self.curr_subgoal, waypoint=self.waypoint_subgoal)
                        images.append(image)
                    if self.graphplanner.graph is not None:
                        # if (bef_subgoal != self.curr_subgoal).any() or self.way_to_subgoal == self.subgoal_freq-1:
                        if timestep % 60 == 0:
                            # if self.env.env_name != 'Reacher3D-v0':
                            graph_image, _ = self.graphplanner.draw_graph(start=ag, subgoal=self.curr_subgoal, goal=bg)
                            # high_dist_from_point = self.plot_dist_from_point(ob=ob, bg=bg, sg=self.curr_subgoal, agent=self.high_agent)
                            # low_dist_from_point = self.plot_dist_from_point(ob=ob, bg=bg, sg=None, agent=self.low_agent, nog = True)
                            # high_dist_to_point = self.plot_dist_to_point(ob=None, bg=bg, sg=None, agent=self.high_agent)
                            # low_dist_to_point = self.plot_dist_to_point(ob=None, bg=bg, sg=None, agent=self.low_agent, nog = True)
                            # value_image = self.plot_value(ob=ob, bg=bg, sg=self.curr_subgoal, agent = self.high_agent)
                            # high_uncertainty_image = self.plot_uncertainty(ob=ob, bg=None, sg=None, agent = self.high_agent)
                            # low_uncertainty_image = self.plot_uncertainty(ob=ob, bg=None, sg=None, agent = self.low_agent)
                            # value_images.append(value_image)
                            # high_images.append(high_dist_from_point)
                            # low_images.append(low_dist_from_point)
                            # high_images2.append(high_dist_to_point)
                            # low_images2.append(low_dist_to_point)
                            graph_images.append(graph_image)
                            # high_uncertainty_images.append(high_uncertainty_image)
                            # low_uncertainty_images.append(low_uncertainty_image)
                            bef_subgoal = self.curr_subgoal
                
                if self.args.ada_hindsight:
                    if ((self.way_to_subgoal == 0) or (timestep == self.env_params['max_timesteps'] - 1)):
                        high_ob_list.append(self.curr_highpolicy_obs.copy())
                        high_ag_list.append(self.curr_highpolicy_obs[:self.args.subgoal_dim].copy())
                        high_bg_list.append(bg.copy())
                        high_a_list.append(self.curr_subgoal.copy())
                        high_wp_list.append(self.last_waypoint_obs.copy())
                        self.last_waypoint_obs = None
                        self.bef_waypoint_subgoal = None                    

                observation, _, done, info = env.step(act)
                ob = observation['observation']
                ag = observation['achieved_goal']
                if info['is_success']:
                    success_cnt += 1
                    if success_first == False:
                        success_first = True
                        success_timestep = timestep
                if success_cnt > self.args.subgoal_freq:
                    break
            
            if n_test == 0:
                image_coverage = None
                if epoch % self.args.eval_interval == 0 and self.args.eval_visualization:
                    image = env.get_image(subgoal=self.curr_subgoal, waypoint=self.waypoint_subgoal)
                    images.append(image)
                if self.graphplanner.graph is not None:
                    # if self.env.env_name != 'Reacher3D-v0':
                    _ , graph_image = self.graphplanner.draw_graph()
                    #edge_graph_image = self.graphplanner.draw_edge_graph()
                    curriculum_goal = self.draw_curriculum_goal()
                    # high_dist_from_point = self.plot_dist_from_point(ob=ob, bg=None, agent=self.high_agent)
                    # low_dist_from_point = self.plot_dist_from_point(ob=ob, bg=None, agent=self.low_agent)
                if self.args.eval_coverage:
                    if epoch % self.args.eval_coverage_freq == 0:
                        if self.graphplanner.graph is not None:
                            results = self.run_eval_coverage(epoch, use_test_env_coverage=True, render=self.args.eval_render)
                            print(epoch)
                            print('Coverage')
                            print(np.mean(results[:, 2]))
                            image_coverage = self.plot_coverage(epoch, results)
                            self.monitor.store(eval_coverage=np.mean(results[:, 2]))
                if self.args.eval_RIRG:
                    if epoch % self.args.eval_RIRG_freq == 0:
                        if self.graphplanner.graph is not None:
                            results = self.run_eval_RIRG(epoch, use_test_env_coverage=True, render=self.args.eval_render)
                            print('RIRG')
                            print(np.mean(results))
                            self.monitor.store(eval_RIRG=np.mean(results))
                self.monitor.store_video(images, graph_images, graph_image, edge_graph_image, curriculum_goal, eval_coverage_images = image_coverage, high_image=high_images, low_image=low_images, high_image2=high_images2, low_image2=low_images2, value_images=value_images, high_uncertainty_images=high_uncertainty_images, low_uncertainty_images=low_uncertainty_images)

            if self.args.ada_hindsight:
                batch['ob'] = high_ob_list
                batch['bg'] = high_bg_list
                batch['a'] = high_a_list
                batch['ag'] = high_ag_list
                batch['wp']= high_wp_list
                batch = {k: np.array(v) for k, v in batch.items()}
                # novelty_score = self.high_score.get_batch_score(batch)
                # uncertainty_score = self.low_score.get_batch_score(batch)
                # self.monitor.store(eval_rnd_mean_score=np.mean(novelty_score))
                # self.monitor.store(eval_uncertainty_score=np.mean(uncertainty_score))

            TestEvn_Dist = env.goal_distance(ag, bg)
            self.monitor.store(TestEvn_Dist=np.mean(TestEvn_Dist))
            self.monitor.store(success_timestep=success_timestep)
            total_trial_count += 1
            if(self.args.env_name == "AntMazeSmall-v0"):
                if (TestEvn_Dist <= 2.5):
                    total_success_count += 1
            elif(self.args.env_name == "AntMazeMultiPath-v0"):
                if (TestEvn_Dist <= 2.5):
                    total_success_count += 1
            elif(self.args.env_name == "Reacher3D-v0"):
                if (TestEvn_Dist <= 0.25):
                    total_success_count += 1
            elif(self.args.env_name == "AntMazeCB"):
                if (TestEvn_Dist <= 0.5):
                    total_success_count += 1
            elif(self.args.env_name == "AntMazeCBcomplex-v0"):
                if (TestEvn_Dist <= 0.5):
                    total_success_count += 1
            else:
                if (TestEvn_Dist <= 5):
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        return success_rate

    def eval_get_actions(self, ob, ag, bg, a_max=1, random_goal=False, act_randomly=False, graph=False, first = False):
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)):
            self.curr_highpolicy_obs = ob
            if self.args.nosubgoal:
                sub_goal = bg
            else:
                sub_goal = self.high_agent.get_actions(ob, bg)
            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            if (self.graphplanner.graph is not None):
                new_sg = self.graphplanner.find_path(ob, self.curr_subgoal, ag, bg, train = False, first = first)
                if new_sg is not None:
                    self.curr_subgoal = new_sg

        # which waypoint to chase
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, ag, self.curr_subgoal, bg)
        if (self.bef_waypoint_subgoal != self.waypoint_subgoal).any():
            self.bef_waypoint_subgoal = self.waypoint_subgoal.copy()
            self.last_waypoint_obs = ob.copy()
            
        clipped_waypoint_subgoal = ag + np.min([2, np.linalg.norm(self.waypoint_subgoal-ag)]) / np.linalg.norm(self.waypoint_subgoal-ag) * (self.waypoint_subgoal-ag)
        act = self.low_agent.get_actions(ob, clipped_waypoint_subgoal)
        # act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
        self.way_to_subgoal -= 1 
        return act

    def inference(self):
        use_test_env = True
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            print("use test env")
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        images = []
        graph_images = []
        graph_image = None
        edge_graph_image = None
        curriculum_goal = None
        success_cnt = 0
        self.curr_subgoal = None
        bef_subgoal = self.curr_subgoal
        while True:
            print('------------------------------------------------------------')
            images=[]
            success_cnt = 0
            # init = np.array([0., 0.])
            # goal = np.array([4., 0.])
            init_x = float(input('Initial x point:'))
            init_y = float(input('Initial y point:'))
            goal_x = float(input('Goal x point:'))
            goal_y = float(input('Goal y point:'))
            init = np.array([init_x, init_y])
            goal = np.array([goal_x, goal_y])
            observation = env.reset(inference=True, init_position=init, goal_position=goal)
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            for timestep in range(self.env_params['max_timesteps']):
                cur_timestep = timestep
                with torch.no_grad():
                    if timestep % self.args.subgoal_freq == 0:
                        print('Current state: ', ag)
                        sg_method = input("Subgoal Method: ")
                        if sg_method == 'net':
                            sg = self.high_agent.get_actions(ob, bg)
                        elif sg_method == 'end':
                            break
                        else:
                            subgoal_x = float(input('Subgoal x point:'))
                            subgoal_y = float(input('Subgoal y point:'))
                            sg = np.array([subgoal_x, subgoal_y])
                        dist = self.low_agent._get_point_to_point(ob, sg)
                        a = self.low_agent.get_pis(ob, sg)
                        a2 = torch.unsqueeze(self.low_agent.to_tensor(self.low_agent.get_actions(ob, sg)), 0)
                        q_low = self.low_agent.get_qs(ob, sg, a)
                        q_low_g = self.low_agent.get_qs_g(ob, sg, a)
                        q_high = self.high_agent.get_qs(ob, bg, [sg])
                        val_mean, val_std, act_mean, act_std = self.get_distribution(ob, sg)
                        print('Value Mean: ', val_mean , ' Value Std: ', val_std)
                        print('Action Mean:', act_mean, 'Action Std:', act_std)
                        print('Subgoal: ', sg)
                        print('High_value:', q_high)
                        print('Low_value:', q_low)
                        print('Low_value_graph', q_low_g)
                        print('Dist: ', dist)
                    act = self.low_agent.get_actions(ob, sg)
                image = env.get_image(goal=bg, subgoal=sg, waypoint=self.waypoint_subgoal)
                images.append(image)
                if self.graphplanner.graph is not None:
                    if (bef_subgoal != self.curr_subgoal).any():
                        graph_image, _ = self.graphplanner.draw_graph(start=ag, subgoal=self.curr_subgoal, goal=bg)
                        graph_images.append(graph_image)
                        bef_subgoal = self.curr_subgoal
                observation, _, done, info = env.step(act)
                ob = observation['observation']
                ag = observation['achieved_goal']
                TestEvn_Dist = env.goal_distance(ag, bg)
                #print(TestEvn_Dist)
                if TestEvn_Dist <= 0.3:
                    success_cnt += 1
                # if info['is_success']:
                #     print('First Reach:', timestep)
                # elif done:
                #     break
                if success_cnt > 10:
                    break
            
            #print('First reach at', timestep-10)
            image = env.get_image(subgoal=self.curr_subgoal, waypoint=self.waypoint_subgoal)
            images.append(image)
            if self.graphplanner.graph is not None:
                _ , graph_image = self.graphplanner.draw_graph()
                edge_graph_image = self.graphplanner.draw_edge_graph()
                curriculum_goal = self.draw_curriculum_goal()
            self.monitor.store_video(images, graph_images, graph_image, edge_graph_image, curriculum_goal)

            TestEvn_Dist = env.goal_distance(ag, bg)
            self.monitor.store(TestEvn_Dist=np.mean(TestEvn_Dist))
            
            total_trial_count += 1
            if(self.args.env_name == "AntMazeSmall-v0"):
                if (TestEvn_Dist <= 2.5):
                    total_success_count += 1
            elif(self.args.env_name == "Reacher3D-v0"):
                if (TestEvn_Dist <= 0.25):
                    total_success_count += 1
            else:
                if (TestEvn_Dist <= 5):
                    total_success_count += 1
            end = input('End?')
            if end =="y":
                break
        success_rate = total_success_count / total_trial_count
        return success_rate

    def inference_get_actions(self, ob, bg, a_max=1, random_goal=False, act_randomly=False, graph=False, goal_scheduling=False,  high_level=True):
        act = self.low_agent.get_actions(ob, bg)
        return act

    def add_noise_to_state_action(self, state, goal, noise_std):
        noisy_state = state
        noise_goal = goal
        noise = np.random.normal(0, noise_std, self.args.subgoal_dim)
        #noisy_state[:self.args.subgoal_dim] += noise
        noise_goal += noise
        noisy_action = self.low_agent.get_pis(noisy_state, noise_goal)
        return noisy_state, noisy_action, noise_goal

    def get_distribution(self, state, goal, noise_std=0.01, n_samples=100):
        value_samples = []
        action_samples = []
        for i in range(n_samples):
            noisy_state, noisy_action, noise_goal = self.add_noise_to_state_action(state, goal, noise_std)
            action_samples.append(noisy_action.detach().cpu().numpy()[0])
            value_samples.append(self.low_agent.get_qs(noisy_state, noise_goal, noisy_action).cpu())
        #value_samples = np.clip(value_samples, clip_range[0], clip_range[1])
        value_mean = np.mean(value_samples)
        value_std = np.std(value_samples)
        action_mean = np.mean(action_samples, axis=0)
        action_std = np.std(action_samples, axis=0)
        return value_mean, value_std, action_mean, action_std
    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']

    def draw_curriculum_goal(self):
        map_size = [-4, 20]
        if self.env.env_name == 'AntMaze':
            # -4~ 20
            map_size = [-4, 20]
            Map_x, Map_y = (24, 24)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeBottleneck':
            map_size = [-4, 20]
            Map_x, Map_y = (24, 24)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeMultiPathBottleneck':
            map_size = [-4, 20]
        elif self.env.env_name == 'AntMazeSmall-v0':
            # -2 ~ 12
            map_size = [-2, 12]
            Map_x, Map_y = (12, 12)
            start_x, start_y = (2,2)
        elif self.env.env_name == 'AntMazeS':
            map_size = [-4, 36]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeW':
            map_size_x = [-4, 36]
            map_size_y = [-12, 28]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 12)
        elif self.env.env_name == 'AntMazeP':
            map_size_x = [-12, 28]
            map_size_y = [-4, 36]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (12, 4)
        elif self.env.env_name == 'AntMazeComplex-v0':
            # -4 ~ 52
            map_size = [-4, 52]
            Map_x, Map_y = (56, 56)
            start_x, start_y = (4, 4)

        fig3, ax3 = plt.subplots()
        if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW':
            ax3.set_xlim(map_size_x)
            ax3.set_ylim(map_size_y)
        else:
            ax3.set_xlim(map_size)
            ax3.set_ylim(map_size)
        x_vertex = []
        y_vertex = []
        for number, curr_goal in enumerate(self.curriculum_goal):
            #print(curr_goal)
            x_vertex.append(curr_goal[0])
            y_vertex.append(curr_goal[1])
            ax3.annotate(str(number), (curr_goal[0], curr_goal[1]))

        ax3.scatter(x_vertex, y_vertex, c='r', marker='o', alpha=1)
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png')
        buf3.seek(0)
        image1 = Image.open(buf3)
        curr_goal_array = np.array(image1)
        plt.close()
        return curr_goal_array

    def plot_coverage(self, epoch, results):
        map_size = [-4, 20]
        if self.env.env_name == 'AntMazeW':
            map_size_x = [-6, 38]
            map_size_y = [-14, 30]
            wall_x = [ -4,  36, 36, -4, -4,  4,  4, 28, 28, 12, 12, 28, 28,  4,  4, -4, -4]
            wall_y = [-12, -12, 28, 28, 12, 12, 20, 20, 12, 12,  4,  4, -4, -4,  4,  4, -12]
            size = 4
        elif self.env.env_name == 'AntMazeBottleneck':
            map_size = [-4, 20]
            wall_x = [-4, 20, 20, 17, 17, 20, 20, -4, -4, 12, 12, 15, 15, 12, 12, -4, -4]
            wall_y = [-4, -4,  7,  7,  9,  9, 20, 20, 12, 12,  9,    9,    7,  7,  4,  4, -4]
            size = 4
        elif self.env.env_name == 'AntMaze':
            map_size = [-4, 20]
            wall_x = [-4, 20, 20, -4, -4, 12, 12, -4, -4]
            wall_y = [-4, -4, 20, 20, 12, 12,  4,  4, -4]
            size = 4
        elif self.env.env_name == 'AntMazeS':
            map_size = [-4, 36]
            wall_x = [-4, -4, 28, 28,  0,  0, 36, 36,  4,  4, 36, 36, -4]
            wall_y = [-4,  4,  4, 12, 12, 36, 36, 28, 28, 20, 20, -4, -4] 
            size = 4
        elif self.env.env_name == 'AntMazeP':
            map_size_x = [-14, 30]
            map_size_y = [-6, 38]
            wall_x = [-12,  4,  4, -4, -4,  4,  4, 12, 12, 20, 20, 12, 12, 28, 28, 20, 20, 28, 28, -12, -12, -4, -4, -12, -12]
            wall_y = [ -4, -4,  4,  4, 12, 12, 28, 28, 12, 12,  4,  4, -4, -4, 20, 20, 28, 28, 36,  36,  28, 28, 20,  20,  -4]
            size = 4
        elif self.env.env_name == 'AntMazeComplex-v0':
            # -4 ~ 52
            map_size = [-4, 52]
            wall_x = [-4, -4, 12, 12, -4, -4,  4,  4, 12, 12, 20, 20, 28, 28, 52, 52, 44, 44, 36, 36, 44, 44, 52, 52, 28, 28, 36, 36, 28, 28, 20, 20, 12, 12,  4,  4, 20, 20, -4]
            wall_y = [-4,  4,  4, 12, 12, 52, 52, 44, 44, 52, 52, 44, 44, 52, 52, 36, 36, 44, 44, 28, 28, 12, 12, -4, -4, 12, 12, 20, 20, 36, 36, 28, 28, 36, 36, 20, 20, -4, -4]
            size = 4    
        figc, axc = plt.subplots()
        if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW':
            axc.set_xlim(map_size_x)
            axc.set_ylim(map_size_y)
        else:
            axc.set_xlim(map_size)
            axc.set_ylim(map_size)
            
        axc.plot(wall_x, wall_y, c ='k')

        for [x, y, s] in results:
            axc.add_patch(
                patches.Rectangle(
                    (x-size, y-size),
                    2*size, 2*size,
                    ec = None,
                    fc = cmap(s)
            ))
            plt.text(x, y, s, ha = 'center', va = 'center')
        #plt.savefig(f'test_{self.args.setting}_{epoch}.png')
        bufc = io.BytesIO()
        figc.savefig(bufc, format='png')
        bufc.seek(0)
        imagec = Image.open(bufc)
        coverage_goal_array = np.array(imagec)
        plt.close()
              
        return coverage_goal_array
                       
    def draw_subgoal(self, bg):
        map_size = [-4, 20]
        if self.env.env_name == 'AntMaze':
            # -4~ 20
            map_size = [-4, 20]
            Map_x, Map_y = (24, 24)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeBottleneck':
            map_size = [-4, 20]
            Map_x, Map_y = (24, 24)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeMultiPathBottleneck':
            map_size = [-4, 20]   
        elif self.env.env_name == 'AntMazeSmall-v0':
            # -2 ~ 12
            map_size = [-2, 12]
            Map_x, Map_y = (12, 12)
            start_x, start_y = (2,2)
        elif self.env.env_name == 'AntMazeS':
            map_size = [-4, 36]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeW':
            map_size_x = [-4, 36]
            map_size_y = [-12, 28]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 12)
        elif self.env.env_name == 'AntMazeP':
            map_size_x = [-12, 28]
            map_size_y = [-4, 36]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (12, 4)
        elif self.env.env_name == 'AntMazeComplex-v0':
            # -4 ~ 52
            map_size = [-4, 52]
            Map_x, Map_y = (56, 56)
            start_x, start_y = (4, 4)

        fig4, ax4 = plt.subplots()
        if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW':
            ax4.set_xlim(map_size_x)
            ax4.set_ylim(map_size_y)
        else:
            ax4.set_xlim(map_size)
            ax4.set_ylim(map_size)
        x_vertex = []
        y_vertex = []
        x_vertex_ag = []
        y_vertex_ag = []        
        for number, value in enumerate(zip(self.subgoal_list, self.ag_list)):
            #print(curr_goal)
            sub_goal, ag = value
            x_vertex.append(sub_goal[0])
            y_vertex.append(sub_goal[1])
            x_vertex_ag.append(ag[0])
            y_vertex_ag.append(ag[1])
            ax4.annotate(str(number), (sub_goal[0], sub_goal[1]))
            ax4.annotate(str(number), (ag[0], ag[1]))

        x_goal, y_goal = bg[0], bg[1]
        ax4.scatter([x_goal], [y_goal], c='r', marker='o', alpha=1)
        ax4.scatter(x_vertex, y_vertex, c='b', marker='o', alpha=1)
        ax4.scatter(x_vertex_ag, y_vertex_ag, c='g', marker='o', alpha=1)
        buf4 = io.BytesIO()
        fig4.savefig(buf4, format='png')
        buf4.seek(0)
        image1 = Image.open(buf4)
        sub_goal_array = np.array(image1)
        plt.close()
        return sub_goal_array
    
    def plot_path(self, ob, bg):
        if self.env.env_name == 'AntMaze':
            size = 48
        elif self.env.env_name == 'AntMazeBottleneck':
            size = 48
        elif self.env.env_name == 'AntMazeComplex-v0':
            size = 112
        
        self.graphplanner.graph_construct(0)
        self.graphplanner.find_path(ob, bg)
        path = np.concatenate((np.reshape(ob[:2], (1,2)), np.reshape(self.graphplanner.landmarks[self.graphplanner.waypoint_vec[:]], (-1,2))), axis=0)
        path = np.concatenate((path, np.reshape(bg[:2], (1,2))), axis = 0)
                
        plt.figure(figsize=(6, 6))
        
        plt.xlim(0, size)
        plt.ylim(0, size)
        plt.xticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
        plt.yticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
        
        if self.env.env_name == 'AntMaze':
            plt.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        
        plt.plot(self.graphplanner.landmarks[self.graphplanner.graph.nodes, 0]*2+8, self.graphplanner.landmarks[self.graphplanner.graph.nodes, 1]*2+8, 'ko')
        plt.plot(self.graphplanner.landmarks[self.graphplanner.waypoint_vec[-1], 0]*2+8, self.graphplanner.landmarks[self.graphplanner.waypoint_vec[-1], 1]*2+8, 'go')
        plt.plot(path[:, 0]*2+8, path[:, 1]*2+8, 'k') 
        plt.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'bo')
        plt.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ro')
        
        plt.savefig('path.png')
        
    def plot_dist_from_point(self, ob, bg, sg=None, agent=None, nog = False):
        if self.env.env_name == 'AntMaze':
            size = 48
            square = 8
        elif self.env.env_name == 'AntMazeSmall-v0':
            size = 24
            square = 4
        elif self.env.env_name == 'AntMazeBottleneck':
            size = 48
            square = 8        
        elif self.env.env_name == 'AntMazeComplex-v0':
            size = 112
            square = 8
        
        if self.env.env_name == 'AntMazeSmall-v0':
            x = np.arange(-1.75, ((size-square)/2), 0.5)
            y = np.arange(-1.75, ((size-square)/2), 0.5)
        else:
            x = np.arange(-3.75, ((size-square)/2), 0.5)
            y = np.arange(-3.75, ((size-square)/2), 0.5)

        X, Y = np.meshgrid(x, y)
        bgs = np.array([X.flatten(),Y.flatten()]).T

        if nog == True:
            dists = agent._get_dist_from_start_nog(ob, bgs).reshape((size, size))
        else:
            dists = agent._get_dist_from_start(ob, bgs).reshape((size, size))
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the distance heatmap
        im = ax.pcolor(dists)
        plt.colorbar(im)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        if self.env.env_name == 'AntMazeSmall-v0':
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-2, (size-square)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-2, (size-square)/2+1, 4))
        else:
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-4, (size-square)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-4, (size-square)/2+1, 4))

        if self.env.env_name == 'AntMaze':
            ax.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        elif self.env.env_name == 'AntMazeSmall-v0':
            ax.plot([0, 16, 16, 0], [8, 8, 16, 16], 'k')

        if ob is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(ob[0]*2 + 4., ob[1]*2 + 4., 'ro')
            else:
                ax.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'ro')

        if bg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(bg[0]*2 + 4., bg[1]*2 + 4., 'ko')
            else:
                ax.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ko')
        
        if sg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(sg[0]*2 + 4., sg[1]*2 + 4., 'bo')
            else:
                ax.plot(sg[0]*2 + 8., sg[1]*2 + 8., 'bo')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        dist_from_point_plot_array = np.array(image)
        plt.close()
        return dist_from_point_plot_array
                
        # plt.figure(figsize=(8, 6))
        # plt.pcolor(dists)
        # plt.colorbar()
        
        # plt.xlim(0, size)
        # plt.ylim(0, size)
        # plt.xticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
        # plt.yticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
        
        # if self.env.env_name == 'AntMaze':
        #     plt.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        
        # if ob is not None:
        #     plt.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'bo')
        
        # if bg is not None:
        #     plt.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ro')
        
        # plt.savefig('dist_from_point.png')
    
    
    def plot_dist_to_point(self, ob, bg, sg, agent, nog = False):
        
        if self.env.env_name == 'AntMaze':
            size = 48
            square = 8
        elif self.env.env_name == 'AntMazeSmall-v0':
            size = 24
            square = 4        
        elif self.env.env_name == 'AntMazeBottleneck':
            size = 48
            square = 8
        elif self.env.env_name == 'AntMazeComplex-v0':
            size = 112
            square = 8
        else:
            size = 100
            square = 8
        if self.env.env_name == 'AntMazeSmall-v0':
            x = np.arange(-1.75, ((size-square)/2), 0.5)
            y = np.arange(-1.75, ((size-square)/2), 0.5)
        else:
            x = np.arange(-3.75, ((size-square)/2), 0.5)
            y = np.arange(-3.75, ((size-square)/2), 0.5)
        X, Y = np.meshgrid(x, y)
        obs = np.array([X.flatten(),Y.flatten()]).T
        pos = np.zeros((obs.shape[0], 27))
        obs = np.concatenate((obs, pos), axis = 1)
        
        if nog == True:
            dists = agent._get_dist_to_goal_nog(obs, bg).reshape((size, size))
        else:
            dists = agent._get_dist_to_goal(obs, bg).reshape((size, size))
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the distance heatmap
        im = ax.pcolor(dists)
        plt.colorbar(im)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        if self.env.env_name == 'AntMazeSmall-v0':
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-2, (size-4)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-2, (size-4)/2+1, 4))
        else:
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))

        if self.env.env_name == 'AntMaze':
            ax.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        elif self.env.env_name == 'AntMazeSmall-v0':
            ax.plot([0, 16, 16, 0], [8, 8, 16, 16], 'k')

        if ob is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(ob[0]*2 + 4., ob[1]*2 + 4., 'ro')
            else:
                ax.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'ro')

        if bg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(bg[0]*2 + 4., bg[1]*2 + 4., 'ko')
            else:
                ax.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ko')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        dist_to_point_plot_array = np.array(image)
        plt.close()
        return dist_to_point_plot_array    
        # plt.figure(figsize=(8, 6))
        # plt.pcolor(dists)
        # plt.colorbar()
        
        # plt.xlim(0, size)
        # plt.ylim(0, size)
        # plt.xticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
        # plt.yticks(np.arange(0, size+1, 8), np.arange(-4, (size-8)/2+1, 4))
        
        # if self.env.env_name == 'AntMaze':
        #     plt.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        
        # if ob is not None:
        #     plt.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'bo')
        
        # if bg is not None:
        #     plt.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ro')
        
        # plt.savefig('dist_to_point.png')
    
    def plot_value(self, ob, bg, sg=None, agent=None):
        if self.env.env_name == 'AntMaze':
            size = 48
            square = 8
        elif self.env.env_name == 'AntMazeSmall-v0':
            size = 24
            square = 4        
        elif self.env.env_name == 'AntMazeBottleneck':
            size = 48
            square = 8            
        elif self.env.env_name == 'AntMazeComplex-v0':
            size = 112
            square = 8
        else:
            size = 100
            square = 8
        
        if self.env.env_name == 'AntMazeSmall-v0':
            x = np.arange(-1.75, ((size-square)/2), 0.5)
            y = np.arange(-1.75, ((size-square)/2), 0.5)
        else:
            x = np.arange(-3.75, ((size-square)/2), 0.5)
            y = np.arange(-3.75, ((size-square)/2), 0.5)

        X, Y = np.meshgrid(x, y)
        sgs = np.array([X.flatten(),Y.flatten()]).T

        ob_repeat = np.ones((sgs.shape[0], np.squeeze(ob).shape[0])) * np.expand_dims(ob, axis=0)
        bg_repeat = np.ones((sgs.shape[0], np.squeeze(bg).shape[0])) * np.expand_dims(bg, axis=0)
        with torch.no_grad():
            values = agent.get_qs(ob_repeat, bg_repeat, sgs).reshape((size, size)).detach().cpu().numpy()
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the distance heatmap
        im = ax.pcolor(values)
        plt.colorbar(im)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        if self.env.env_name == 'AntMazeSmall-v0':
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-2, (size-square)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-2, (size-square)/2+1, 4))
        else:
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-4, (size-square)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-4, (size-square)/2+1, 4))

        if self.env.env_name == 'AntMaze':
            ax.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        elif self.env.env_name == 'AntMazeSmall-v0':
            ax.plot([0, 16, 16, 0], [8, 8, 16, 16], 'k')

        if ob is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(ob[0]*2 + 4., ob[1]*2 + 4., 'ro')
            else:
                ax.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'ro')

        if bg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(bg[0]*2 + 4., bg[1]*2 + 4., 'ko')
            else:
                ax.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ko')
        
        if sg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(sg[0]*2 + 4., sg[1]*2 + 4., 'bo')
            else:
                ax.plot(sg[0]*2 + 8., sg[1]*2 + 8., 'bo')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        value_from_point_plot_array = np.array(image)
        plt.close()
        return value_from_point_plot_array

    def plot_uncertainty(self, ob, bg, sg=None, agent=None):
        if self.env.env_name == 'AntMaze':
            size = 48
            square = 8
        elif self.env.env_name == 'AntMazeSmall-v0':
            size = 24
            square = 4  
        elif self.env.env_name == 'AntMazeBottleneck':
            size = 48
            square = 8      
        elif self.env.env_name == 'AntMazeComplex-v0':
            size = 112
            square = 8
        
        if self.env.env_name == 'AntMazeSmall-v0':
            x = np.arange(-1.75, ((size-square)/2), 0.5)
            y = np.arange(-1.75, ((size-square)/2), 0.5)
        else:
            x = np.arange(-3.75, ((size-square)/2), 0.5)
            y = np.arange(-3.75, ((size-square)/2), 0.5)

        X, Y = np.meshgrid(x, y)
        sgs = np.array([X.flatten(),Y.flatten()]).T

        uncertainty = agent._get_dist_dist_from_start(ob, sgs).reshape((size, size))
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the distance heatmap
        im = ax.pcolor(uncertainty)
        plt.colorbar(im)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        if self.env.env_name == 'AntMazeSmall-v0':
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-2, (size-square)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-2, (size-square)/2+1, 4))
        else:
            ax.set_xticks(np.arange(0, size+1, 8), np.arange(-4, (size-square)/2+1, 4))
            ax.set_yticks(np.arange(0, size+1, 8), np.arange(-4, (size-square)/2+1, 4))

        if self.env.env_name == 'AntMaze':
            ax.plot([0, 32, 32, 0], [16, 16, 32, 32], 'k')
        elif self.env.env_name == 'AntMazeSmall-v0':
            ax.plot([0, 16, 16, 0], [8, 8, 16, 16], 'k')

        if ob is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(ob[0]*2 + 4., ob[1]*2 + 4., 'ro')
            else:
                ax.plot(ob[0]*2 + 8., ob[1]*2 + 8., 'ro')

        if bg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(bg[0]*2 + 4., bg[1]*2 + 4., 'ko')
            else:
                ax.plot(bg[0]*2 + 8., bg[1]*2 + 8., 'ko')
        
        if sg is not None:
            if self.env.env_name == 'AntMazeSmall-v0':
                ax.plot(sg[0]*2 + 4., sg[1]*2 + 4., 'bo')
            else:
                ax.plot(sg[0]*2 + 8., sg[1]*2 + 8., 'bo')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        uncertainty_from_point_plot_array = np.array(image)
        plt.close()
        return uncertainty_from_point_plot_array