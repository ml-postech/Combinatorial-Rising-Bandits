GPU=$1
SEED=$2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

python DHRL/main.py \
--env_name 'AntMazeCB-v0' \
--test_env_name 'AntMazeCB-v0' \
--action_max 30. \
--max_steps 500 \
--start_planning_epoch 0 \
--n_cycles 1 \
--n_test_rollouts 1 \
--high_future_step 1 \
--subgoal_freq 500 \
--subgoal_scale 10. 6. \
--subgoal_offset 8. 4. \
--low_future_step 100 \
--subgoaltest_threshold 1 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--cutoff 15 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--subgoal_noise_eps 2 \
--n_epochs 2001 \
--cuda_num ${GPU} \
--seed ${SEED} \
--eval_interval 100000 \
--method 'grid8' \
--high_future_p 0.9 \
--high_penalty 0.3 \
--uncertainty 'value' \
--alpha 0.0 \
--beta 0.0 \
--nosubgoal \
--setting 'CRUCB' \
--init_dist 8.0 \
--waypoint_chase_step_threshold 100. \
--waypoint_count_threshold 3 
