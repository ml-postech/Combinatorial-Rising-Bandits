from gym.envs.registration import registry, register, make, spec

register(
        id='AntMazeBottleneck-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeBottleneckEnv',
        max_episode_steps=600,
        reward_threshold=0.0,
    )

register(
        id='AntMazeBottleneck-eval-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeBottleneckEvalEnv',
        max_episode_steps=600,
        reward_threshold=0.0,
    )

register(
        id='AntMazeMultiPathBottleneck-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeMultiPathBottleneckEnv',
        max_episode_steps=600,
        reward_threshold=0.0,
    )

register(
        id='AntMazeMultiPathBottleneck-eval-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeMultiPathBottleneckEvalEnv',
        max_episode_steps=600,
        reward_threshold=0.0,
    )

register(
        id='AntMazeCB-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeCBEnv',
        max_episode_steps=100,
        reward_threshold=0.0,
    )

register(
        id='AntMazeCB-eval-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeCBEvalEnv',
        max_episode_steps=100,
        reward_threshold=0.0,
    )

register(
        id='AntMazeCBcomplex-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeCBcomplexEnv',
        max_episode_steps=100,
        reward_threshold=0.0,
    )

register(
        id='AntMazeCBcomplex-eval-v0',
        entry_point='envs.antenv.ant_maze_bottleneck:AntMazeCBcomplexEvalEnv',
        max_episode_steps=100,
        reward_threshold=0.0,
    )


register(
        id='Reacher3D-v0',
        entry_point='envs.fetchenv.create_fetch_env:create_fetch_env',
        kwargs={'env_name': 'Reacher3D-v0'},
        max_episode_steps=100
)

register(
        id='Pusher-v0',
        entry_point='envs.fetchenv.create_fetch_env:create_fetch_env',
        kwargs={'env_name': 'Pusher-v0'},
        max_episode_steps=100
)