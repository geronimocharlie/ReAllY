from gym.envs.registration import register
from gridworlds.envs.gridworld import GridWorld
from gridworlds.envs.gridworld_view import GridWorld_View
from gridworlds.envs.gridworld_global import GridWorld_Global
from gridworlds.envs.gridworld_global_multi import GridWorld_Global_Multi
register(
    id="gridworld-v0",
    entry_point="gridworlds.envs:GridWorld",
    max_episode_steps=100000,
)
register(
    id="gridworld-v1",
    entry_point="gridworlds.envs:GridWorld_Global",
    max_episode_steps=100000,
)
register(
    id="gridworld-v2",
    entry_point="gridworlds.envs:GridWorld_View",
    max_episode_steps=100000,
)

register(
    id="gridworld-v3",
    entry_point="gridworlds.envs:GridWorld_Multi",
    max_episode_steps=100000,
)
