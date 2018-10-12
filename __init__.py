from gym.envs.registration import register as _register

from .oned_slider import OneDSlider
from .nd_slider import NDSlider
from .gridworld import GridWorld

def register(env, max_episode_steps=2048, **kwargs):
    _register(
        id='{}-v0'.format(env),
        entry_point='CustomGymEnvs:{}'.format(env),
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )

def register_all():
    register("NDSlider", max_episode_steps=2048, her=True, N=3, goal_sample='zero', weight_sample='rand')
    register("OneDSlider", max_episode_steps=2048, her=True, goal_sample='zero', weight_sample='rand')

# register(
#     id='NDSlider-v0',
#     entry_point='CustomGymEnvs:NDSlider',
#     max_episode_steps=2048,
#     kwargs={
#         "her": True,
#         "N": 3,
#         "goal_sample": "zero", # options are 'zero', 'pos', 'vel'
#         "weight_sample": "rand", # options are "rand", "const"
#     },
# )
