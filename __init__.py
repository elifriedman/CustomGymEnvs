from gym.envs.registration import register

from .oned_slider import OneDSlider
from .nd_slider import NDSlider

register(
    id='OneDSlider-v0',
    entry_point='CustomGymEnvs:OneDSlider',
    max_episode_steps=2048,
    kwargs={
        "her": True,
        "goal_sample": "pos", # options are 'zero', 'pos', 'vel'
        "weight_sample": "rand", # options are "rand", "const"
    },
)

register(
    id='NDSlider-v0',
    entry_point='CustomGymEnvs:NDSlider',
    max_episode_steps=2048,
    kwargs={
        "her": True,
        "N": 3,
        "goal_sample": "pos", # options are 'zero', 'pos', 'vel'
        "weight_sample": "rand", # options are "rand", "const"
    },
)
