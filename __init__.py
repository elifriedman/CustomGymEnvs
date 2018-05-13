from gym.envs.registration import register

from .oned_slider import OneDSlider
from .nd_slider import NDSlider

register(
    id='OneDSlider-v0',
    entry_point='CustomGymEnvs:OneDSlider',
    max_episode_steps=2048,
    kwargs={
        "her": True,
    },
)

register(
    id='NDSlider-v0',
    entry_point='CustomGymEnvs:NDSlider',
    max_episode_steps=2048,
    kwargs={
        "her": True,
        "N": 3,
    },
)
