from gym.envs.registration import register

from .oned_slider import OneDSlider
from .nd_slider import NDSlider

register(
    id='OneDSlider-v0',
    entry_point='custom_envs:OneDSlider',
    max_episode_steps=208,
    kwargs={
        "her": False,
    },
)

register(
    id='NDSlider-v0',
    entry_point='custom_envs:NDSlider',
    max_episode_steps=208,
    kwargs={
        "her": True,
        "N": 3,
    },
)