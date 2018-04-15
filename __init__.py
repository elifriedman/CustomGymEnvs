from gym.envs.registration import register

from .oned_slider import OneDSlider

register(
    id='OneDSlider-v0',
    entry_point='custom_envs:OneDSlider',
)

