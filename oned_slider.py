# -*- coding: utf-8 -*-
"""
@author: Eli Friedman

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from 
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class OneDSlider(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, her=True):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -100.
        self.max_position = 100.
        self.max_speed = 20.

        self.goal = None  # initialized in self.reset()
        self.tolerance = 0.5
        self._max_episode_steps = 10000
        self.weights = np.array([0.5, 0.5])

        self.prev_render_info = None
        self.curr_render_info = None
        self.t = 0

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype="float32")

        self.her = her
        if self.her:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=self.low_state, high=self.high_state, dtype="float32"),
                achieved_goal=spaces.Box(low=self.low_state, high=self.high_state, dtype="float32"),
                observation=spaces.Box(low=self.low_state, high=self.high_state, dtype="float32"),
            ))
        else:
            self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype="float32")

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.prev_render_info = self.curr_render_info

        position, velocity = self.state
        action = float(np.clip(action, self.action_space.low, self.action_space.high))

        velocity += action
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0
        if (position==self.max_position and velocity>0): velocity = 0

        self.state[:] = [position, velocity]
        done = bool(np.linalg.norm(self.state - self.goal) < self.tolerance) or self.t >= self._max_episode_steps

        # Used by compute_reward() and _compute_vector_reward()
        # Needed, in order to make the env compatible w/ HER code
        info = {
            "weights": self.weights,
            "action": action,
        }
        vector_reward = self.compute_reward(self.state, self.goal, info, vector=True)
        reward = vector_reward.dot(info["weights"])

        obs = self._make_obs()

        info.update({
            'is_success': done,
            'vector_reward': vector_reward,
        })

        self.curr_render_info = [position/2, velocity*5, action*20]
        self.t += 1
        if self.her:
            return obs, reward, done, info
        else:
            return self.state.copy(), reward, done, info

    def compute_reward(self, achieved_goal, goal, info, vector=False):
        weights = info["weights"]
        action = info["action"]
        dist = np.linalg.norm(achieved_goal - goal, axis=-1)
        if len(dist.shape) == 0:
            reward = np.zeros(2) if dist <= self.tolerance else np.array([-1, -abs(action)])
            return reward.dot(weights) if not vector else reward
        else:
            cond = (dist <= self.tolerance).reshape(-1, 1).repeat(2, axis=1)  # make it (N, 2)
            neg_reward = -1 * np.ones_like(cond)
            neg_reward[:, 1] = action.squeeze()
            reward = np.where(cond, np.zeros_like(cond), neg_reward)
            return np.sum(reward * weights, axis=1) if not vector else reward

    def _sample_goal(self):
        # randomize only goal position or also velocity?
        # return np.array([self.np_random.uniform(low=self.low_state[0], high=self.high_state[0]), 0])
        return np.random.uniform(low=self.low_state, high=self.high_state)

    def _make_obs(self):
        return {
            'observation': self.state.copy(),
            'achieved_goal': self.state.copy(),
            'desired_goal': self.goal.copy(),
        }

    def reset(self):
        self.goal = self._sample_goal()
        self.state = np.array([self.np_random.uniform(low=self.low_state[0], high=self.high_state[0]), 0])
        self.prev_render_info = None
        self.curr_render_info = None
        if self.viewer:
            self.clear_render()
        self.t = 0
        return self._make_obs()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scaler = screen_width/world_width

        COLORS = np.eye(3)
        upscale = 150

        if self.viewer is None:
            self.lines = []
            carwidth = 20
            carheight = 75
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = np.ones(100)*screen_height / 2
            xys = list(zip((xs-self.min_position)*scaler, ys))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            for i in range(3):
                ys = np.ones(100) * upscale * i + upscale / 2
                xys = list(zip((xs-self.min_position)*scaler, ys))

                zeros = rendering.make_polyline(xys)
                self.viewer.add_geom(zeros)

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, screen_height/2 - carheight/2)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scaler, 0)
        

        if self.t > 1:
            from gym.envs.classic_control import rendering
            t = self.t % (screen_width / scaler)
            for i in range(3):
                prev, curr = self.prev_render_info[i], self.curr_render_info[i]
                line = rendering.Line(start=(scaler*(t-1), prev+upscale*i + upscale/2), end=(scaler*t, curr+upscale*i + upscale/2))
                line.set_color(*COLORS[i])
                line.linewidth.stroke = 2
                self.lines.append(line)
                self.viewer.add_geom(line)

            if t == 0:
                self.clear_render()

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def clear_render(self):
        BLANK = np.zeros(4)
        for line in self.lines:
            line._color.vec4 = BLANK
        self.lines = []

    def close(self):
        if self.viewer: self.viewer.close()
