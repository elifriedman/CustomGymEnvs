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

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -100.
        self.max_position = 100.
        self.max_speed = 20.
        self.goal_state = np.array([0, 0])
        self.tolerance = 0.5

        self.prev_info = None
        self.curr_info = None
        self.t = 0

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.prev_info = self.curr_info

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0
        if (position==self.max_position and velocity>0): velocity = 0

        done = bool(np.sum(np.square(self.state - self.goal_state)) < self.tolerance)

        reward = np.array([-1., -abs(action[0])])
        if done:
            reward = np.zeros(2)

        self.state = np.array([position, velocity])

        self.curr_info = [position/2, velocity*5, force*20]
        self.t += 1
        return self.state, sum(reward), done, {"reward": reward}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-5, high=5), 0])
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scaler = screen_width/world_width

        COLORS = np.eye(3)
        BLANK = np.zeros(4)
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
                prev, curr = self.prev_info[i], self.curr_info[i]
                line = rendering.Line(start=(scaler*(t-1), prev+upscale*i + upscale/2), end=(scaler*t, curr+upscale*i + upscale/2))
                line.set_color(*COLORS[i])
                line.linewidth.stroke = 2
                self.lines.append(line)
                self.viewer.add_geom(line)

            if t == 0:
                for line in self.lines:
                    line._color.vec4 = BLANK
                self.lines = []


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
