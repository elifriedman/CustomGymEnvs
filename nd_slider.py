# -*- coding: utf-8 -*-
"""
@author: Eli Friedman

A double integrator task. The agent controls the acceleration of a unit mass object sliding along a line.
The reward is a trade off between time-to-goal [T] and fuel-use [u] (absolute value of control input). It's
controlled by the weight env.weights = [alpha, 1 - alpha], where 0 <= alpha <= 1 controls how valuable it is
to minimize time.

The optimal solution is a bang-off-bang controller with 
      1  t < t1
u = { 0  t1 < t < t2
     -1  t2 < t
where
t1 = sqrt( abs(start_pos - goal_pos) * alpha / (2 - alpha))
t2 = sqrt( abs(start_pos - goal_pos) * (2 - alpha) / alpha)

time-to-goal = sqrt( abs(start_pos - goal_pos) / (alpha * (2 - alpha)))

Solution found with the help of [1]
[1] https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-323-principles-of-optimal-control-spring-2008/lecture-notes/lec9.pdf
"""

import math
import gym, gym.spaces as spaces
from gym.utils import seeding
import numpy as np

class NDSlider(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, her=True, N=2, goal_sample='pos', weight_sample='rand'):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -100.
        self.max_position = 100.
        self.max_speed = 20.
        self.N = N
        self.her = her

        self.goal = None  # initialized in self.reset()
        self.weights = None  # initialized in self.reset()
        self.tolerance = 1.0
        self.OBS_SAMPLE_STRATEGY = goal_sample  # options are 'zero', 'pos', 'vel'
        self.WEIGHT_SAMPLE_STRATEGY = weight_sample  # options are 'const', 'rand'
        if 'grid' in weight_sample:
            weight_density = int(weight_sample.split("_")[1])
            self._init_weights(weight_density)
            self.WEIGHT_SAMPLE_STRATEGY = 'grid'

        self.prev_render_info = None
        self.curr_render_info = None
        self.t = 0

        self.low_state = np.array([self.min_position]*N + [-self.max_speed]*N)
        self.high_state = np.array([self.max_position]*N + [self.max_speed]*N)

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(N,), dtype="float32")

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

        position, velocity = self.state[:self.N], self.state[self.N:]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        s = np.abs(action).sum()
        action = action / s if s > 1 else action

        velocity += action
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        velocity[(position==self.min_position)*(velocity<0)] = 0
        velocity[(position==self.max_position)*(velocity>0)] = 0

        self.state = np.concatenate([position, velocity])
        done = bool(np.linalg.norm(self.state - self.goal) < self.tolerance)

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

        self.curr_render_info = [position/2, velocity*5, action*25]
        self.t += 1
        if self.her:
            return obs, reward, done, info
        else:
            return self.state.copy(), reward, done, info

    def compute_reward(self, achieved_goal, goal, info, vector=False):
        """Reward is (0, 0) for each position dimension that's close to the goal and (-1, -|u|)
           for each dimension that's not.
        """
        weights = info["weights"]
        action = info["action"]

        unbatched = len(achieved_goal.shape) == 1  # didn't pass in a batch
        if unbatched:
            achieved_goal = achieved_goal.reshape(1, -1)
            goal = goal.reshape(1, -1)

        diff = np.square(achieved_goal - goal)
        positions, vels = diff[:, :self.N], diff[:, self.N:2*self.N]  # calculate per-dimension
        dist = np.sqrt(positions + vels)
        cond = dist <= self.tolerance

        # reward shape is K x 2*N where K is batch size and N is number of position dimensions (ie self.N)
        # 2*N because first N elements for time reward and second N elements for -|action| reward
        cond = np.tile(cond, 2)  # [cond, cond] is K x 2*N first cond for time, second cond for action
        reward = np.where(cond,
                          np.zeros((cond.shape[0], 2*self.N)),
                          np.concatenate([ -np.ones(action.shape), -np.abs(action)], axis=-1))

        if vector:
            return reward.squeeze() if unbatched else reward
        elif unbatched:
            return float(reward.dot(weights)) 
        else:
            return np.sum(reward * weights, axis=1)

    def _sample_goal(self):
        # randomize only goal position or also velocity?
        if self.OBS_SAMPLE_STRATEGY == 'zero':
            return np.zeros(2*self.N)
        elif self.OBS_SAMPLE_STRATEGY == 'pos':
            return np.concatenate([self.np_random.uniform(low=self.low_state[0:self.N], high=self.high_state[:self.N]),
                                   np.zeros(self.N)])
        else:
            return self.np_random.uniform(low=self.low_state, high=self.high_state)

    def _init_weights(self, weight_density=3):
        weights = np.meshgrid(*[np.linspace(0, 1, weight_density)]*(2*self.N - 1))
        weights = np.array([w.reshape(-1) for w in weights]).T
        weights = weights[np.sum(weights, axis=1) <= 1]
        self.sample_weights = np.concatenate([weights, 1 - weights.sum(axis=1, keepdims=True)], axis=1)

    def _sample_weights(self):
        if self.WEIGHT_SAMPLE_STRATEGY == 'const':
            w = np.concatenate([np.ones(self.N), np.zeros(self.N)])
        elif self.WEIGHT_SAMPLE_STRATEGY == 'rand':
            # generate uniformly over the simplex (=multidimensional line-segment, triangle, ...)
            # https://stats.stackexchange.com/questions/14059/generate-uniformly-distributed-weights-that-sum-to-unity
            w = self.np_random.rand(2*self.N)
            w = -np.log(w)
        elif self.WEIGHT_SAMPLE_STRATEGY == 'grid':
            idx = np.random.choice(range(len(self.sample_weights)))
            w = self.sample_weights[idx]
        return w / w.sum()

    def _make_obs(self):
        if self.her:
            return {
                'observation': self.state.copy(),
                'achieved_goal': np.concatenate([self.state, self.weights]),
                'desired_goal': np.concatenate([self.goal, self.weights]),
            }
        else:
            return self.state.copy()

    def reset(self):
        self.goal = self._sample_goal()
        self.weights = self._sample_weights()
        self.state = np.concatenate([self.np_random.uniform(low=self.low_state[0:self.N], high=self.high_state[:self.N]),
                                     np.zeros(self.N)])
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
