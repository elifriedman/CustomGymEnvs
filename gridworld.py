# -*- coding: utf-8 -*-
"""
@author: Eli Friedman

"""

import gym, gym.spaces as spaces
from gym.utils import seeding
import numpy as np

class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    APPLE = np.array([1, 0, 0, 0])
    PEAR  = np.array([0, 1, 0, 0])
    BLUEB = np.array([0, 0, 1, 0])

    def __init__(self, N=5):

        self.N = N
        self.action_space = spaces.MultiDiscrete([2, N**2])

        self.items = {'a': self.APPLE, 'p': self.PEAR, 'b': self.BLUEB}
        self.grid = np.zeros([N, N, 4])
        self.locations = {}
        self.griditems = {'a': [], 'p': [], 'b': []}


        self.observation_space = spaces.Box(low=0, high=1, shape=(N, N, 4), dtype=np.int)
        # TODO make spaces.Tuple

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def command(self, command):
        pass

    def step(self, action):
        action, location = action
        if action == 0:  # mark
            self.grid[location[0], location[1], 3] = 1
        elif action == 1: # pick up / put down
            pass

        return self.grid, reward, done, info

    def reset(self):

        self.grid = np.zeros([self.N, self.N, 4])
        self.locations = {}
        self.griditems = {'a': [], 'p': [], 'b': []}

        for i in range(10):
            key = self.np_random.choice(list(self.items.keys()))
            item = self.items[key]
            pos = self.np_random.randint(self.N, size=2)
            if tuple(pos) in self.locations:
                continue
            self.locations[tuple(pos)] = key
            self.griditems[key].append(pos)

            self.grid[pos[0], pos[1], :] = item

        return self.grid

    def render(self, mode='human'):
        pass
