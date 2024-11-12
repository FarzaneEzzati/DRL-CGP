"""
Environment for Chavatal Gomory Cut agent.
Created by: Farzane Ezzati and Reza Mirjalili
"""
import copy
import math
import random

import numpy as np


class Chavatal():
    def __init__(self):
        self._capacity = 1000
        self._constraints = 4
        self._A = [[13, 13, -5, 19, 0,  22],
                   [10,	1,	5,	-4,	0,	21],
                   [-1,	12,	16,	19,	0,	32],
                   [7,	-2,	11,	5,	0,	20]]  # _A = [A | b]
        self._x = [0.7143, 1, 1, 0.2481, -1]  # _x = [x | -1 ]
        self._remaining_capacity = copy.copy(self._constraints)
        self._actions = np.zeros(self._constraints)
        self._step = 0  # step is integer and between 0 and self._constraints, excluding the latter
        self._state = [self._step, self._remaining_capacity]
        self._done = True

    def reset(self):
        self._step = 0
        self._remaining_capacity = copy.copy(self._constraints)
        # here it returns Python array.)
        self._state = [self._step, self._remaining_capacity]
        self._done = False
        return self._state

    def step(self, action: int):
        """
        Args:
            action: discrete value between 0 and self._capacity
        """
        if self._done:
            raise Exception("Cannot run step() before reset")

        # update state
        self._actions[self._step] = action  # save the action taken at node step
        self._remaining_capacity -= action
        self._state = [self._step, self._remaining_capacity]

        # determine reward
        reward = np.inner(np.floor(np.divide(action/self._capacity)*self._A[self._step]), self._x)
        self._step += 1

        # check whether the episode is done
        self._done = 1 if self._remaining_capacity <= 0 else 0
        return self._state, reward, {}
