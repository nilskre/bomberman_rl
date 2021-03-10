from collections import deque

import numpy as np

from agent_code.big_bertha_v1.parameters import (BATCH_SIZE,
                                                 EXPERIENCE_BUFFER_SIZE_MAX)


class ExperienceBuffer(object):
    def __init__(self):
        self.size = 0
        self.states = deque(maxlen=EXPERIENCE_BUFFER_SIZE_MAX)
        self.actions = deque(maxlen=EXPERIENCE_BUFFER_SIZE_MAX)
        self.rewards = deque(maxlen=EXPERIENCE_BUFFER_SIZE_MAX)
        self.next_states = deque(maxlen=EXPERIENCE_BUFFER_SIZE_MAX)

    def remember(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.size < EXPERIENCE_BUFFER_SIZE_MAX:
            self.size += 1

    def sample(self, batch_size=BATCH_SIZE):
        indices = np.random.choice(self.size, batch_size)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices]
