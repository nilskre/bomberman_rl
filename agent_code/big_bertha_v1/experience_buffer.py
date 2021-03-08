import numpy as np

from agent_code.big_bertha_v1.parameters import NUMBER_OF_ACTIONS, STATE_SHAPE, BATCH_SIZE, EXPERIENCE_BUFFER_SIZE


class ExperienceBuffer(object):
    def __init__(self):
        self.size = EXPERIENCE_BUFFER_SIZE
        self.memory_index = 0
        self.filled = False

        # TODO: Normalize data for more efficient training, use float instead of int for numpy arrays
        self.states = np.zeros((self.size, STATE_SHAPE), dtype=np.int8)  # depending on Nils state representation
        self.actions = np.zeros((self.size, NUMBER_OF_ACTIONS), dtype=np.int8)  # one hot encoded
        self.rewards = np.zeros(self.size, dtype=np.int16)
        self.next_states = np.zeros((self.size, STATE_SHAPE), dtype=np.int8)

    def remember(self, state, action, reward, next_state):
        self.states[self.memory_index] = state
        self.actions[self.memory_index] = action
        self.rewards[self.memory_index] = reward
        self.next_states[self.memory_index] = next_state
        self.memory_index = (self.memory_index + 1) % self.size
        if self.memory_index == 0:
            self.filled = True

    def sample(self, batch_size=BATCH_SIZE):
        if self.filled:
            indices = np.random.choice(self.size, batch_size)
        else:
            # TODO: Only call sampling when number of instances remembered is greater or equal to the batch size
            indices = np.random.choice(self.memory_index + 1, batch_size)

        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices]
