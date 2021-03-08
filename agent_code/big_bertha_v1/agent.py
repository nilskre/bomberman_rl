from tensorflow.python.keras.models import load_model

from agent_code.big_bertha_v1.experience_buffer import ExperienceBuffer
from agent_code.big_bertha_v1.model import compile_deep_q_network
from agent_code.big_bertha_v1.parameters import NUMBER_OF_ACTIONS, EPSILON

import numpy as np


class Agent(object):
    def __init__(self):
        self.experience_buffer = ExperienceBuffer()
        self.model = compile_deep_q_network()
        self.action_space = [i for i in range(NUMBER_OF_ACTIONS)]

    def remember(self, state, action, reward, new_state):
        self.experience_buffer.remember(state, action, reward, new_state)

    def choose_action(self, state):
        rand = np.random.random()
        if rand < EPSILON:
            action = np.random.choice(self.action_space)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        pass  # TODO

    def save_model(self):
        self.model.save('model.h5')

    def load_model(self):
        self.model = load_model('model.h5')
