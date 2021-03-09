from tensorflow.python.keras.models import load_model

from agent_code.big_bertha_v1.experience_buffer import ExperienceBuffer
from agent_code.big_bertha_v1.model import compile_deep_q_network
from agent_code.big_bertha_v1.parameters import NUMBER_OF_ACTIONS, BATCH_SIZE, GAMMA, EPSILON_START, EPSILON_DECAY, EPSILON_END

import numpy as np


class Agent(object):
    def __init__(self):
        self.experience_buffer = ExperienceBuffer()
        self.model = compile_deep_q_network()
        self.epsilon = EPSILON_START

    def remember(self, state, action, reward, new_state):
        self.experience_buffer.remember(state, action, reward, new_state)

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.randint(NUMBER_OF_ACTIONS)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.experience_buffer.filled or self.experience_buffer.memory_index >= BATCH_SIZE:
            states, actions, rewards, new_states = self.experience_buffer.sample()

            qs_current = self.model.predict(states)
            qs_next = self.model.predict(new_states)
            qs_target = qs_current.copy()

            batch_index = np.arange(BATCH_SIZE, dtype=np.int8)

            qs_target[batch_index, actions] = rewards + GAMMA * np.max(qs_next)
            _ = self.model.fit(states, qs_target, verbose=0)

            self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END

    def save_model(self):
        self.model.save('model.h5')

    def load_model(self):
        self.model = load_model('model.h5')
