import glob

import numpy as np
from tensorflow.python.keras.models import load_model

from agent_code.big_bertha_v1.features import state_to_features
from agent_code.big_bertha_v1.model import compile_deep_q_network
from agent_code.big_bertha_v1.parameters import ACTIONS, NUMBER_OF_ACTIONS


def setup(self):
    if self.train:
        self.logger.info("Setting up model from scratch...")

        self.model = compile_deep_q_network()
        self.predict_model = compile_deep_q_network()
        self.predict_model.set_weights(self.model.get_weights())
    else:
        self.logger.info("Loading model...")
        for filename in glob.glob("models/*_recent"):
            self.predict_model = load_model(filename)


def act(self, game_state: dict) -> str:
    feature_vector = state_to_features(game_state)
    feature_vector = feature_vector[np.newaxis, :]
    rand = np.random.random()
    if self.train and rand < self.epsilon:
        action = np.random.randint(NUMBER_OF_ACTIONS)
    else:
        actions = self.predict_model.predict(feature_vector)
        action = np.argmax(actions[0])

    return ACTIONS[action]
