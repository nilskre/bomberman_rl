import glob

import numpy as np
from agent_code.big_bertha_v1.features import state_to_features
from agent_code.big_bertha_v1.model import compile_deep_q_network
from agent_code.big_bertha_v1.parameters import ACTIONS, NUMBER_OF_ACTIONS
from tensorflow.python.keras.models import load_model


def setup(self):
    if self.train:
        self.logger.info("Setting up model from scratch...")
        self.model = compile_deep_q_network()
    else:
        self.logger.info("Loading model...")
        for filename in glob.glob("models/*_recent"):
            with open(filename, "r") as file:
                self.model = load_model(file)


def act(self, game_state: dict) -> str:
    feature_vector = state_to_features(game_state)
    feature_vector = feature_vector[np.newaxis, :]
    rand = np.random.random()
    if self.train and rand < self.epsilon:
        action = np.random.randint(NUMBER_OF_ACTIONS)
    else:
        actions = self.model.predict(feature_vector)
        action = np.argmax(actions)

    return ACTIONS[action]
