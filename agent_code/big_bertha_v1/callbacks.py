import glob

import numpy as np
from agent_code.big_bertha_v1.features import state_to_features
from agent_code.big_bertha_v1.model import compile_deep_q_network
from agent_code.big_bertha_v1.parameters import ACTIONS, NUMBER_OF_ACTIONS
from tensorflow.python.keras.models import load_model


def setup(self):
    if self.train:
        self.logger.info("Setting up model from scratch...")

        self.online_model = compile_deep_q_network()
        self.target_model = compile_deep_q_network()
        self.target_model.set_weights(self.online_model.get_weights())
    else:
        self.logger.info("Loading model...")
        for filename in glob.glob("models/*_recent"):
            self.online_model = load_model(filename)


def act(self, game_state: dict) -> str:
    feature_vector = state_to_features(game_state)
    feature_vector = feature_vector[np.newaxis, :]
    actions = self.online_model.predict(feature_vector)

    rand = np.random.random()
    if self.train and rand < self.epsilon:
        # Exploration function or
        # action = np.argmax(np.random.multinomial(1, actions[0]))
        # or:
        action = np.random.randint(NUMBER_OF_ACTIONS)
    else:
        action = np.argmax(actions[0])

    return ACTIONS[action]
