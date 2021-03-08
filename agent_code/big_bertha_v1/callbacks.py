import os
import pickle
import random

import numpy as np

from settings import BOMB_TIMER, BOMB_POWER

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    print("GAME STATE")
    print(game_state)
    feature_vector = state_to_features(game_state)
    print("FEATURES")
    print(feature_vector.shape)
    print(feature_vector)

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []

    field_matrix = game_state["field"]
    channels.append(field_matrix)

    own_position_x = game_state["self"][3][0]
    own_position_y = game_state["self"][3][1]
    own_position = np.zeros((field_matrix.shape[0], field_matrix.shape[1]))
    own_position[own_position_x][own_position_y] = 1
    channels.append(own_position)

    others_positions = np.zeros((field_matrix.shape[0], field_matrix.shape[1]))
    for agent in game_state["others"]:
        other_position_x = agent[3][0]
        other_position_y = agent[3][1]
        others_positions[other_position_x][other_position_y] = 1
    channels.append(others_positions)

    positions_danger = np.zeros((field_matrix.shape[0], field_matrix.shape[1]))
    for bomb in game_state["bombs"]:
        time_passed = bomb[1]
        time_needed_to_explode = BOMB_TIMER
        bomb_x = bomb[0][0]
        bomb_y = bomb[0][1]
        # TODO: je weiter weg und je höher der timer noch mit einbeziehen
        # set horizontally
        for number in range(BOMB_POWER):
            try:
                positions_danger[bomb_x][bomb_y-number] = time_passed/time_needed_to_explode
                positions_danger[bomb_x][bomb_y+number] = time_passed/time_needed_to_explode
            except IndexError:
                print("Out of playing field")

        # set vertically
        for number in range(BOMB_POWER):
            try:
                positions_danger[bomb_x-number][bomb_y] = time_passed/time_needed_to_explode
                positions_danger[bomb_x+number][bomb_y] = time_passed/time_needed_to_explode
            except IndexError:
                print("Out of playing field")
    print("DANGER")
    print(positions_danger)
    channels.append(positions_danger)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
