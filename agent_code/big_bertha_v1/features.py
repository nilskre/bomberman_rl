import numpy as np

from settings import BOMB_POWER, BOMB_TIMER, ROWS, COLS


class Features:

    def get_field_state(self, game_state: dict) -> np.array:
        return game_state["field"]

    def get_own_position(self, game_state: dict) -> np.array:
        own_position_x = game_state["self"][3][0]
        own_position_y = game_state["self"][3][1]
        own_position = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
        own_position[own_position_x][own_position_y] = 1
        return own_position

    def get_others_positions(self, game_state: dict) -> np.array:
        others_positions = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
        for agent in game_state["others"]:
            other_position_x = agent[3][0]
            other_position_y = agent[3][1]
            others_positions[other_position_x][other_position_y] = 1
        return others_positions

    def get_position_danger(self, game_state: dict) -> np.array:
        positions_danger = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
        for bomb in game_state["bombs"]:
            time_passed = BOMB_TIMER - bomb[1]
            time_needed_to_explode = BOMB_TIMER
            bomb_x = bomb[0][0]
            bomb_y = bomb[0][1]
            # set horizontally
            for number in range(1,BOMB_POWER):
                if bomb_y+number < ROWS:
                    positions_danger[bomb_x][bomb_y+number] = self._calculate_danger(time_passed, time_needed_to_explode, number)
                if bomb_y-number >= 0:
                    positions_danger[bomb_x][bomb_y-number] = self._calculate_danger(time_passed, time_needed_to_explode, number)
            # set vertically
            for number in range(1, BOMB_POWER):
                if bomb_x+number < COLS:
                    positions_danger[bomb_x+number][bomb_y] = self._calculate_danger(time_passed, time_needed_to_explode, number)
                if bomb_x-number >= 0:
                    positions_danger[bomb_x-number][bomb_y] = self._calculate_danger(time_passed, time_needed_to_explode, number)
            # set center
            positions_danger[bomb_x][bomb_y] = time_passed/time_needed_to_explode
        return positions_danger

    def _calculate_danger(self, time_passed, time_needed_to_explode, distance):
        return np.round((time_passed/time_needed_to_explode) / np.sqrt(distance), 2)

    def get_position_desirability(self, game_state: dict) -> np.array:
        coin_positions = np.zeros((game_state["field"].shape[0], game_state["field"].shape[1]))
        for coin in game_state["coins"]:
            coin_x = coin[0]
            coin_y = coin[1]
            coin_positions[coin_x][coin_y] = 1
        return coin_positions
