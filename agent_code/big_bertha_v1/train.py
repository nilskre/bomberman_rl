import glob
import os
from datetime import datetime
from typing import List

import numpy as np
from agent_code.big_bertha_v1.features import state_to_features
from agent_code.big_bertha_v1.experience_buffer import ExperienceBuffer
from agent_code.big_bertha_v1.parameters import (BATCH_SIZE, EPSILON_DECAY, EPSILON_END, EPSILON_START,
                                                 GAMMA, REWARDS, ACTIONS)


def setup_training(self):
    self.experience_buffer = ExperienceBuffer()
    self.epsilon = EPSILON_START


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    self.experience_buffer.remember(
        state_to_features(old_game_state),
        ACTIONS.index(self_action),
        reward_from_events(self, events),
        state_to_features(new_game_state)
    )

    if self.experience_buffer.filled or self.experience_buffer.memory_index >= BATCH_SIZE:
        states, actions, rewards, new_states = self.experience_buffer.sample()

        qs_current = self.model.predict(states)
        qs_next = self.model.predict(new_states)
        qs_target = qs_current.copy()

        batch_index = np.arange(BATCH_SIZE, dtype=np.int8)

        qs_target[batch_index, actions] = rewards + GAMMA * np.max(qs_next)
        _ = self.model.fit(states, qs_target, verbose=0)

        self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END


# Called at the end of each game or when the agent died to hand out final rewards.
# TODO: eventually update experience buffer
# TODO: actually the model should be saved after 500 games
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    for filename in glob.glob("models/*_recent"):
        os.rename(filename, filename[:-7])
    current_time = datetime.now().strftime("%H_%M_%S")
    self.model.save("models/{}_recent".format(current_time))


def reward_from_events(self, occurred_events: List[str]) -> int:
    reward_sum = 0
    for event in occurred_events:
        reward_sum += REWARDS[event]
    return reward_sum
