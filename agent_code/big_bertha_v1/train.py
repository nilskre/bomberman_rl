import glob
import os
from datetime import datetime
from typing import List

import numpy as np

from agent_code.big_bertha_v1.experience_buffer import ExperienceBuffer
from agent_code.big_bertha_v1.features import state_to_features
from agent_code.big_bertha_v1.parameters import (ACTIONS, BATCH_SIZE,
                                                 EPSILON_DECAY, EPSILON_END,
                                                 EPSILON_START,
                                                 EXPERIENCE_BUFFER_SIZE_MIN,
                                                 GAMMA, REWARDS,
                                                 TRAINING_ROUNDS,
                                                 UPDATE_PREDICT_MODEL)


def setup_training(self):
    self.experience_buffer = ExperienceBuffer()
    self.episodes = 0
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

    update_q_values(self)
    self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    update_q_values(self)
    self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END

    self.episodes += 1
    if self.episodes == UPDATE_PREDICT_MODEL:
        self.predict_model.set_weights(self.model.get_weights())
        self.episodes = 0

    if last_game_state["round"] == TRAINING_ROUNDS:
        for filename in glob.glob("models/*_recent"):
            os.rename(filename, filename[:-7])
        current_time = datetime.now().strftime("%H_%M_%S")
        self.model.save("models/{}_recent".format(current_time))


def reward_from_events(self, occurred_events: List[str]) -> int:
    reward_sum = 0
    for event in occurred_events:
        reward_sum += REWARDS[event]
    return reward_sum


def update_q_values(self):
    if self.experience_buffer.size < EXPERIENCE_BUFFER_SIZE_MIN:
        return

    states, actions, rewards, new_states = self.experience_buffer.sample()

    qs_current = self.model.predict(states)
    qs_next_actual = self.predict_model.predict(new_states)
    qs_next_train = self.model.predict(new_states)
    qs_target = qs_current.copy()

    batch_index = np.arange(BATCH_SIZE, dtype=np.int8)

    qs_target[batch_index, actions] = rewards + GAMMA * qs_next_actual[batch_index, np.argmax(qs_next_train, axis=1)]
    _ = self.model.fit(states, qs_target, verbose=0)
