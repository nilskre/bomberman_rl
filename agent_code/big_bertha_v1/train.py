import glob
import os
from datetime import datetime
from typing import List

import numpy as np
from agent_code.big_bertha_v1.experience_buffer import ExperienceBuffer
from agent_code.big_bertha_v1.features import state_to_features
from agent_code.big_bertha_v1.modifiedtensorboard import ModifiedTensorBoard
from agent_code.big_bertha_v1.parameters import (ACTIONS, BATCH_SIZE,
                                                 EPSILON_DECAY, EPSILON_END,
                                                 EPSILON_START,
                                                 EXPERIENCE_BUFFER_SIZE_MIN,
                                                 GAMMA, REWARDS,
                                                 TRAINING_ROUNDS,
                                                 UPDATE_TARGET_MODEL,
                                                 UPDATE_TENSORBOARD_EVERY)


def setup_training(self):
    self.experience_buffer = ExperienceBuffer()
    self.epsilon = EPSILON_START

    self.episodes_past = 0

    self.episode_rewards = []
    self.episode_reward = 0

    current_time = datetime.now().strftime("%H_%M_%S")
    self.tensorboard = ModifiedTensorBoard(log_dir="tensorboard_logs/{}".format(current_time))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    reward = reward_from_events(self, events)

    self.experience_buffer.remember(
        state_to_features(old_game_state),
        ACTIONS.index(self_action),
        reward,
        state_to_features(new_game_state)
    )

    update_q_values(self)
    self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END
    self.episode_reward += reward


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    update_q_values(self)
    self.epsilon = self.epsilon * EPSILON_DECAY if self.epsilon > EPSILON_END else EPSILON_END
    self.episode_reward += reward_from_events(self, events)

    self.episode_rewards.append(self.episode_reward)
    self.episode_reward = 0

    if not self.episodes_past % UPDATE_TENSORBOARD_EVERY:
        self.tensorboard.update_stats(average_reward=sum(self.episode_rewards[-UPDATE_TENSORBOARD_EVERY:])/UPDATE_TENSORBOARD_EVERY, epsilon=self.epsilon)
    self.tensorboard.step += 1

    self.episodes_past += 1
    if self.episodes_past == UPDATE_TARGET_MODEL:
        self.target_model.set_weights(self.online_model.get_weights())
        self.episodes_past = 0

    if last_game_state["round"] == TRAINING_ROUNDS:
        for filename in glob.glob("models/*_recent"):
            os.rename(filename, filename[:-7])
        current_time = datetime.now().strftime("%H_%M_%S")
        self.online_model.save("models/{}_recent".format(current_time))


def reward_from_events(self, occurred_events: List[str]) -> int:
    reward_sum = 0
    for event in occurred_events:
        reward_sum += REWARDS[event]
    return reward_sum


def update_q_values(self):
    if not self.experience_buffer.filled and self.experience_buffer.memory_index < EXPERIENCE_BUFFER_SIZE_MIN:
        return

    states, actions, rewards, new_states = self.experience_buffer.sample()

    qs_next_actual = self.target_model.predict(new_states)
    qs_next_train = self.online_model.predict(new_states)
    qs_target = self.online_model.predict(states)

    batch_index = np.arange(BATCH_SIZE, dtype=np.int8)

    qs_target[batch_index, actions] = rewards + GAMMA * qs_next_actual[batch_index, np.argmax(qs_next_train, axis=1)]
    _ = self.online_model.fit(states, qs_target, verbose=0)
