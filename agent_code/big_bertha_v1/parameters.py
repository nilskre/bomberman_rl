import events
import tensorflow as tf

# Architecture
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STATE_SHAPE = 1_445
NUMBER_OF_ACTIONS = 6
DENSE_LAYER_DIMS = 128

# Hyperparameters
ACTIVATION_FUNCTION = tf.keras.activations.elu  # alpha=1.0
LOSS_FUNCTION = tf.keras.losses.Huber()  # delta=1.0
LEARNING_RATE = 0.01

EXPERIENCE_BUFFER_SIZE_MIN = 8_192  # Batch size should be <1% of the total experience buffer size
EXPERIENCE_BUFFER_SIZE_MAX = 65_536
BATCH_SIZE = 64

TRAINING_ROUNDS = 1000

GAMMA = 0.95
EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 0.996  # Diminishing Epsilon-Greedy

UPDATE_TARGET_MODEL = 5
UPDATE_TENSORBOARD_EVERY = 5

# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Rewards
# TODO: Add custom events for reward shaping
REWARDS = {
    events.MOVED_LEFT: -0.01,
    events.MOVED_RIGHT: -0.01,
    events.MOVED_UP: -0.01,
    events.MOVED_DOWN: -0.01,
    events.WAITED: -0.05,
    events.INVALID_ACTION: -0.05,
    events.BOMB_DROPPED: 0.15,
    events.BOMB_EXPLODED: 0,
    events.CRATE_DESTROYED: 0.5,
    events.COIN_FOUND: 0.15,
    events.COIN_COLLECTED: 0.2,
    events.KILLED_OPPONENT: 1,
    events.KILLED_SELF: -1,
    events.GOT_KILLED: -1,
    events.OPPONENT_ELIMINATED: 0.05,
    events.SURVIVED_ROUND: 0.05,
}
