import events

# Architecture
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STATE_SHAPE = 405
NUMBER_OF_ACTIONS = 6
DENSE_LAYER_DIMS = 128

# Hyperparameters
ACTIVATION_FUNCTION = 'relu'
LOSS_FUNCTION = 'mse'
LEARNING_RATE = 0.0005
EXPERIENCE_BUFFER_SIZE = 8192
BATCH_SIZE = 64  # Batch size should be <1% of the total experience buffer size

GAMMA = 0.99
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 0.996  # Diminishing Epsilon-Greedy

# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Rewards
# TODO: Add custom events for reward shaping
REWARDS = {
    events.MOVED_LEFT: -1,
    events.MOVED_RIGHT: -1,
    events.MOVED_UP: -1,
    events.MOVED_DOWN: -1,
    events.WAITED: -5,
    events.INVALID_ACTION: -2,
    events.BOMB_DROPPED: 30,
    events.BOMB_EXPLODED: 30,
    events.CRATE_DESTROYED: 30,
    events.COIN_FOUND: 5,
    events.COIN_COLLECTED: 20,
    events.KILLED_OPPONENT: 100,
    events.KILLED_SELF: -350,
    events.GOT_KILLED: -300,
    events.OPPONENT_ELIMINATED: 5,
    events.SURVIVED_ROUND: 10,
}
