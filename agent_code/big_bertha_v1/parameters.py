# Architecture
STATE_SHAPE = 324
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
