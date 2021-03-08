from agent_code.big_bertha_v1.parameters import DENSE_LAYER_DIMS, NUMBER_OF_ACTIONS, STATE_SHAPE, ACTIVATION_FUNCTION, \
    LOSS_FUNCTION, LEARNING_RATE

from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def compile_deep_q_network():
    model = Sequential()([
        Dense(DENSE_LAYER_DIMS, input_shape=(STATE_SHAPE,)),
        Activation(ACTIVATION_FUNCTION),
        Dense(DENSE_LAYER_DIMS),
        Activation(ACTIVATION_FUNCTION),
        Dense(NUMBER_OF_ACTIONS)
    ])
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS_FUNCTION)
    return model
