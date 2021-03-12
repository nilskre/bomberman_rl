from agent_code.big_bertha_v1.parameters import (ACTIVATION_FUNCTION,
                                                 DENSE_LAYER_DIMS,
                                                 LEARNING_RATE, LOSS_FUNCTION,
                                                 NUMBER_OF_ACTIONS,
                                                 STATE_SHAPE)
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.init_ops_v2 import variance_scaling_initializer


# TODO: Normalize last layer, maybe softmax?
def compile_deep_q_network():
    he = variance_scaling_initializer()
    model = Sequential([
        Dense(DENSE_LAYER_DIMS, input_shape=(STATE_SHAPE,), kernel_initializer=he, activation=ACTIVATION_FUNCTION),
        Dense(DENSE_LAYER_DIMS, kernel_initializer=he, activation=ACTIVATION_FUNCTION),
        Dense(NUMBER_OF_ACTIONS, kernel_initializer=he, activation=softmax),
    ])
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS_FUNCTION)
    return model
