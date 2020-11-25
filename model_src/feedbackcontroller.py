# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
from model_src.basismodel import BasisModel

class FeedbackController(tf.keras.layers.Layer):
    def __init__(self, robot_state_size, rnn_state_size, dimensions, basis_functions, special, **kwargs):
        super(FeedbackController, self).__init__(**kwargs)
        self.robot_state_size = robot_state_size
        self.state_size       = rnn_state_size
        self.dims             = dimensions
        self.n_bfuncs         = basis_functions

    def build(self, input_shape):
        self.robot_gru      = tf.keras.layers.GRUCell(units=self.robot_state_size)

        self.weight_dense_1 = tf.keras.layers.Dense(units=self.dims * self.n_bfuncs, activation=tf.keras.activations.relu)
        self.weight_dense_2 = tf.keras.layers.Dense(units=self.dims * self.n_bfuncs, activation=tf.keras.activations.relu)
        self.weight_dense_3 = tf.keras.layers.Dense(units=self.dims * self.n_bfuncs, activation=tf.keras.activations.linear)

        self.phase_dense_1 = tf.keras.layers.Dense(units=int(self.robot_state_size / 2.0), activation=tf.keras.activations.relu)
        self.phase_dense_2 = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.hard_sigmoid)

        self.basismodel = BasisModel(dimensions=self.dims, nfunctions=self.n_bfuncs, scale=0.012)

    # @tf.function
    def call(self, inputs, states, constants=None, training=False, mask=None, **kwargs):
        # Get data ready
        in_robot       = inputs
        st_robot_last  = states[0]
        st_gru_last    = states[1]
        cn_features    = constants[0]
        cn_delta_t     = constants[1]
        
        # Robot GRU:
        in_robot           = tf.cond(tf.convert_to_tensor(training), lambda: st_robot_last, lambda: in_robot)
        e_robot, gru_state = self.robot_gru(inputs=in_robot, states=[st_gru_last])

        # Internal state:
        x = tf.keras.backend.concatenate((cn_features, e_robot), axis=1)

        # Use x to calcate the weights:
        weights = self.weight_dense_3(self.weight_dense_2(self.weight_dense_1(x)))
        weights = tf.keras.backend.reshape(weights, shape=[-1, self.dims, self.n_bfuncs])

        # Phase estimation, based on x:
        dt    = 1.0 / (500.0 * cn_delta_t) # Calculates the actual dt
        phase = self.phase_dense_2(self.phase_dense_1(x))
        phase = phase + dt

        # Apply basis model:
        action, _ = self.basismodel((weights, tf.zeros_like(weights), phase))
        action = tf.squeeze(action)

        # Rebuild the state:
        new_states = (action, gru_state[0])

        # Return results (and state)
        return (action, phase, weights), new_states