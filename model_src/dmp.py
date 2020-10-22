# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np

class DynamicMovementPrimitive(tf.keras.layers.Layer):
    def __init__(self, dimensions, nfunctions, ax, ay, by, tau, **kwargs):
        super(DynamicMovementPrimitive, self).__init__(**kwargs)
        self.ax         = ax
        self.dimensions = dimensions

        self.p_ay       = ay
        self.ay         = tf.ones(self.dimensions, dtype=tf.float32) * self.p_ay
        self.p_by       = by
        self.by         = self.ay / self.p_by
        self.tau        = tau
        # self.dt         = dt
        self.state_size = tf.TensorShape(dims=[4,self.dimensions]) 
        self.nbasis     = nfunctions
        self._generateBasiFunctions(self.nbasis)

    def _generateBasiFunctions(self, n_functions):
        des_c                = np.linspace(0.0, 1.0, n_functions)
        self.basis_locations = np.zeros(des_c.shape[0])
        for i in range(des_c.shape[0]):
            self.basis_locations[i] = np.exp(-self.ax * des_c[i])
        self.basis_variance  = np.ones(n_functions) * n_functions**1.75 / self.basis_locations / self.ax

        self.basis_locations = tf.convert_to_tensor(self.basis_locations, dtype=tf.float32)
        self.basis_variance  = tf.convert_to_tensor(self.basis_variance,  dtype=tf.float32)

    # @tf.function
    def call(self, inputs, states, constants=None, training=None, mask=None, **kwargs):
        current = inputs
        # weights = tf.transpose(constants[0], perm=[0, 2, 1])
        weights = constants[0]
        goal    = constants[1]
        start   = constants[2]
        dt      = constants[3]

        batch_size     = tf.shape(weights)[0]
        state          = states[0]
        y_term         = state[:,1,:]
        dy_term        = state[:,2,:]
        ddy_term       = state[:,3,:]
        error          = tf.keras.metrics.mean_squared_error(current, y_term)
        error_coupling = 1.0 / (1.0 + error)
        cs_x           = state[:,0,0] # Variable 1
        step           = state[:,0,1] # Variable 2 
        cs_x          += (-self.ax * tf.math.multiply(cs_x, error_coupling)) * self.tau * dt[:,0]
        psi            = tf.math.exp(-tf.tile(tf.expand_dims(self.basis_variance, 0), [batch_size, 1]) * 
                                      tf.math.pow(tf.tile(tf.expand_dims(cs_x, 1), [1, self.nbasis]) - 
                                                  tf.tile(tf.expand_dims(self.basis_locations, 0), [batch_size, 1]),
                                                 2.0)
                                    )

        front_term     = tf.tile(tf.expand_dims(cs_x, 1), [1, self.dimensions])
        front_term     = tf.math.multiply(front_term, (goal - start))
        # weights        = tf.transpose(weights, perm=[0,2,1])
        forcing_term   = tf.linalg.matmul(tf.expand_dims(psi,1), weights)
        forcing_term   = tf.squeeze(forcing_term, 1) / tf.tile(tf.expand_dims(tf.math.reduce_sum(psi, axis=1), 1), [1, self.dimensions])
        forcing_term   = tf.math.multiply(front_term, forcing_term)

        ddy_term       = (self.ay * (self.by * (goal - y_term) - dy_term / self.tau) + forcing_term) * self.tau
        error_coupling = tf.tile(tf.expand_dims(error_coupling, 1), [1, self.dimensions])
        dy_term       += tf.math.multiply(tf.math.multiply(error_coupling, ddy_term) * self.tau, tf.tile(dt, [1, self.dimensions]))
        y_term        += tf.math.multiply(tf.math.multiply(error_coupling, dy_term ), tf.tile(dt, [1, self.dimensions]))

        # Mask steps beyond the desired length described by dt
        y_term    = tf.where(tf.math.greater(tf.expand_dims(step, 1), 1.0/dt), state[:,1,:], y_term)

        # Rebuild State
        step     += 1.0
        cs_x      = tf.expand_dims(tf.expand_dims(cs_x,1),1)
        step      = tf.expand_dims(tf.expand_dims(step,1),1)
        variables = tf.keras.backend.concatenate((cs_x, step), axis=2)
        variables = tf.pad(variables, tf.constant([[0,0],[0,0],[0,self.dimensions-2]]), "CONSTANT")
        state     = tf.keras.backend.concatenate((variables, tf.expand_dims(y_term,1), tf.expand_dims(dy_term,1), tf.expand_dims(ddy_term,1)), axis=1)

        return y_term, state

    def get_config(self):
        config = super(TopDownAttention, self).get_config()
        config.update({"dimensions": self.dimensions, "nfunctions": self.nbasis, "ax": self.ax, "ay": self.p_ay, "by": self.p_by, "tau": self.tau})
        return config
