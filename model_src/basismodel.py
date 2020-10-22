# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np

class BasisModel(tf.keras.layers.Layer):
    def __init__(self, dimensions, nfunctions, scale, **kwarg):
        super(BasisModel, self).__init__(name="attention", **kwarg)
        self._degree = nfunctions
        self.scale   = scale

    def build(self, input_shape):
        self.centers = np.linspace(0.0, 1.01, self._degree, dtype = np.float32)
        self.centers = tf.convert_to_tensor(self.centers)

    @tf.function
    def call(self, inputs, training=None):
        weights     = tf.transpose(inputs[0], perm=[0,2,1])
        weights_std = inputs[1]
        positions   = inputs[2]
        basis_funcs = self.compute_basis_values(positions)
        result      = tf.linalg.matmul(basis_funcs, weights)

        return result, tf.zeros_like(result)

    def get_config(self):
        config = super(TopDownAttention, self).get_config()
        config.update({'units': self.units})
        return config

    def compute_basis_values(self, x):
        centers = tf.tile(tf.expand_dims(self.centers,0), [tf.shape(x)[1], 1])
        x       = tf.expand_dims(x, 2)
        funcs   = tf.exp(-( tf.math.pow((x - centers), 2) / (2.0 * self.scale) ))
        return funcs
