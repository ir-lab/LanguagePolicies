# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf

class TopDownAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwarg):
        super(TopDownAttention, self).__init__(name="attention", **kwarg)
        self.units   = units

    def build(self, input_shape):
        self.w1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.units, activation=tf.keras.activations.tanh))
        self.w2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.units, activation=tf.keras.activations.sigmoid))
        self.wt = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear, use_bias=False))

    def call(self, inputs, training=None):
        language = inputs[0]
        features = inputs[1]
        k        = tf.shape(features)[1]

        language = tf.tile(tf.expand_dims(language, 1), [1, k, 1])            # bxkxm
        att_in   = tf.keras.backend.concatenate((language, features), axis=2) # bxkx(m+n)
        
        y_1 = self.w1(att_in)
        y_2 = self.w2(att_in)
        y   = tf.math.multiply(y_1, y_2)
        a   = self.wt(y)
        a   = tf.squeeze(a, axis=2)

        return tf.nn.softmax(a)

    def get_config(self):
        config = super(TopDownAttention, self).get_config()
        config.update({'units': self.units})
        return config