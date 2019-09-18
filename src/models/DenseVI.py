# Trying to replicate this:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/dense_variational_v2.py
# Understanding this will help me
# - build other kinds of layers, and perhaps
# - learn nuts and bolts of edward2


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd 
import numpy as np
import util

class DenseVI(tf.keras.layers.Layer):
    def __init__(self, units,
                 num_obs=None,
                 make_prior=util.make_default_dense_prior,
                 make_guide=util.make_default_dense_guide,
                 use_bias=True, activation=None, input_shape=None):

        super(DenseVI, self).__init__()

        assert units > 0

        self.units = units
        self.make_prior = make_prior
        self.make_guide = make_guide
        self.activation = activation
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.num_obs = num_obs


    def build(self, input_shape):
        pass


    def call(self):
        pass

