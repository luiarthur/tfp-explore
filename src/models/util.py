import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd 
import numpy as np

DistLambda = tfp.layers.DistributionLambda


def smooth_relu(x, c=np.log(np.expm1(1.)), eps=1e-5):
    """
    Exponentiating real values so that the resulting quantity is positive can
    lead to numerically unstable derivatives. Relu on real values lead to
    non-continuous derivaties. A compromise is to use a smoothed relu
    (softplus). The shift using +c is not essential but centers the result so
    that smooth_relu(0.) == 1. eps>0 helps to ensure that the result is always
    positive.
    """
    return tf.nn.softplus(c + x) + eps


# Define a generic negative loglikelihood function
# for computing a loss function
def neg_loglike(y, rv_y):
    # return -rv_y.log_prob(y)  # I think these two are equivalent
    return -tf.reduce_mean(rv_y.log_prob(y))


def make_default_dense_prior(kernel_size, bias_size=0, dtype=None,
                             prior_mean=0, prior_sd=100):
    n = kernel_size + bias_size
    mu = [prior_mean] * n
    return tf.keras.Sequential([
        # tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda _: tfd.Independent(
            tfd.Normal(loc=mu, scale=prior_sd),
            reinterpreted_batch_ndims=1))
    ])


def make_default_dense_guide(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        # Construct some trainable variational parameters
        # of the appropriate data type.
        # `VariableLayer(2*n)(None)` returns a length 2xn 
        # vector of zeros, which are trainable.
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        # Create a distribution lambda
        tfp.layers.DistributionLambda(lambda var_params:
            # These should be independent RVs
            tfd.Independent(
                # Independent Normals of the correct dimensions.
                # There should be as many variational distributions
                # as parameters.
                tfd.Normal(loc=var_params[:n],
                           scale=smooth_relu(var_params[n:])),
                reinterpreted_batch_ndims=1
            )
        ),
    ])
