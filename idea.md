# IDEA

Use tensorflow.

- keep track of KL
    - KL = log p(theta) - log q(theta)
    - `tf.keras.layers.Dense` has a property `losses` which keeps track of
      regularizing terms values. Similarly, KL can be stored as `losses`.
    - See [custom tf2 layers][1], [Dense layer implementation in tf2][2]
- return loglike
- loss function should be `mean_neg_elbo` == `-ELBO / N`


|     x           | Dense | Dense variational |
| ------------ | --- | --------------- |
| regularization | l1/l2 penalty | kl -> log p - log q|
| activation     | same | same |
| others | | make sure to get samples before any computations|
| returns | Dense(x) | sample from Dense(x) |
| losses | l1/l2 penalty | kl |
| loss function | MSE, etc. | `-ELBO / N = -(ll - kl) / N` |

[1]: https://www.tensorflow.org/tutorials/eager/custom_layers
[2]: https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/layers/core.py#L968-L996


