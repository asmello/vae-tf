import tensorflow as tf


class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity, initialization="uniform"):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity
        self.initialization = initialization

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value,
                        self.size, uniform=(self.initialization == "uniform"))
                    self.w = tf.nn.dropout(self.w, self.dropout)

    @staticmethod
    def wbVars(fan_in: int, fan_out: int, uniform=True):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        if uniform:
            r = tf.cast((6. / (fan_in + fan_out))**.5, tf.float32)
            initial_w = tf.random_uniform([fan_in, fan_out], -r, r)
        else:
            stddev = tf.cast((3. / (fan_in+fan_out))**.5, tf.float32)
            initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))
