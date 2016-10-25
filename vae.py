from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf

from layers import Dense
# import plot
from utils import composeAll, print_


class VAE:
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 save_graph_def=True, log_dir="./log"):
        """(Re)build a symmetric VAE model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs,
               10-D latents, & end-to-end architecture
               [1000, 500, 250, 10, 250, 500, 1000]

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if not meta_graph: # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sesh.run(tf.initialize_all_variables())

        else: # restore saved model
            model_datetime, model_name = \
                os.path.basename(meta_graph).split("_vae_")
            self.datetime = "{}_reloaded".format(model_datetime)
            *model_architecture, _ = re.split("_|-", model_name)
            self.architecture = [ int(n) for n in model_architecture ]

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(VAE.RESTORE_KEY)

        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_out, self.cost, self.global_step, self.train_op) = handles

        if save_graph_def: # tensorboard
            self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=(None, self.architecture[0]),
                              name="x_in")
        dropout = tf.placeholder_with_default(1., shape=(), name="dropout")

        # encoding / "recognition": q(z|x)
        encoding = [ Dense("encoding", hidden_size, dropout, self.nonlinearity)
                     # hidden layers reversed for function composition:
                     # outer -> inner
                     for hidden_size in reversed(self.architecture[1:-1]) ]
        h_encoded = composeAll(encoding)(x_in)

        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense("z_mean", self.architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma",
                            self.architecture[-1], dropout)(h_encoded)

        # kingma & welling: only 1 draw necessary as long as
        # minibatch large enough (>100)
        self.z = tf.identity(self.sampleGaussian(z_mean, z_log_sigma), name="z")
        # this is also the entry point for latent space exploration

        # decoding / "generative": p(x|z)
        decoding = [ Dense("decoding", hidden_size, dropout, self.nonlinearity)
                     for hidden_size in self.architecture[1:-1] ]
                     # assumes symmetry
        # final reconstruction: restore original dims
        # prepend as outermost function
        decoding.insert(0, Dense("x_decoding", self.architecture[0], dropout))
        outer_layer = composeAll(decoding)(self.z)
        x_out = tf.nn.sigmoid(outer_layer, name="x_out")

        # reconstruction loss: mismatch b/w x & x_out
        # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
        rec_loss = VAE.crossEntropy(outer_layer, x_in)

        # Kullback-Leibler divergence: mismatch b/w approximate
        # vs. imposed/true posterior
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)

        with tf.name_scope("l2_regularization"):
            regularizers = [ tf.nn.l2_loss(var) for var in
                self.sesh.graph.get_collection("trainable_variables")
                if "weights" in var.name ]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            cost += l2_reg

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            # gradient clipping
            clipped = [ (tf.clip_by_value(grad, -5, 5), tvar)
                        for grad, tvar in grads_and_vars ]
            train_op = optimizer.apply_gradients(clipped,
                                                 global_step=global_step,
                                                 name="minimize_cost")

        return (x_in, dropout, z_mean, z_log_sigma, x_out,
                cost, global_step, train_op)

    def sampleGaussian(self, mu, log_sigma):
        """(Differentiably!) draw sample from Gaussian with given shape,
        subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

    @staticmethod
    def crossEntropy(obs, actual):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # assumes sigmoid has not been applied yet (more numerically stable)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(obs, actual)
            return tf.reduce_sum(cross_entropy, 1)

    @staticmethod
    def l1_loss(obs, actual):
        """L1 loss (a.k.a. LAD), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l1_loss"):
            return tf.reduce_sum(tf.abs(obs - actual) , 1)

    @staticmethod
    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p),
        per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 -
                                        tf.exp(2 * log_sigma), 1)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma],
                             feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input
        space; a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        # coerce to np.array, if zs is tensor
        zs = self.sesh.run(zs) if hasattr(zs, "eval") else zs \
             or np.random.normal(size=(1, self.architecture[-1]))
             # if zs None, defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_out, feed_dict={self.z: zs})

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, data, max_steps=np.inf, max_epochs=np.inf,
              cross_validate=True, verbose=True, save=True, outdir="./out",
              plots_outdir="./png", plot_latent_over_time=False):
        if save:
            saver = tf.train.Saver(tf.all_variables())

        try:
            err_train = 0
            last_printed_i = 0
            last_printed_err = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            # if plot_latent_over_time: # plot latent space over log_BASE time
            #     BASE = 2
            #     INCREMENT = 0.5
            #     pow_ = 0

            while True:
                x, _ = data.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.dropout_: self.dropout}
                fetches = [ self.x_out, self.cost,
                            self.global_step, self.train_op ]
                x_out, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                # if plot_latent_over_time:
                #     while int(round(BASE**pow_)) == i:
                #         plot.exploreLatent(self, nx=30, ny=30, ppf=True,
                #                            outdir=plots_outdir,
                #                            name="explore_ppf30_{}".format(pow_))
                #
                #         names = ("train", "validation", "test")
                #         datasets = (X.train, X.validation, X.test)
                #         for name, dataset in zip(names, datasets):
                #             plot.plotInLatent(self, dataset.images,
                #                               dataset.labels, range_=(-6, 6),
                #                               title=name, outdir=plots_outdir,
                #                               name="{}_{}".format(name, pow_))
                #
                #         print("{}^{} = {}".format(BASE, pow_, i))
                #         pow_ += INCREMENT

                # if i%2000 == 0 and verbose:# and i >= 10000:
                    # visualize `n` examples of current minibatch inputs +
                    # reconstructions
                    # plot.plotSubset(self, x, x_out, n=10,
                    #                 name="train", outdir=plots_outdir)

                    # if cross_validate:
                    #     x, _ = X.validation.next_batch(self.batch_size)
                    #     feed_dict = {self.x_in: x}
                    #     fetches = [self.x_out, self.cost]
                    #     x_out, cost = self.sesh.run(fetches,feed_dict)
                    #
                    #     print("round {} --> CV cost: ".format(i), cost)
                    #     plot.plotSubset(self, x, x_out, n=10,
                    #                     name="cv", outdir=plots_outdir)

                if i >= max_steps or data.epochs_completed >= max_epochs:
                    avg = (err_train - last_printed_err) / (i - last_printed_i)
                    print("final avg cost (@ step {} = epoch {}): {}"\
                          .format(i, data.epochs_completed, avg))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir),
                            "{}_vae_{}".format(self.datetime,
                            "_".join(map(str, self.architecture))))
                        saver.save(self.sesh, outfile,
                                   global_step=self.global_step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        continue
                    break

                if i%1000 == 0 and verbose:
                    avg = (err_train - last_printed_err) / (i - last_printed_i)
                    print("step {} --> avg cost: {}".format(i, avg))
                    last_printed_err = err_train
                    last_printed_i = i

        except(KeyboardInterrupt):
            avg = (err_train - last_printed_err) / (i - last_printed_i)
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, data.epochs_completed, avg))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)
