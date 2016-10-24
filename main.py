import os
import sys

import numpy as np
import tensorflow as tf

# import plot
from data import Data
import vae


ARCHITECTURE = [ None, # input length, leave it like this
                 500, 500, # intermediate encoding
                 2 ] # latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 2000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"

def main(to_reload=None):
    data = Data('data/sentiment_set_norm.pickle')
    ARCHITECTURE[0] = data.sample_size

    if to_reload: # restore
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else: # train
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(data, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS,
                cross_validate=False, verbose=True, save=True,
                outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=False)
        print("Trained!")

    # all_plots(v, mnist)


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()
