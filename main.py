import os
import sys
import argparse

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

LOG_DIR = "./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"

def main(hp, max_iter, max_epochs, filename=None, to_reload=None):

    if to_reload: # restore
        v = vae.VAE(ARCHITECTURE, hp, meta_graph=to_reload)
        print("Loaded!")

    else: # train
        data = Data(filename)
        ARCHITECTURE[0] = data.sample_size
        v = vae.VAE(ARCHITECTURE, hp, log_dir=LOG_DIR)
        v.train(data, max_iter=max_iter, max_epochs=max_epochs,
                cross_validate=False, verbose=True, save=True,
                outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=False)
        print("Trained!")

    # all_plots(v, mnist)

    return v

if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    parser = argparse.ArgumentParser(description='Variational Autoencoder.')
    parser.add_argument('--num_epochs', type=int, default=np.inf,
                        help='Maximum number of epochs.')
    parser.add_argument('-n', '--num_iters', type=int, default=2000,
                        help='Maximum number of iterations.')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Batch size to use for training.')
    parser.add_argument('-r', '--learning_rate', type=float, default=5e-4,
                        help='Learning rate for optimizer.')
    parser.add_argument('-p', '--dropout_prob', type=float, default=0.9,
                        help='Keep probability for dropout in training.')
    parser.add_argument('--lambda', type=float, default=1e-5, dest='lambda_',
                        help='Lambda in L2 regularization.')
    parser.add_argument('-i', '--input', help='Use pre-trained graph in file.')
    parser.add_argument('-t', '--train', dest='datafile',
                        help='Use pickled datafile for training.')
    args = parser.parse_args()

    hp = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout_prob,
        "lambda_l2_reg": args.lambda_,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }

    if args.datafile or args.input:
        model = main(hp, args.num_iters, args.num_epochs,
                     to_reload=args.input, filename=args.datafile)
    else:
        print('No action specified. See --help.')
