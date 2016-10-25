#!/usr/bin/env python3

import os
import sys
import argparse

import numpy as np
import tensorflow as tf

# import plot
from data import Data
from vae import VAE

LOG_DIR = "./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"

def load(args):
    v = VAE(meta_graph=args.graph_file)
    print("Loaded!")
    return v

def train(args):
    hp = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout_prob,
        "lambda_l2_reg": args.lambda_,
        "nonlinearity": tf.nn.elu
    }

    data = Data(args.data_file)

    arch = [ data.sample_size ] + \
           [ args.layers_width ] * args.num_layers + \
           [ args.latent_dim ]
    # (and symmetrically back out again)

    v = VAE(arch, hp, log_dir=LOG_DIR)
    v.train(data, max_steps=args.num_steps, max_epochs=args.num_epochs,
            cross_validate=False, verbose=True, save=True,
            outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
            plot_latent_over_time=False)
    print("Trained!")
    return v

if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    parser = argparse.ArgumentParser(description='Variational Autoencoder.')
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    subparsers.required = True

    train_parser = subparsers.add_parser('train',
                                         help='Train the VAE with data.')
    train_parser.add_argument('--num_epochs', type=int, default=np.inf,
                              help='Maximum number of epochs.')
    train_parser.add_argument('-n', '--num_steps', type=int, default=2000,
                              help='Maximum number of steps.')
    train_parser.add_argument('-b', '--batch_size', type=int, default=128,
                              help='Batch size to use for training.')
    train_parser.add_argument('-r', '--learning_rate', type=float, default=5e-4,
                              help='Learning rate for optimizer.')
    train_parser.add_argument('-p', '--dropout_prob', type=float, default=0.9,
                              help='Keep probability for dropout in training.')
    train_parser.add_argument('--lambda', type=float, default=1e-5,
                              dest='lambda_', metavar='LAMBDA',
                              help='Lambda in L2 regularization.')
    train_parser.add_argument('-l', '--num_layers', type=int, default=2,
                              help='Number of hidden layers in each network.')
    train_parser.add_argument('-w', '--layers_width', type=int, default=500,
                              help='Layer width (constant).')
    train_parser.add_argument('-L', '--latent_dim', type=int, default=2,
                              help='Number of latent variable dimensions.')
    train_parser.add_argument('data_file',
                              help='Pickled data for training.')
    train_parser.set_defaults(func=train)

    load_parser = subparsers.add_parser('load', help='Load a pre-trained VAE.')
    load_parser.add_argument('graph_file', help='Stored computational graph.')
    load_parser.set_defaults(func=load)

    args = parser.parse_args()
    model = args.func(args)
