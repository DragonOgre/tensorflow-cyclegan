import random
import os
import numpy as np
import scipy.misc
import argparse
import tensorflow as tf
import utils
from cyclegan import generator
import json
from flask import Flask

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-cd", "--check-dir", dest="checkpoint_dir", help="Directory where checkpoint file will be stored", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("-c", "--check", help="Name of the checkpoint file", type=str, default=CHECKPOINT_FILE)

    # Parse arguments
    args = parser.parse_args()

    # Raw print arguments
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))

    return args


def get_model(sess, graph, checkpoint_dir):
    with graph.as_default():
        real_X = tf.placeholder(tf.float32, [None, 256, 256, 3])
        real_Y = tf.placeholder(tf.float32, [None, 256, 256, 3])

        # genG(X) => Y            - fake_B
        genG = generator(real_X, name="generatorG")
        # genF(Y) => X            - fake_A
        genF = generator(real_Y, name="generatorF")

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)

    def predictG(image_data):
        img = (image_data / 127.5) - 1.
        res = sess.run(genG, feed_dict={real_X: [img]})
        res = utils.inverse_transform(res[0])
        # scipy.misc.imsave('resultG.jpg', res)
        return res

    def predictF(image_data):
        img = (image_data / 127.5) - 1.
        res = sess.run(genF, feed_dict={real_Y: [img]})
        res = utils.inverse_transform(res[0])
        # scipy.misc.imsave('resultF.jpg', res)
        return res

    return predictG, predictF


def get_predictors():
    # args = parseArguments()
    # CHECKPOINT_FILE = args.check
    # CHECKPOINT_DIR = args.checkpoint_dir
    checkpoint_dirs = [name for name in os.listdir("checkpoints/") if os.path.isdir(os.path.join('checkpoints/', name)) and name.startswith('checkpoint-')]
    print('Checkpoint dirs: {}'.format(checkpoint_dirs))
    checkpoints = {}
    predictors = {}
    for dir_name in checkpoint_dirs:
        name = dir_name.split('-')[1]
        checkpoints[name] = dir_name
    
    for checkpoint_name in checkpoints:
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        predictG, predictF = get_model(sess, graph, os.path.join('checkpoints/', checkpoints[checkpoint_name]))
        predictors['{}/{}'.format(checkpoint_name, checkpoint_name.split('_')[0])] = predictG
        predictors['{}/{}'.format(checkpoint_name, checkpoint_name.split('_')[1])] = predictF

    return predictors
