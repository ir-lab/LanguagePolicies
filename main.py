# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals

from utils.tf_util import limitGPUMemory, trainOnCPU
from model_src.model import PolicyTranslationModel
from utils.network import Network
import tensorflow as tf
import hashids
import time
import glob
import pickle
import numpy as np
import shutil
import os.path
import sys

# Location of the training data
TRAIN_DATA      = "../GDrive/train.tfrecord"
# Location of the validation data
VALIDATION_DATA = "../GDrive/validate.tfrecord"
# Location of the GloVe word embeddings
GLOVE_PATH      = "../GDrive/glove.6B.50d.txt"
# Learning rate for the adam optimizer
LEARNING_RATE   = 0.0001
# Weight for the attention loss
WEIGHT_ATTN     = 1.0
# Weight for the motion primitive weight loss
WEIGHT_W        = 50.0
# Weight for the trajectroy generation loss
WEIGHT_TRJ      = 5.0
# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 1.0
# Number of epochs to train
TRAIN_EPOCHS    = 200

trainOnCPU()

hid             = hashids.Hashids()
LOGNAME         = hid.encode(int(time.time() * 1000000))

class DatasetRSS():
    def __init__(self, record_path):
        self.feature_descriptor = {
            "language/tokens":          tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "image/features":           tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "robot/trajectory/in":      tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "robot/trajectory/out":     tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "robot/trajectory/length":  tf.io.FixedLenFeature([], tf.int64),
            "robot/trajectory/dt":      tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "aux/onehot_class":         tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "dmp/w":                    tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "robot/trajectory/phase":   tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "robot/trajectory/padding": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }

        self.ds = tf.data.TFRecordDataset(record_path)
        self.ds = self.ds.map(self._parseFeatureDescriptor)
        self.ds = self.ds.shuffle(buffer_size=500)
        self.ds = self.ds.batch(16, drop_remainder=True)

    def _parseFeatureDescriptor(self, proto):
        features   = tf.io.parse_single_example(proto, self.feature_descriptor)
        language   = features["language/tokens"]
        
        img_ftr    = features["image/features"]
        img_ftr    = tf.keras.backend.reshape(img_ftr, shape=[-1, 5])

        state      = features["robot/trajectory/in"]
        state      = tf.keras.backend.reshape(state, shape=[-1, 7])

        trajectory = features["robot/trajectory/out"]
        trajectory = tf.keras.backend.reshape(trajectory, shape=[-1, 7])

        dt         = tf.cast(features["robot/trajectory/length"], dtype=tf.float32) / 350.0

        onehot     = features["aux/onehot_class"]

        weights    = features["dmp/w"]
        weights    = tf.keras.backend.reshape(weights, shape=[7,-1])

        phase      = tf.cast(features["robot/trajectory/phase"],   dtype=tf.float32)
        loss_atn   = tf.cast(features["robot/trajectory/padding"], dtype=tf.float32)

        return ((language, img_ftr, state), (trajectory, onehot, dt, weights, phase, loss_atn))

def setupModel():
    print("  --> Running with default settings")
    model   = PolicyTranslationModel(od_path="", glove_path=GLOVE_PATH)
    network = Network(model, logname=LOGNAME, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS)
    network.setDatasets(train=DatasetRSS(TRAIN_DATA), validate=DatasetRSS(VALIDATION_DATA))
    network.train(epochs=TRAIN_EPOCHS)
    return network

network = setupModel()
# model.summary()
