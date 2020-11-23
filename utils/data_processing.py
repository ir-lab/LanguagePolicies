# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np
import glob
import json
from PIL import Image, ImageDraw
import pickle
import re
import os
from intprim.gaussian_model import GaussianModel

def getDMPData(data, n_bfunc):
    basismodel   = GaussianModel(degree=n_bfunc, scale=0.012, observed_dof_names=("Base","Shoulder","Ellbow","Wrist1","Wrist2","Wrist3","Gripper"))
    domain       = np.linspace(0, 1, data.shape[0], dtype=np.float64)
    intprim_w    = basismodel.fit_basis_functions_linear_closed_form(domain, data)
    return intprim_w

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def limitGPUMemory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

class DataConverter():
    def __init__(self, dict_path, frcnn_path):
        self.dictionary  = self._loadDictionary(dict_path)
        self.regex       = re.compile('[^a-z ]')
        self.frcnn       = tf.saved_model.load(str(frcnn_path))
        self.frcnn       = self.frcnn.signatures['serving_default']

    def _loadDictionary(self, file):
        __dictionary = {}
        __dictionary[""] = 0 # Empty string
        fh = open(file, "r")
        for line in fh:
            if len(__dictionary) >= 300000:
                break
            tokens = line.strip().split(" ")
            __dictionary[tokens[0]] = len(__dictionary)
        fh.close()
        return __dictionary

    def createRecord(self, raw, out, min_samples=None, max_samples=None):
        print("Creating TFRecord")
        writer = tf.io.TFRecordWriter(out)
        files  = glob.glob(raw)
        
        if min_samples is not None:
            files = files[min_samples:]

        if max_samples is not None:
            files = files[:max_samples]

        for idx, filepath in enumerate(files):
            if idx % 100 == 0:
                print(" {:5d}/{:5d}".format(idx, len(files)))

            tf_example = self.createTfExample(*self.loadDemonstrationFromFile(filepath))
            writer.write(tf_example.SerializeToString())

        writer.close()
    
    def createNormalization(self, raw, out):
        if os.path.exists(out):
            print("Using existing normalization")
            self.normalization = pickle.load(open(out, mode="rb"))
            return

        print("Creating Normalization")
        normalization               = {}
        normalization["max_length"] = 0
        normalization["values"]     = None
        files                       = glob.glob(raw)

        for fn, file_handle in enumerate(files):
            if fn % 100 == 0:
                print("  {:5d}/{:5d}".format(fn, len(files)))
            data = json.load(open(file_handle))
            state = np.asarray(data["state/raw"], dtype=np.float32)

            normalization["max_length"] = max(normalization["max_length"], state.shape[0])
            if normalization["values"] is None: 
                normalization["values"] = np.zeros((4, state.shape[1]), dtype=np.float32)
                normalization["values"][0,:] = 1e10
                normalization["values"][1,:] = -1e10
            normalization["values"][0,:] = np.minimum(normalization["values"][0,:], np.min(state, axis=0))
            normalization["values"][1,:] = np.maximum(normalization["values"][1,:], np.max(state, axis=0))
            normalization["values"][2,:] += np.std(state, axis=0)
            normalization["values"][3,:] += np.var(state, axis=0)
        normalization["values"][2:3,:] /= len(files)
        pickle.dump(normalization, open(out, "wb"))
        self.normalization = normalization

    def createTfExample(self, tokens, features, trajectory_in, trajectory_out, trajectory_length, dt, onehot, dmp_w):
        phase = np.linspace(0.0, 1.0, trajectory_length)
        phase = np.pad(phase, [0, 350-trajectory_length], mode="edge")

        loss_atn                     = np.zeros((350), dtype=np.float32)
        loss_atn[:trajectory_length] = 1.0

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "language/tokens":          int64_list_feature(tokens),
            "image/features":           float_list_feature(features),
            "robot/trajectory/in":      float_list_feature(trajectory_in),
            "robot/trajectory/out":     float_list_feature(trajectory_out),
            "robot/trajectory/length":  int64_feature(trajectory_length),
            "robot/trajectory/dt":      float_list_feature([dt]),
            "aux/onehot_class":         int64_list_feature(onehot),
            "dmp/w":                    float_list_feature(dmp_w),
            "robot/trajectory/phase":   float_list_feature(phase.tolist()),
            "robot/trajectory/padding": float_list_feature(loss_atn.tolist())
        }))
        return tf_example
    
    def normalize(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise ArrayDimensionMismatch()
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value
    
    def tokenize(self, language):
        voice  = self.regex.sub("", language.strip().lower())
        tokens = []
        for w in voice.split(" "):
            idx = 0
            try:
                idx = self.dictionary[w]
            except:
                print("Unknown word: " + w)
            tokens.append(idx)
        return tokens
    
    def padSequence(self, sequence, length):
        if len(sequence.shape) != 2:
            raise ArrayDimensionMismatch()
        pad_size = length - sequence.shape[0]
        return np.pad(sequence, [[0, pad_size], [0, 0]], mode="edge")

    def interpolateTrajectory(self, trj, target):
        current_length = trj.shape[0]
        dimensions     = trj.shape[1]
        result         = np.zeros((target, trj.shape[1]), dtype=np.float32)
              
        for i in range(dimensions):
            result[:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[:,i])
        
        return result

    def gipperToSigmoid(self, trajectory):
        def sigmoid(x, mean, var):
            var = mean if mean - var < 0 else var
            a   = 5.0 / var
            b   = -(5.0 * mean) / var
            x   = x * a + b
            return 1.0/(1.0 + np.exp(-x))

        def fixGripper(trajectory):
            rng = len(trajectory)
            mu  = min([i for i in range(rng) if trajectory[i] > 0.5])
            g_n = [sigmoid(x, mu, 30) for x in range(rng)]
            return g_n
        
        if trajectory[0,6] < 0.5 and trajectory[-1,6] > 0.5:
            trajectory[:,6] = fixGripper(trajectory[:,6])
        return trajectory

    def loadDemonstrationFromFile(self, path):
        with open(path, "r") as fh:
            data = json.load(fh)
        image                       = data["image"]
        image                       = np.asarray(image, dtype=np.uint8)
        input_tensor                = tf.convert_to_tensor(image[:,:,::-1])
        input_tensor                = input_tensor[tf.newaxis,...]
        image_cnn                   = self.frcnn(input_tensor)
        data_dict                   = {}
        data_dict["fcnn/classes"]   = image_cnn["detection_classes"].numpy()[:,:6]
        data_dict["fcnn/scores"]    = image_cnn["detection_scores"].numpy()[:,:6]
        data_dict["fcnn/boxes"]     = image_cnn["detection_boxes"].numpy()[:,:6]

        scores                      = [0.0 if v < 0.5 else 1.0 for v in data_dict["fcnn/scores"][0][:6].astype(dtype=np.float32).tolist()]
        classes                     = [v * scores[k] for k, v in enumerate(data_dict["fcnn/classes"][0][:6].astype(dtype=np.int64).tolist())]
        boxes                       = data_dict["fcnn/boxes"][0][:6].astype(dtype=np.float32)
        features                    = np.concatenate((np.expand_dims(classes,1), boxes), axis=1).flatten().tolist()

        trajectory                  = np.take(np.asarray(data["state/raw"]), indices=[0,1,2,3,4,5,30], axis=1)        
        rob_tcp                     = np.asarray(data["state/raw"])[:,12:14]
        pos_start                   = rob_tcp[0]
        pos_end                     = rob_tcp[-1]
        dt                          = np.sqrt(np.power(pos_start[0] - pos_end[0], 2) + np.power(pos_start[1] - pos_end[1], 2))
        dt                          = dt # * data["speed_factor"]

        max_trj_length              = 350
        norm                        = np.take(self.normalization["values"], indices=[0,1,2,3,4,5,30], axis=1)
        trajectory_length           = trajectory.shape[0]
        
        if trajectory_length > max_trj_length:
            trajectory_in     = self.interpolateTrajectory(trajectory, max_trj_length)
            trajectory_length = max_trj_length
        elif trajectory_length < max_trj_length:
            trajectory_in = self.padSequence(trajectory, max_trj_length)  
        else:
            trajectory_in = trajectory     
        
        trajectory_in               = self.normalize(trajectory_in, norm[0,:], norm[1,:])
        trajectory_in               = self.gipperToSigmoid(trajectory_in)
        dmp_w                       = getDMPData(trajectory_in, 11)
        trajectory_out              = np.roll(trajectory_in, -1, axis=0)
        trajectory_out[-1,:]        = trajectory_out[-2,:]
        trajectory_out              = trajectory_out.flatten()
        trajectory_in               = trajectory_in.flatten()
        language                    = data["voice"]
        tokens                      = self.tokenize(language)
        tokens                      = tokens + [0] * (15 - len(tokens))
        onehot                      = np.zeros(6, dtype=np.int32)

        tid = data["target/id"]
        if data["target/type"] == "cup":
            tid += 20
        
        for i in range(6):
            if tid == int(np.round(classes[i])):
                onehot[i] = 1

        return tokens, features, trajectory_in, trajectory_out, trajectory_length, dt, onehot, dmp_w.flatten()

if __name__ == '__main__':
    limitGPUMemory()
    # Specify where GloVe and FRCNN can be found
    dc = DataConverter(dict_path="../GDrive/glove.6B.50d.txt", frcnn_path="../GDrive/rcnn/saved_model")

    # Specify where the raw data can be found and where you want the normalization to be saved
    dc.createNormalization(
        raw="../GDrive/collected/*.json",
        out="../GDrive/normalization_custom.pkl"
        )

    # Specify where the raw data can be found and where you want the validation data to be saved
    # Also specify how many samples should be part of the validation data
    # Note: Depending on how many data your have, you should leave some for training
    dc.createRecord(
        raw="../GDrive/collected/*.json",
        out="../GDrive/validate_custom.tfrecord",
        max_samples=4000
        )

    # Specify where the raw data can be found and where you want the training data to be saved
    # Also specify how many samples should be part of the training data.
    # Please also set min_samples to remove the data used for validation from the training data
    dc.createRecord(
        raw="../GDrive/collected/*.json",
        out="../GDrive/train_custom.tfrecord",
        min_samples=4000,
        max_samples=40000
        )