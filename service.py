#!/usr/bin/env python

# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
if sys.version_info[0] < 3:
    PYTHON_VERSION = 2
    import rospy
    from policy_translation.srv import NetworkPT, NetworkPTResponse, TuneNetwork, TuneNetworkResponse
else:
    PYTHON_VERSION = 3
    import rclpy
    from policy_translation.srv import NetworkPT, TuneNetwork

from model_src.model import PolicyTranslationModel
from utils.network import Network
from utils.tf_util import trainOnCPU, limitGPUMemory
from utils.intprim.gaussian_model import GaussianModel
import tensorflow as tf
import numpy as np
import re
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
from utils.intprim.gaussian_model import GaussianModel
import glob
import json
import pickle
import copy

FORCE_CPU    = True
RUN_BASELINE = False
USE_DROPOUT  = True

if FORCE_CPU:
    trainOnCPU()
else:
    limitGPUMemory()

if RUN_BASELINE:
    pass
else:
    print("Running Policy Translation Model")
    # model = PolicyTranslationModel(
    #     od_path="object_detection/exported_sim_rss",
    #     glove_path="Data/glove.6B.50d.txt",
    #     special=None # None, atn, rnn, ff
    # )
    model = PolicyTranslationModel(
        od_path="../GDrive/rcnn",
        glove_path="../GDrive/glove.6B.50d.txt",
        special=None # None, atn, rnn, ff
    )

bs = 2
model((
    np.ones((bs, 15), dtype=np.int64),
    np.ones((bs, 6, 5), dtype=np.float32),
    np.ones((bs, 500, 7), dtype=np.float32)
))
if RUN_BASELINE:
    model.load_weights("Data/Model/VmzPBGDmByW/best/naive_baseline") # RNN Baseline
else:
    model.load_weights("../GDrive/model/policy_translation") # Best One
    # model.load_weights("Data/IntelResults/Intel_abl_itl_1_epfinal_models/best/policy_translation") # Ablation TRJ
    # model.load_weights("Data/IntelResults/Intel_abl_aux_2_ep300_models/best/policy_translation") # Ablation TRJ + Controller
    # model.load_weights("Data/IntelResults/Intel_abl_aux_3_ep300_models/best/policy_translation") # Ablation TRJ +  Att
    # model.load_weights("Data/IntelResults/Intel_abl2_mlp_atn_ep300_models/best/policy_translation") # Variant mlp attention
    # model.load_weights("Data/IntelResults/Intel_abl2_mlp_tgt_ep350_models/best/policy_translation") # Variant mlp trajectory
    # model.load_weights("Data/IntelResults/Intel_abl2_rnn_ep325_models/best/policy_translation") # Variant mlp trajectory
    # model.load_weights("Data/IntelResults/Intel_ds_30000_ep300_models/best/policy_translation") # Dataset 30000 samples
    # model.load_weights("Data/IntelResults/Intel_ds_20000_ep300_models/best/policy_translation") # Dataset 20000 samples
    # model.load_weights("Data/IntelResults/Intel_ds_10000_epfinal_models/best/policy_translation") # Dataset 10000 samples
    # model.load_weights("Data/IntelResults/Intel_ds_2500_v2_ep1200_models/best/policy_translation") # Dataset 10000 samples
model.summary()

class Otto():
    def __init__(self):
        self.dictionary    = self._loadDictionary("../GDrive/glove.6B.50d.txt")
        self.regex         = re.compile('[^a-z ]')
        self.bridge        = CvBridge()
        self.history       = []

        if PYTHON_VERSION == 2:
            rospy.init_node("neural_network")
            self.service_nn = rospy.Service("/network", NetworkPT, self.cbk_network_dmp)
            # self.service_tn = rospy.Service("/network/tune", TuneNetwork, self.cbk_finetune)
            self.normalization = pickle.load(open("../GDrive/normalization_v2.pkl", mode="rb"))

        elif PYTHON_VERSION == 3:
            rclpy.init(args=None)
            self.node = rclpy.create_node("neural_network")
            self.service_nn = self.node.create_service(NetworkPT,   "/network",      self.cbk_network_dmp_ros2)
            # self.service_tn = self.node.create_service(TuneNetwork, "/network/tune", self.cbk_finetune)
            self.normalization = pickle.load(open("../GDrive/normalization_v2.pkl", mode="rb"), encoding="latin1")
        print("Ready")

    def runNode(self):
        if PYTHON_VERSION == 2:
            rospy.spin()
        elif PYTHON_VERSION == 3:
            while rclpy.ok():
                rclpy.spin_once(self.node)
            self.node.destroy_service(self.service_nn)
            self.node.destroy_service(self.service_tn)
            rclpy.shutdown()

    def _loadDictionary(self, file):
        __dictionary = {}
        __dictionary[""] = 0 # Empty string
        fh = open(file, "r", encoding="utf-8")
        for line in fh:
            if len(__dictionary) >= 300000:
                break
            if PYTHON_VERSION == 2:
                line = line.decode("utf-8")
            tokens = line.strip().split(" ")
            __dictionary[tokens[0]] = len(__dictionary)
        fh.close()
        return __dictionary

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
    
    def normalize(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise ArrayDimensionMismatch()
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value

    def interpolateTrajectory(self, trj, target):
        current_length = trj.shape[0]
        dimensions     = trj.shape[1]
        result         = np.zeros((target, trj.shape[1]), dtype=np.float32)
              
        for i in range(dimensions):
            result[:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[:,i])
        
        return result

    def cbk_network_dmp_ros2(self, req, res):
        res.trajectory, res.confidence, res.timesteps, res.weights, res.phase = self.cbk_network_dmp(req)
        return res
    
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):   
        if img_msg.encoding != "8UC3":     
            self.node.get_logger().info("Unrecognized image type: " + encoding)
            exit(0)
        # dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
        dtype      = "uint8"
        n_channels = 3

        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, img_msg.width),
                            dtype=dtype, buffer=img_buf)
        else:
            im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_buf)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()

        if desired_encoding == 'passthrough':
            return im

        from cv_bridge.boost.cv_bridge_boost import cvtColor2

        try:
            res = cvtColor2(im, img_msg.encoding, desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def cbk_network_dmp(self, req):
        if req.reset:
            self.req_step = 0
            self.sfp_history = []
            try:
                if PYTHON_VERSION == 2:
                    image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
                elif PYTHON_VERSION == 3:
                    image = self.imgmsg_to_cv2(req.image)
            except CvBridgeError as e:
                print(e)
            language = self.tokenize(req.language)
            self.language = language + [0] * (15-len(language))

            image_features = model.frcnn(tf.convert_to_tensor([image], dtype=tf.uint8))

            scores   = image_features["detection_scores"][0, :6].numpy().astype(dtype=np.float32)
            scores   = [0.0 if v < 0.5 else 1.0 for v in scores.tolist()]

            classes  = image_features["detection_classes"][0, :6].numpy().astype(dtype=np.int32)
            classes  = [v * scores[k] for k, v in enumerate(classes.tolist())]

            boxes    = image_features["detection_boxes"][0, :6, :].numpy().astype(dtype=np.float32)
            
            # features = image_features["detection_features"][0, :5, :, :, :].numpy().astype(dtype=np.float32)
            # features = np.mean(features, axis=(1,2))

            self.features = np.concatenate((np.expand_dims(classes,1), boxes), axis=1)

            self.history  = []        

        self.history.append(list(req.robot)) 

        robot           = np.asarray(self.history, dtype=np.float32)
        self.input_data = (
            tf.convert_to_tensor(np.tile([self.language],[250, 1]), dtype=tf.int64), 
            tf.convert_to_tensor(np.tile([self.features],[250, 1, 1]), dtype=tf.float32),
            tf.convert_to_tensor(np.tile([robot],[250, 1, 1]), dtype=tf.float32)
        )

        if RUN_BASELINE:
            generated, phase = model(self.input_data, training=tf.constant(False), use_dropout=tf.constant(USE_DROPOUT))
            dmp_dt    = np.zeros((1,10))
            phase     = tf.expand_dims(phase, 2)
            weights   = np.zeros((1,7,11))
        else:
            generated, (atn, dmp_dt, phase, weights) = model(self.input_data, training=tf.constant(False), use_dropout=tf.constant(True))
        self.trj_gen    = tf.math.reduce_mean(generated, axis=0).numpy()
        self.trj_std    = tf.math.reduce_std(generated, axis=0).numpy()
        self.timesteps  = int(tf.math.reduce_mean(dmp_dt).numpy() * 500)
        self.b_weights  = tf.math.reduce_mean(weights, axis=0).numpy()

        phase_value     = tf.math.reduce_mean(phase, axis=0).numpy()
        phase_value     = phase_value[-1,0]

        # self.history.append(self.trj_gen[-1,:].tolist()) 

        # #### Future predictions from the network
        self.sfp_history.append(self.b_weights[-1,:,:])
        if phase_value > 0.95 and len(self.sfp_history) > 100:
            trj_len    = len(self.sfp_history)
            basismodel = GaussianModel(degree=11, scale=0.012, observed_dof_names=("Base","Shoulder","Ellbow","Wrist1","Wrist2","Wrist3","Gripper"))
            domain     = np.linspace(0, 1, trj_len, dtype=np.float64)
            trajectories = []
            for i in range(trj_len):
                trajectories.append(np.asarray(basismodel.apply_coefficients(domain, self.sfp_history[i].flatten())))
            trajectories = np.asarray(trajectories)
            np.save("trajectories", trajectories)
            np.save("history", self.history)

            gen_trajectory = []
            var_trj        = np.zeros((trj_len, trj_len, 7), dtype=np.float32)
            for w in range(trj_len):
                gen_trajectory.append(trajectories[w,w,:])
                # for t in range(w, trj_len):
                #     mean = trajectories[t,t,:]
                #     ftr  = trajectories[w,t,:]
                #     diff = np.abs(mean - ftr)
                #     var_trj[w,t,:] = np.asarray([max(var_trj[w,t,i], diff[i]) for i in range(7)])
            gen_trajectory = np.asarray(gen_trajectory)
            np.save("gen_trajectory", gen_trajectory)            
            # print(var_trj[0].shape)
            # for fid in range(0,trj_len):
            #     print("{}/{}".format(fid, trj_len))
            #     fig, ax = plt.subplots(3,3)
            #     fig.set_size_inches(9, 9)
            #     for sp in range(7):
            #         idx = sp // 3
            #         idy = sp  % 3
            #         ax[idx,idy].clear()

            #         # GT Trajectory:
            #         ax[idx,idy].plot(range(fid), gen_trajectory[:fid,sp], alpha=0.75, linewidth=3, color='mediumslateblue')
            #         # ax[idx,idy].errorbar(range(trj_len), gen_trajectory[:,sp], xerr=None, yerr=var_trj[fid,:,sp], alpha=0.2, fmt='none', color='mediumslateblue')
            #         # for line in range(fid,trj_len):
            #         ax[idx,idy].plot(range(fid, trj_len), trajectories[fid,fid:,sp], alpha=0.5, color='orange')

            #         ax[idx,idy].set_ylim([-0.1, 1.1])
            #     # plt.show()
            #     # break
            #     plt.savefig("Data/Images/1_" + str(fid).zfill(3) + ".png", dpi=100)
            #     plt.close()            
            # fig, ax = plt.subplots(3,3)
            # fig.set_size_inches(9, 9)
            # for fid in range(0,trj_len):
            #     for sp in range(7):
            #         idx = sp // 3
            #         idy = sp  % 3
            #         ax[idx,idy].clear()
            #         # GT Trajectory:
            #         if fid == trj_len -1:
            #             ax[idx,idy].plot(range(trj_len), gen_trajectory[:,sp], alpha=0.75, linestyle=":", linewidth=1, color='mediumslateblue')
            #         # ax[idx,idy].errorbar(range(trj_len), gen_trajectory[:,sp], xerr=None, yerr=var_trj[fid,:,sp], alpha=0.2, fmt='none', color='mediumslateblue')
            #         # for line in range(fid,trj_len):
            #         ax[idx,idy].plot(range(fid, trj_len), trajectories[fid,fid:,sp], alpha=1, color='orange')
            #         # ax[idx,idy].set_ylim([-0.1, 1.1])
            # # plt.savefig("Data/Images/1_" + str(fid).zfill(3) + ".png", dpi=100)
            # plt.show()
            self.sfp_history = []

        #### Just plots the image regions
        # if req.reset:
        #     self.plotImageRegions(image, image_features, tf.math.reduce_mean(atn, axis=0).numpy())
        #     plt.show()

        #### Predicting stuff into the future by feeding the output directly back to the network
        # tmp = copy.deepcopy(self.history)
        # tmp.append(self.trj_gen[-1,:].tolist())
        # if True:
        #     n_cut           = 150    
        #     while len(tmp) < n_cut:
        #         robot           = np.asarray(tmp , dtype=np.float32)
        #         self.input_data = (
        #             tf.convert_to_tensor(np.tile([self.language],[250, 1]), dtype=tf.int64), 
        #             tf.convert_to_tensor(np.tile([self.features],[250, 1, 1]), dtype=tf.float32),
        #             tf.convert_to_tensor(np.tile([robot],[250, 1, 1]), dtype=tf.float32)
        #         )
        #         generated, (_, _, _, _) = model(self.input_data, training=tf.constant(False), use_dropout=tf.constant(True))
        #         tmp.append(tf.math.reduce_mean(generated, axis=0).numpy()[-1,:])
        #     self.sfp_history.append(np.asarray(tmp))
        # if len(self.history) == n_cut:
        #     fig, ax = plt.subplots(3,3)
        #     fig.set_size_inches(9, 9)
        #     for i in range(len(self.sfp_history)):
        #         for sp in range(7):
        #             idx = sp // 3
        #             idy = sp  % 3
        #             ax[idx,idy].clear()
        #             for i in range(len(self.sfp_history)):
        #                 ax[idx, idy].plot(range(self.sfp_history[i].shape[0]), self.sfp_history[i][:,sp], color='mediumslateblue', alpha=0.1)
        #             ax[idx,idy].set_ylim([-0.1, 1.1])
        #     plt.show()            

        #### Actual stochastic forward passes
        # n_passes = 5
        # self.sfp_history.append(generated.numpy()[:n_passes,-1,:])
        # sfp_len = len(self.sfp_history)
        # print(sfp_len)
        # if sfp_len == n_cut:
        #     fig, ax = plt.subplots(3,3)
        #     fig.set_size_inches(9, 9)
        #     for sp in range(7):
        #         idx = sp // 3
        #         idy = sp  % 3
        #         ax[idx,idy].clear()
        #         for i in range(sfp_len):
        #             ax[idx, idy].scatter(np.ones((n_passes)) * i, self.sfp_history[i][:,sp], color='mediumslateblue', alpha=0.1)
        #         ax[idx,idy].set_ylim([-0.1, 1.1])
        #     plt.show()

        # if req.plot or self.req_step == 0:
        #     tgt_object = np.argmax(tf.math.reduce_mean(atn, axis=0))
        #     print("Task Summary:")
        #     print("  Sentence: " + req.language)
        #     print("  Target Object: {} (Confidence: {:.3f})".format(self.idToText(int(classes[tgt_object])), tf.math.reduce_mean(atn, axis=0)[tgt_object]))
        #     print("  Timesteps", self.timesteps)
        #     irb = boxes[tgt_object]
        #     h,w,_ = image.shape
        #     img = image[int(h*irb[0]):int(h*irb[2]),int(w*irb[1]):int(w*irb[3]),:]
        #     # self.plotTrajectory(self.trj_gen, self.trj_std, img)
        #     self.plotImageRegions(image, image_features, tf.math.reduce_mean(atn, axis=0).numpy())
        #     plt.savefig("/home/sstepput/Desktop/NueIPS Figures/color/{}.png".format(np.random.randint(0, 10000)))
        #     plt.close()
        #     plt.show()
        
        self.req_step += 1
        if PYTHON_VERSION == 3:
            return (self.trj_gen.flatten().tolist(), self.trj_std.flatten().tolist(), self.timesteps, self.b_weights.flatten().tolist(), float(phase_value)) 
          
        response            = NetworkPTResponse()
        response.trajectory = self.trj_gen.flatten().tolist()
        response.confidence = self.trj_std.flatten().tolist()
        response.timesteps  = self.timesteps
        response.weights    = self.dmp_w_mean.flatten().tolist()
        response.phase      = phase_value
        return response
    
    def idToText(self, id):
        names = ["", "Yellow Small Round", "Red Small Round", "Green Small Round", "Blue Small Round", "Pink Small Round",
                     "Yellow Large Round", "Red Large Round", "Green Large Round", "Blue Large Round", "Pink Large Round",
                     "Yellow Small Square", "Red Small Square", "Green Small Square", "Blue Small Square", "Pink Small Square",
                     "Yellow Large Square", "Red Large Square", "Green Large Square", "Blue Large Square", "Pink Large Square",
                     "Cup Red", "Cup Green", "Cup Blue"]
        return names[id]
    
    def plotTrajectory(self, trj, error, image):
        fig, ax = plt.subplots(3,3)
        fig.set_size_inches(9, 9)

        for sp in range(7):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()
            ax[idx,idy].plot(range(trj.shape[0]), trj[:,sp], alpha=0.5, color='mediumslateblue')
            ax[idx,idy].errorbar(range(trj.shape[0]), trj[:,sp], xerr=None, yerr=error[:,sp], alpha=0.1, fmt='none', color='mediumslateblue')
            ax[idx,idy].set_ylim([-0.1, 1.1])

        ax[2,1].imshow(image)

    def plotImageRegions(self, image_np, image_dict, atn):
        # Visualization of the results of a detection.
        tgt_object   = np.argmax(atn)
        num_detected = len([v for v in image_dict["detection_scores"][0] if v > 0.5]) 
        num_detected = min(num_detected, len(atn))
        for i in range(num_detected):
            ymin, xmin, ymax, xmax = image_dict['detection_boxes'][0][i,:]
            pt1 = (int(xmin*image_np.shape[1]), int(ymin*image_np.shape[0]))
            pt2 = (int(xmax*image_np.shape[1]), int(ymax*image_np.shape[0]))
            image_np = cv2.rectangle(image_np, pt1, pt2, (156, 2, 2), 1)
            if i == tgt_object:
                image_np = cv2.rectangle(image_np, pt1, pt2, (30, 156, 2), 2)
                image_np = cv2.putText(image_np, "{:.1f}%".format(atn[i] * 100), (pt1[0]-10, pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 156, 2), 2, cv2.LINE_AA)
            # image_np = cv2.putText(image_np, self.idToText(int(image_dict['detection_classes'][0][i].numpy())) + " {:.1f}%".format(image_dict["detection_scores"][0][i] * 100), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            # image_np = cv2.putText(image_np, "{:.1f}%".format(atn[i] * 100), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        fig = plt.figure()
        plt.imshow(image_np)
    
if __name__ == "__main__":
    ot = Otto()
    ot.runNode()