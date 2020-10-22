# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hashids import Hashids
import time
import cv2
from utils.voice import Voice
import sklearn
import sklearn.preprocessing
from utils.intprim.gaussian_model import GaussianModel

class TBoardGraphs():
    def __init__(self, logname):
        self.__hashids           = Hashids()
        self.logdir              = "Data/TBoardLog/" + logname + "/"
        self.__tboard_train      = tf.summary.create_file_writer(self.logdir + "train/")
        self.__tboard_validation = tf.summary.create_file_writer(self.logdir + "validate/")
        self.voice               = Voice()

    def startDebugger(self):
        tf.summary.trace_on(graph=True, profiler=True)
    
    def stopDebugger(self):
        with self.__tboard_validation.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=self.logdir)

    def finishFigure(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def addTrainScalar(self, name, value, stepid):
        with self.__tboard_train.as_default():
            tf.summary.scalar(name, value, step=stepid)

    def addValidationScalar(self, name, value, stepid):
        with self.__tboard_validation.as_default():
            tf.summary.scalar(name, value, step=stepid)

    def plotTrajectory(self, y_true, y_pred, dt_true, dt_pred, stepid):
        fig, ax = plt.subplots(3,3)
        fig.set_size_inches(9, 9)

        dt_true = 1.0/dt_true.numpy()
        dt_pred = 1.0/dt_pred.numpy()[0]

        max_trj_len = y_true.shape[0]
        for sp in range(7):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()
            ax[idx,idy].plot(range(max_trj_len), y_pred[:,sp], alpha=0.5, color='midnightblue')
            ax[idx,idy].plot(range(max_trj_len), y_true[:,sp], alpha=0.5, color='forestgreen')
            # ax[idx,idy].plot([dt_pred, dt_pred], [-0.1, 1.1], alpha=0.5, linestyle=":", color="midnightblue")
            # ax[idx,idy].plot([dt_true, dt_true], [-0.1, 1.1], alpha=0.5, linestyle=":", color="forestgreen")
            # ax[idx,idy].set_ylim([-0.1, 1.1])

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("Trajectory", data=result, step=stepid)

    def idToText(self, id):
        names = ["", "ysr", "rsr", "gsr", "bsr", "psr", "ylr", "rlr", "glr", "blr", "plr", "yss", "rss", "gss", "bss", "pss", "yls", "rls", "gls", "bls", "pls"]
        return names[id]

    def plotImageRegions(self, image, image_dict, stepid):
        # Visualization of the results of a detection.
        num_detected = len([v for v in image_dict["detection_scores"][0] if v > 0.5]) 
        image_np     = image.numpy()       
        for i in range(num_detected):
            ymin, xmin, ymax, xmax = image_dict['detection_boxes'][0][i,:]
            pt1 = (int(xmin*image_np.shape[1]), int(ymin*image_np.shape[0]))
            pt2 = (int(xmax*image_np.shape[1]), int(ymax*image_np.shape[0]))
            image_np = cv2.rectangle(image_np, pt1, pt2, (255, 0, 0), 2)
            image_np = cv2.putText(image_np, self.idToText(image_dict['detection_classes'][0][i]) + " {:.1f}%".format(image_dict["detection_scores"][0][i] * 100), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        fig = plt.figure()
        plt.imshow(image_np)

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("Image", data=result, step=stepid)

    def plotAttention(self, attention_weights, image_dict, language, stepid):
        attention_weights = attention_weights.numpy()
        classes           = image_dict["detection_classes"][0][:len(attention_weights)].numpy().astype(dtype=np.int32)
        classes           = [self.idToText(i) for i in classes]
        x                 = np.arange(len(attention_weights))
        
        fig, ax = plt.subplots()
        plt.bar(x, attention_weights)
        plt.xticks(x, classes)
        ax.set_ylim([0, 1])
        plt.text(0.01, 0.95, self.voice.tokensToSentence(language.numpy().tolist()), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("Attention", data=result, step=stepid)
    
    def plotClassAccuracy(self, gt_class, pred_class, pred_class_std, language, stepid):
        labels     = ["ysr", "rsr", "gsr", "bsr", "psr", "ylr", "rlr", "glr", "blr", "plr", "yss", "rss", "gss", "bss", "pss", "yls", "rls", "gls", "bls", "pls"]
        gt_class   = gt_class.numpy()
        pred_class = pred_class.numpy()
        x          = np.arange(len(gt_class))
        width      = 0.35
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, gt_class, width, label='GT', color="forestgreen")
        rects2 = ax.bar(x + width/2, pred_class, width, yerr=pred_class_std, label='Pred', color="midnightblue")
        ax.set_xticks(x)
        # ax.set_xticklabels(labels)
        plt.text(0.01, 0.95, self.voice.tokensToSentence(language.numpy().tolist()), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("Attention", data=result, step=stepid)

    def plotDeltaT(self, y_true, y_pred, stepid):
        gt = y_true.numpy()
        pd = y_pred.numpy()[:,0]
        jdata = np.stack((gt,pd), axis=1)
        svals = jdata[np.argsort(jdata[:,0]),:]
        x     = np.arange(svals.shape[0])
        width = 0.35
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, svals[:,0], width, label='GT', color="forestgreen")
        rects2 = ax.bar(x + width/2, svals[:,1], width, label='Pred', color="midnightblue")
        ax.set_xticks(x)

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("DeltaT", data=result, step=stepid)

    def plotWeights(self, gt_w, pred_w, stepid):
        fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,sharex=True)
        # fig.set_size_inches(4, 10)

        combined_weights = np.concatenate((gt_w.numpy(), pred_w.numpy()), axis=0).T

        ax1.imshow(combined_weights[:,:7], cmap="RdBu")
        ax2.imshow(combined_weights[:,7:], cmap="RdBu")

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("Weights", data=result, step=stepid)

    def interpolateTrajectory(self, trj, target):
        current_length = trj.shape[0]
        dimensions     = trj.shape[1]
        result         = np.zeros((target, dimensions), dtype=np.float32)
    
        for i in range(dimensions):
            result[:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[:,i])
        
        return result

    def plotDMPTrajectory(self, y_true, y_pred, y_pred_std, phase, dt, p_dt, stepid):
        y_true      = y_true.numpy()
        y_pred      = y_pred.numpy()
        y_pred_std  = y_pred_std.numpy()
        phase       = phase.numpy()
        dt          = dt.numpy() * 350.0
        p_dt        = p_dt.numpy()
        trj_len      = y_true.shape[0]
        
        fig, ax = plt.subplots(3,3)
        fig.set_size_inches(9, 9)
        for sp in range(7):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()

            # GT Trajectory:
            ax[idx,idy].plot(range(trj_len), y_true[:,sp],   alpha=1.0, color='forestgreen')            
            ax[idx,idy].plot(range(y_pred.shape[0]), y_pred[:,sp], alpha=0.75, color='mediumslateblue')
            ax[idx,idy].errorbar(range(y_pred.shape[0]), y_pred[:,sp], xerr=None, yerr=y_pred_std[:,sp], alpha=0.25, fmt='none', color='mediumslateblue')
            ax[idx,idy].set_ylim([-0.1, 1.1])
            ax[idx,idy].plot([dt, dt], [0.0,1.0], linestyle=":", color='forestgreen')
        
        ax[2,2].plot(range(y_pred.shape[0]), phase[:,0], color='orange')
        ax[2,2].plot([dt, dt], [0.0,1.0], linestyle=":", color='forestgreen')
        ax[2,2].plot([p_dt*350.0, p_dt*350.0], [0.0,1.0], linestyle=":", color='mediumslateblue')
        ax[2,2].set_ylim([-0.1, 1.1])

        result = np.expand_dims(self.finishFigure(fig), 0)
        plt.close()
        with self.__tboard_validation.as_default():
            tf.summary.image("Trajectory", data=result, step=stepid)