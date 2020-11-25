# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import numpy as np
from utils.graphs import TBoardGraphs
import shutil
import pycurl
import pickle
import os

class Network():
    def __init__(self, model, logname, lr, lw_atn, lw_w, lw_trj, lw_dt, lw_phs, log_freq=25):
        self.optimizer         = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        self.model             = model
        self.total_steps       = 0
        self.logname           = logname

        if self.logname.startswith("Intel$"):
            self.instance_name = self.logname.split("$")[1]
            self.logname       = self.logname.split("$")[0]
        else:
            self.instance_name = None

        self.tboard            = TBoardGraphs(self.logname)
        self.loss              = tf.keras.losses.CategoricalCrossentropy()
        self.global_best_loss  = 10000
        self.last_written_step = -1
        self.log_freq          = log_freq

        self.lw_atn = lw_atn 
        self.lw_w   = lw_w 
        self.lw_trj = lw_trj
        self.lw_dt  = lw_dt
        self.lw_phs = lw_phs

    def setDatasets(self, train, validate):
        self.train_ds = train.ds
        self.val_ds   = validate.ds

    def train(self, epochs):
        self.global_step = 0
        for epoch in range(epochs):
            print("Epoch: {:3d}/{:3d}".format(epoch+1, epochs)) 
            validation_loss = 0.0
            train_loss = []
            for step, (d_in, d_out) in enumerate(self.train_ds):
                if step % 100 == 0:
                    validation_loss = self.runValidation(quick=True, pnt=False)                    
                train_loss.append(self.step(d_in, d_out, train=True))
                self.loadingBar(step, self.total_steps, 25, addition="Loss: {:.6f} | {:.6f}".format(np.mean(train_loss[-10:]), validation_loss))
                if epoch == 0:
                    self.total_steps += 1
                self.global_step += 1
            self.loadingBar(self.total_steps, self.total_steps, 25, addition="Loss: {:.6f}".format(np.mean(train_loss)), end=True)
            self.runValidation(quick=False)

            self.model.saveModelToFile(self.logname + "/")

            if epoch % self.log_freq == 0 and self.instance_name is not None:
                self._uploadToCloud(epoch)

        if self.instance_name is not None:
            self._uploadToCloud()
    
    def _uploadToCloud(self, epoch=None):
        # Not available in public version
        pass

    def _curlUpload(self, path):
        # Not available in public version
        pass
    
    def runValidation(self, quick=False, pnt=True): 
        if not quick:
            print("Running full validation...")
        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_ds):
            val_loss.append(self.step(d_in, d_out, train=False))
            if quick:
                break
        d_in_graphs  = (tf.tile(tf.expand_dims(d_in[0][0], 0),[50,1]), tf.tile(tf.expand_dims(d_in[1][0], 0),[50,1,1]), tf.tile(tf.expand_dims(d_in[2][0], 0),[50,1,1]))
        d_out_graphs = (tf.tile(tf.expand_dims(d_out[0][0], 0),[50,1,1]), tf.tile(tf.expand_dims(d_out[1][0], 0),[50,1]), 
                        tf.tile(tf.expand_dims([d_out[2][0]], 0),[50,1]), tf.tile(tf.expand_dims(d_out[3][0], 0),[50,1,1]))
        self.createGraphs((d_in[0][0], d_in[1][0], d_in[2][0]),
                          (d_out[0][0], d_out[1][0], d_out[2][0], d_out[3][0]), 
                          self.model(d_in_graphs, training=True, use_dropout=True))
        if pnt:
            print("  Validation Loss: {:.6f}".format(np.mean(val_loss)))
        return np.mean(val_loss)

    def step(self, d_in, d_out, train):
        with tf.GradientTape() as tape:
            result = self.model(d_in, training=train)
            loss, (atn, trj, dt, phs, wght) = self.calculateLoss(d_out, result, train)
        if train:
            gradients = tape.gradient(loss, self.model.getVariables(self.global_step))
            self.optimizer.apply_gradients(zip(gradients, self.model.getVariables(self.global_step)))
            self.tboard.addTrainScalar("Loss", loss, self.global_step)
            self.tboard.addTrainScalar("Loss Attention", atn, self.global_step)
            self.tboard.addTrainScalar("Loss Trajectory", trj, self.global_step)
            self.tboard.addTrainScalar("Loss Phase", phs, self.global_step)
            self.tboard.addTrainScalar("Loss Weight", wght, self.global_step)
            self.tboard.addTrainScalar("Loss Delta T", dt, self.global_step)
        else:
            if self.last_written_step != self.global_step:
                self.last_written_step = self.global_step
                self.tboard.addValidationScalar("Loss", loss, self.global_step)
                self.tboard.addValidationScalar("Loss Attention", atn, self.global_step)
                self.tboard.addValidationScalar("Loss Trajectory", trj, self.global_step)
                self.tboard.addValidationScalar("Loss Phase", phs, self.global_step)
                self.tboard.addValidationScalar("Loss Weight", wght, self.global_step)
                self.tboard.addValidationScalar("Loss Delta T", dt, self.global_step)
                if loss < self.global_best_loss:
                    self.global_best_loss = loss
                    self.model.saveModelToFile(self.logname + "/best/")

        return loss.numpy()
    
    def interpolateTrajectory(self, trj, target):
        batch_size     = trj.shape[0]
        current_length = trj.shape[1]
        dimensions     = trj.shape[2]
        result         = np.zeros((batch_size, target, dimensions), dtype=np.float32)
    
        for b in range(batch_size):
            for i in range(dimensions):
                result[b,:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[b,:,i])
        
        return result

    def calculateMSEWithPaddingMask(self, y_true, y_pred, mask):
        mse = tf.math.pow(y_true - y_pred, 2.0)
        mse = tf.math.multiply(mse, mask)
        n   = mse.shape[-1]
        mse = (1.0 / n) * tf.reduce_sum(mse, axis=-1)
        return mse

    def calculateLoss(self, d_out, result, train):
        gen_trj, (atn, dmp_dt, phs, wght)                       = result
        generated, attention, delta_t, weights, phase, loss_atn = d_out

        weight_dim  = [3.0, 3.0, 3.0, 1.0, 0.5, 1.0, 0.1]
        
        atn_loss = self.loss(y_true=attention, y_pred=atn)

        dt_loss  = tf.math.reduce_mean(tf.keras.metrics.mean_squared_error(delta_t, dmp_dt[:,0]))
        
        trj_loss = self.calculateMSEWithPaddingMask(generated, gen_trj, tf.tile([[weight_dim]], [16, 350, 1]))
        trj_loss = tf.reduce_mean(tf.math.multiply(trj_loss, loss_atn), axis=1)
        trj_loss = tf.reduce_mean(trj_loss)
   
        phs_loss = tf.math.reduce_mean(self.calculateMSEWithPaddingMask(phase, phs[:,:,0], loss_atn))

        weight_loss = tf.math.reduce_mean(tf.keras.metrics.mean_squared_error(wght[:,:-1,:,:], tf.roll(wght, shift=-1, axis=1)[:,:-1,:,:]), axis=-1)
        weight_loss = tf.math.reduce_mean(tf.math.multiply(weight_loss, loss_atn[:,:-1]))

        return (atn_loss * self.lw_atn +
                trj_loss * self.lw_trj +
                phs_loss * self.lw_phs +
                weight_loss * self.lw_w + 
                dt_loss  * self.lw_dt,
                (atn_loss, trj_loss, dt_loss, phs_loss, weight_loss)
            )
    
    def loadingBar(self, count, total, size, addition="", end=False):
        if total == 0:
            percent = 0
        else:
            percent = float(count) / float(total)
        full = int(percent * size)
        fill = size - full
        print("\r  {:5d}/{:5d} [".format(count, total) + "#" * full + " " * fill + "] " + addition, end="")
        if end:
            print("")
        sys.stdout.flush()

    def createGraphs(self, d_in, d_out, result):
        language, image, robot_states            = d_in
        target_trj, attention, delta_t, weights  = d_out
        gen_trj, (atn, dmp_dt, phase, wght)      = result

        self.tboard.plotClassAccuracy(attention, tf.math.reduce_mean(atn, axis=0), tf.math.reduce_std(atn, axis=0), language, stepid=self.global_step)
        self.tboard.plotDMPTrajectory(target_trj, tf.math.reduce_mean(gen_trj, axis=0), tf.math.reduce_std(gen_trj, axis=0),
                                      tf.math.reduce_mean(phase, axis=0), delta_t, tf.math.reduce_mean(dmp_dt, axis=0), stepid=self.global_step)
