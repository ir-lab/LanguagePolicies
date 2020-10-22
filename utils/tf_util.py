# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import os

def limitGPUMemory():
    print("Limiting GPU Memory")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print("Physical GPUs: {}\n\rLogical GPUs:  {}".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            print(e)

def trainOnCPU():
    print("Removing GPUs from device list")
    cpus = tf.config.list_physical_devices(device_type="CPU")
    tf.config.set_visible_devices(devices=cpus, device_type="CPU")
    tf.config.set_visible_devices(devices=[], device_type="GPU")

    if os.getenv("OMP_NUM_THREADS") is None:
        tf.config.threading.set_intra_op_parallelism_threads(4)
    else:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("OMP_NUM_THREADS")))
    tf.config.threading.set_inter_op_parallelism_threads(2)
