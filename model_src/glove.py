# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np

class GloveEmbeddings(tf.keras.layers.Layer):
    def __init__(self, file_path, **kwarg):
        super(GloveEmbeddings, self).__init__(**kwarg)
        self.file_path = file_path

    def build(self, input_shape):
        vsize, esize, data = self._loadData()
        self.embedding = tf.keras.layers.Embedding(vsize, esize, mask_zero=True)
        self.embedding.build(None)
        self.embedding.set_weights([data])
        self.embedding.trainable = False
 
    def call(self, inputs, training=None, mask=None, **kwargs):
        return self.embedding(inputs)
    
    def get_config(self):
        config = super(GloveEmbeddings, self).get_config()
        config.update({'file_path': self.file_path})
        return config

    def _loadData(self):
        embedding = []
        fh = open(self.file_path, "r", encoding="utf-8")
        for cnt, line in enumerate(fh):
            if cnt >= 300000:
                break
            tokens = line.strip().split(" ")
            vector = [float(t) for t in tokens[1:]]
            if len(embedding) == 0:
                embedding.append([0] * len(vector))
            embedding.append(vector)
            
        fh.close()
        embedding      = np.asarray(embedding)
        voc_size       = embedding.shape[0]
        embedding_size = embedding.shape[1]
        return voc_size, embedding_size, embedding.astype(dtype=np.float32)
    