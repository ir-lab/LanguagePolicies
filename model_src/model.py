# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import pathlib
from model_src.attention import TopDownAttention
from model_src.glove import GloveEmbeddings
from model_src.dmp import DynamicMovementPrimitive
from model_src.basismodel import BasisModel
from model_src.feedbackcontroller import FeedbackController

class PolicyTranslationModel(tf.keras.Model):
    def __init__(self, od_path, glove_path, special=None):
        super(PolicyTranslationModel, self).__init__(name="policy_translation")
        self.units               = 32
        self.output_dims         = 7
        self.basis_functions     = 11

        if od_path != "":                
            od_path    = pathlib.Path(od_path)/"saved_model" 
            self.frcnn = tf.saved_model.load(str(od_path))
            self.frcnn = self.frcnn.signatures['serving_default']
            self.frcnn.trainable = False

        self.embedding = GloveEmbeddings(file_path=glove_path)
        self.lng_gru   = tf.keras.layers.GRU(units=self.units)

        self.attention = TopDownAttention(units=64)

        self.dout      = tf.keras.layers.Dropout(rate=0.25)

        # Units needs to be divisible by 7
        self.pt_global = tf.keras.layers.Dense(units=42, activation=tf.keras.activations.relu)

        self.pt_dt_1   = tf.keras.layers.Dense(units=self.units * 2, activation=tf.keras.activations.relu)
        self.pt_dt_2   = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.hard_sigmoid)

        self.controller = tf.keras.layers.RNN(
            FeedbackController(
                robot_state_size = self.units, 
                rnn_state_size   = (tf.TensorShape([self.output_dims]), tf.TensorShape([self.units])),
                dimensions       = self.output_dims, 
                basis_functions  = self.basis_functions,
                special          = None
            ), 
        return_sequences=True)
           
    @tf.function
    def call(self, inputs, training=False, use_dropout=True):
        if training:
            use_dropout = True

        language   = inputs[0]
        features   = inputs[1]
        # local      = features[:,:,:5]
        robot      = inputs[2]
        # dmp_state  = inputs[3]
        batch_size = tf.shape(language)[0]

        language  = self.embedding(language)
        language  = self.lng_gru(inputs=language, training=training) 

        # Calculate attention and expand it to match the feature size
        atn = self.attention((language, features))
        atn_w = tf.expand_dims(atn, 2)
        atn_w = tf.tile(atn_w, [1, 1, 5])
        # Compress image features and apply attention
        cfeatures = tf.math.multiply(atn_w, features)
        cfeatures = tf.math.reduce_sum(cfeatures, axis=1)

        # Add the language to the mix again. Possibly usefull to predict dt
        start_joints  = robot[:,0,:]
        cfeatures = tf.keras.backend.concatenate((cfeatures, language, start_joints), axis=1)

        # Policy Translation: Create weight + goal for DMP
        pt          = self.pt_global(cfeatures)
        pt          = self.dout(pt, training=tf.convert_to_tensor(use_dropout))
        dmp_dt      = self.pt_dt_2(self.pt_dt_1(pt)) + 0.1 # 0.1 prevents division by 0, just in case
        # dmp_dt      = d_out[2]

        # Run the low-level controller
        initial_state = [
            start_joints,
            tf.zeros(shape=[batch_size, self.units], dtype=tf.float32)
        ]
        generated, phase, weights = self.controller(inputs=robot, constants=(cfeatures, dmp_dt), initial_state=initial_state, training=training)

        return generated, (atn, dmp_dt, phase, weights)
    
    def getVariables(self, step=None):
        return self.trainable_variables
    
    def getVariablesFT(self):
        variables = []
        variables += self.pt_w_1.trainable_variables
        variables += self.pt_w_2.trainable_variables
        variables += self.pt_w_3.trainable_variables
        return variables
    
    def saveModelToFile(self, add):
        self.save_weights("Data/Model/" + add + "policy_translation")