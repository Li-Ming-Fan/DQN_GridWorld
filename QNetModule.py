# -*- coding: utf-8 -*-

import numpy as np
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim

class QNet():
    def __init__(self, n_feat, world):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.sizeInput = world.state.size
        self.scalarInput = tf.placeholder(shape = [None, world.state.size],\
                                          dtype = tf.float32)
        #
        # feature extraction
        s = world.state.shape;
        shape = [-1, s[0], s[1], s[2]];
        self.imageIn = tf.reshape(self.scalarInput,shape=shape)
        # 7*7 --> 6*6 --> 5*5 --> 3*3 --> 1*1
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,\
            kernel_size=[2,2], stride=[1,1], padding='VALID',biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,\
            kernel_size=[2,2], stride=[1,1], padding='VALID',biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,\
            kernel_size=[3,3], stride=[1,1], padding='VALID',biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=n_feat,\
            kernel_size=[3,3], stride=[1,1], padding='VALID',biases_initializer=None)
        #
        # take the output from the final convolutional layer
        # and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        #
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        #
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([n_feat//2, world.actions]))
        self.VW = tf.Variable(xavier_init([n_feat//2, 1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        #
        # combine them together to the final Q-values.
        self.quant = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.quant,1)
        
        # learning process
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, world.actions, dtype=tf.float32)
        #
        self.Q = tf.reduce_sum(tf.multiply(self.quant, self.actions_onehot), axis=1)
        #
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        #
        
class experience_buffer():
    def __init__(self, buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

#        
def processState(state):
    return np.reshape(state, state.size)
    
#
def defineTargetNetLearning(tfVars,tau):
    total_vars = len(tfVars)
    half_vars = total_vars//2
    op_holder = []
    for idx,var in enumerate(tfVars[0:half_vars]):
        op_holder.append(tfVars[idx+half_vars].assign( \
            var.value()*tau + (1-tau)*tfVars[idx+half_vars].value() ))
    return op_holder
    # second half, target net ~ steady net,

def updateTargetNet(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    # seperate opertion definition and conduction
        