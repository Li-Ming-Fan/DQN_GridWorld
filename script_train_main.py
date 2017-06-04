# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import GridWorld
import QNetModule as qnm

#
world = GridWorld.GameWorld(size = 5, partial = False);
                       
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 1000.0 #How many steps of training to reduce startE to endE.
#
num_episodes = 2000 #How many episodes of game environment to train network with.
max_epLength = 50 #The max allowed length of our episode.
pre_train_steps = 10000 #How many steps of random actions before training begins.
#
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 64 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network


#
tf.reset_default_graph()
#
mainQN = qnm.QNet(h_size, world)
targetQN = qnm.QNet(h_size, world)
#
init = tf.global_variables_initializer()
#
trainables = tf.trainable_variables()
targetNetOps = qnm.defineTargetNetLearning(trainables,tau)  # second half
#
myBuffer = qnm.experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# make a path for the model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)
#
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    #
    qnm.updateTargetNet(targetNetOps,sess) 
    #
    i = 0;
    while i < num_episodes:
        #
        episodeBuffer = qnm.experience_buffer()
        # 
        s = world.reset()   # reset environment
        s = qnm.processState(s)
        d = False
        rAll = 0
        j = 0  # counter for actions in this episode
        #The Q-Network
        while j < max_epLength: 
            j+=1
            #
            # choose an action by random or prediction
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]
            #
            sNext,r,d = world.step(a)
            sNext = qnm.processState(sNext)
            total_steps += 1
            #
            # save the experience to the episode buffer.
            episodeBuffer.add(np.reshape(np.array([s,a,r,sNext,d]),[1,5]))
            #
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    # get a random batch of experiences.
                    trainBatch = myBuffer.sample(batch_size)
                    #
                    # calculate Quant for future expectation
                    Action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Quant = sess.run(targetQN.quant, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    QFuture = Quant[range(batch_size),Action]
                    #
                    end_multiplier = -(trainBatch[:,4] - 1)
                    targetQ = trainBatch[:,2] + (y*QFuture * end_multiplier)
                    #
                    # update the main network
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]), \
                                   mainQN.targetQ:targetQ,\
                                   mainQN.actions:trainBatch[:,1]})
                    #
                    # update the target network
                    qnm.updateTargetNet(targetNetOps,sess)
                    #
            #
            rAll += r
            s = sNext
            
            if d == True:
                break
        #
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #
        # periodically save the model
        if i % 500 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print('model-'+str(i)+'.ckpt saved')
        if len(rList) % 10 == 0:
            print('episode: ', i+1,\
                  'total_steps: ', total_steps,\
                  'mean_reward: ', np.mean(rList[-10:]),\
                  'e: ', e)
        #
        i+=1;
        #
    #
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
    #
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

#
fold = 10;
rMat = np.resize(np.array(rList),[len(rList)//fold, fold])
rMean = np.average(rMat,1)
plt.figure()
plt.plot(rMean)
#

