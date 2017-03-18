"""
Solves the cartpole-v0 enviroment on OpenAI gym using policy search
A neural network is used to store the policy
At the end of each episode the target value for each taken action is
updated with the total normalized reward (up to a learning rate)
Then a standard supervised learning backprop on the entire batch is
executed
"""

import numpy as np
import numpy.matlib 
import core
import gym
from gym import wrappers
from PIL import Image
import psutil
import time
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from dqn import DQNAgent
from core import ReplayMemory
from dqn import DQNAgent
from core import *
from policy import *
from preprocessors import *
from keras import optimizers
from objectives import *
from linear_network import *

#initialize neural network to store policy

#load environment
env = gym.make('SpaceInvaders-v0')

#make the q_network
q_network = Sequential()

#make the preprocessors
history_preproc = HistoryPreprocessor(4)
atari_preproc = AtariPreprocessor()
preprocessor = PreprocessorSequence(atari_preproc, history_preproc)
#make the replay memory
memory = LinearReplayMemory()

#make the policy
policy = GreedyEpsilonPolicy(6, 0.35)

#take the gamma, nicely
gamma = 0.9

#target_update_freq : DUMMY
target_update_freq = 10

#num_burn_in : DUMMY
num_burn_in = 10

#train_freq : DUMMY
train_freq = 10

#batch_size
batch_size = 100

#Make the linear q network
linear_q_net = LinearQNetwork(q_network, preprocessor, memory, policy, gamma,
			   				  target_update_freq, num_burn_in, train_freq,
			   				  batch_size)
print("Created linear agent")


