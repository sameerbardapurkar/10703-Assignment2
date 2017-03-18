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

#initialize neural network to store policy

#load environment
env = gym.make('SpaceInvaders-v0')
state = env.reset()
sars = []
states = []
print state
#env = gym.wrappers.Monitor(env, 'monitor')
done = False
samples = []
while done == False:
    action = env.action_space.sample()

    new_state, reward, done, info = env.step(action)
    env.render()
    state = new_state
    img = Image.fromarray(state, 'RGB')
    img = img.crop((0, 20, 160, 210))
    img = img.convert('L')
    img = img.resize((84, 84), 3)
    np_img = np.asarray(img)
    print(np_img)
    print(np_img.shape)
    time.sleep(0.5)
    for proc in psutil.process_iter():
    	if proc.name() == "display":
        	proc.kill()
env.close()