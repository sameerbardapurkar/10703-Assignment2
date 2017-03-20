from dqn import DQNAgent
from core import ReplayMemory
from dqn import DQNAgent
from core import *
from policy import *
from preprocessors import *
import keras
from keras import optimizers
from keras import callbacks
from keras.models import Model
from objectives import *
import matplotlib.pyplot as plt
import random
import copy
from keras.models import Sequential

class LinearReplayMemory(ReplayMemory):

    def __init__(self, max_size=10000, window_length=5):
        self.max_size = max_size
        self.window_length = window_length
        self.memory = []
        self.ind = 0

    def append(self, state, action, reward):
        if(self.ind >= self.max_size):
            del self.memory[0]
        self.memory.append([self.ind, state, action, reward])
        self.ind += 1

    def end_episode(self, final_state, is_terminal):
        self.final_state = final_state
        self.is_terminal = is_terminal
        
    def sample(self, batch_size, indexes=None):
        sample_states = []
        sample_actions = []
        sample_rewards = []
        sample_states_prime = []
        sample_ind = self.ind
        if(self.ind >= self.max_size):
            sample_ind = self.max_size
        nums = range(0, sample_ind - 1)
        if(batch_size > len(nums)):
            batch_size = len(nums)
        chosen_nums = random.sample(nums, batch_size) 

        for i in chosen_nums:
            x = np.zeros((84,84,4))
            
            for j in range(i - 3, i+1):
                if j < 0:
                    continue
                np.delete(x, 0, 2)
                shape = self.memory[j][1].shape
                np.append(x, np.reshape(self.memory[j][1],(shape[0], shape[1], 1)), 2)
                
            x_prime = np.zeros((84,84,4))
            
            for j in range(i - 2, i+2):
                if j < 0:
                    continue
                np.delete(x_prime, 0, 2)
                shape = self.memory[j][1].shape
                np.append(x_prime, np.reshape(self.memory[j][1],(shape[0], shape[1], 1)), 2)
                
            sample_states.append(x)
            sample_actions.append(self.memory[i][2])
            sample_rewards.append(self.memory[i][3])
            sample_states_prime.append(x_prime)

        return (sample_states, sample_actions, sample_rewards, sample_states_prime)
    
    def clear(self):
        self.memory = np.zeros((max_size, 4))
		

class LinearQNetworkFixed(DQNAgent):
	
    def __init__(self,
    		     q_network,
    		     preprocessor,
    		     memory,
    		     policy,
    		     gamma,
    		     target_update_freq,
    		     num_burn_in,
    		     train_freq,
    		     batch_size): 
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.policy = policy
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.num_actions = 6
        self.weights = []


    def compile(self, loss_func='mse', optimizer=keras.optimizers.Adam()):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Ada#m
        optimizer. 
        """
        (self.q_network).add(keras.layers.Dense(self.num_actions, input_dim=28224, activation = 'linear'))
        (self.q_network).compile(loss=loss_func, optimizer=keras.optimizers.Adam())
        weights = []
        for i in range(0, len(self.q_network.layers)):
            weights.append(self.q_network.layers[i].get_weights)
        
    def calc_q_values(self, state, predict_next_state=False):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        weights_backup = []
        prediction = []
        if(predict_next_state == True):
            for i in range(0, len(self.q_network.layers)):
                weights_backup.append(self.q_network.layers[i].get_weights())
            for i in range(0, len(self.q_network.layers)):
                self.q_network.layers[i].set_weights(self.weights[i])
            prediction = self.q_network.predict(self.flatten_for_network(state))
            for i in range(0, len(weights_backup)):
                self.q_network.layers[i].set_weights(weights_backup[i]) 
        else:
            prediction = self.q_network.predict(self.flatten_for_network(state))
        return prediction

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(state)
        return self.policy.select_action(q_values)		

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass

    def fit(self, env, num_iterations = 1000000, max_episode_length=1000):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """        
        exp_replay= True

        if exp_replay:
            for i in range(0, num_iterations):
                length = 0
                losses = 0
                done = False
                state = env.reset()
                self.preprocessor.reset()
                while(done == False & length < max_episode_length):
                    net_state_current = self.preprocessor.preprocess_for_network(state)
                    action = self.select_action(net_state_current)
                    new_state, reward, done, info = env.step(action)
                    #net_state_next = self.preprocessor.preprocess_for_network(net_state_next)
                    mem_state = self.preprocessor.preprocess_for_memory(state)
                    self.memory.append(mem_state, action, reward) #added to replay
                    batch_size_num = 100
                    (net_current_batch, actions_set, rewards, net_next_batch) = self.memory.sample(batch_size_num)
                    if(batch_size_num > len(actions_set)): 
                        continue #wait until we have atleast 100 samples
                    net_current_batch_flat = []
                    target_batch_f = np.zeros((len(actions_set), self.num_actions))
                    net_current_batch_flat = np.zeros((len(actions_set),84*84*4))
                    for j in range(0, len(actions_set)):
                        net_state_current = net_current_batch[j]
                        net_state_next = net_next_batch[j]
                        action = actions_set[j]
                        reward = rewards[j]
                        output_qvals = self.calc_q_values(net_state_next)
                        target_f = self.calc_q_values(net_state_current)
                        target_f[0][action] = reward + self.gamma*max(output_qvals[0])
                        net_current_batch_flat[j] = (self.flatten_for_network(net_state_current))
                        target_batch_f[j] = (self.flatten_for_network(target_f))
                    blah = self.q_network.fit(net_current_batch_flat, target_batch_f, batch_size=batch_size_num, epochs=1, verbose=1, callbacks=[keras.callbacks.History()], initial_epoch=0)
                    losses = losses + (blah.history['loss'][0])
                    state = new_state
                    length = length+1
                    if(length%self.target_update_freq == 0):
                       self.weights = []
                       for i in range(0, len(self.q_network.layers)):
                            self.weights.append(self.q_network.layers[i].get_weights())
                        
                print i, " : ", losses/length        
                #net_state is the phi, with four frames
        else:
            for i in range(0, num_iterations):
                length = 0
                losses = 0
                done = False
                state = env.reset()
                self.preprocessor.reset()
                while(done == False & length < max_episode_length):
                    net_state_current = self.preprocessor.preprocess_for_network(state)
                    action = self.select_action(net_state_current)
                    new_state, reward, done, info = env.step(action)
                    length = length+1
                    #env.render()
                    mem_state = self.preprocessor.preprocess_for_memory(new_state)
                    self.memory.append(mem_state, action, reward) #added to replay     
                    net_state_next = self.preprocessor.preprocess_for_network(new_state)
                    output_qvals = self.calc_q_values(net_state_next)
                    target_f = self.calc_q_values(net_state_current)
                    target_f[0][action] = reward + self.gamma*max(output_qvals[0])
                    blah = self.q_network.fit(self.flatten_for_network(net_state_current), self.flatten_for_network(target_f), epochs=1, verbose=0, callbacks=[keras.callbacks.History()], initial_epoch=0)
                    losses = losses + (blah.history['loss'][0])
                    state = new_state
                    #net_state is the phi, with four frames
                print i, " : ", losses/length
    
    def flatten_for_network(self, array):
        shape = array.shape
        total_elem = 1
        for i in shape:
            total_elem = total_elem*i
        array = np.reshape(array,(1, total_elem))
        return array
    
    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass