from dqn import DQNAgent
from core import ReplayMemory
from dqn import DQNAgent
from core import *
from policy import *
from preprocessors import *
import keras
from keras import optimizers
from keras import callbacks
from objectives import *
import matplotlib.pyplot as plt

class LinearReplayMemory(ReplayMemory):

    def __init__(self, max_size=10000, window_length=5):
        self.max_size = max_size
        self.window_length = window_length
        self.memory = [0]*max_size
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
        samples = [0]*batch_size
        
        if(self.ind <= batch_size):
            batch_size = self.ind
        
        nums = range(0, self.ind)
        chosen_nums = random.sample(num, batch_size) 

        for i in chosen_nums:
            samples.append(self.memory[i])

        return samples
    
    def clear(self):
        self.memory = np.zeros((max_size, 4))
		

class LinearQNetwork(DQNAgent):
	
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

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        return self.q_network.predict(self.flatten_for_network(state))      

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
        exp_replay = False

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