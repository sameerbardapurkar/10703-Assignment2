from dqn import DQNAgent
from core import ReplayMemory
from dqn import DQNAgent
from core import *
from policy import *
from preprocessors import *
from objectives import *

class LinearReplayMemory(ReplayMemory):
	def __init__(self, max_size, window_length):
		self.max_size = max_size
		self.window_length = window_length
		self.Memory = [0]*max_size
		self.InsertIndex = 0
		self.final_state = 0
		self.is_terminal = 0
		self.state_count = 0

	def append(self, state, action, reward):
		Memory[InsertIndex%max_size] = (state_count, state, action, reward,
							   state_count + 1)
		InsertIndex = InsertIndex + 1
		state_count = state_count + 1

	def end_episode(self, final_state, is_terminal):
		self.final_state = final_state
		self.is_terminal = is_terminal

	def sample(self, batch_size, indexes=None):
		samples = []
		start_location = 0
		end_location = max_size - 1
		if(InsertIndex >= max_size):
			start_location = InsertIndex%max_size
			end_location = start_location - 1
		else:
			start_location = 0
			end_location = InsertIndex - 1

		stop = False
		location = start_location
		count = 0
		while(stop == False && len(samples) <= batch_size):
			terminal = False
			(state_count, state, action, reward, state_succ_count) = 
				self.Memory[location]
			location = (location + 1)%max_size
			(succ_state_count, succ_state, succ_action, succ_reward,
				succ_state_succ_count) = self.Memory[location]
			if(succ_state_count != state_succ_count):
				print "Warning, invalid successor"
			samples.append(Sample(state, action, reward, succ_state, terminal))
			if(location == end_location):
				stop = True
				if(succ_state != self.final_state):
					print "Warning, memory appears to be corrupted"
				else:
					terminal = True
					samples.append(Sample(succ_state, succ_action, succ_reward,
										  succ_state, terminal))
		return samples

	def clear(self):
		self.Memory = [0]*max_size
		self.InsertIndex = 0
		self.final_state = 0
		self.is_terminal = 0
		self.state_count = 0			

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
		pass

	def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        pass

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
		return (self.q_network).predict(state)        

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
        epsilon = 0.3; # hardcoded for now
		return policy.GreedyEpsilonPolicy(self.num_actions, epsilon)		

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

    def fit(self, env, num_iterations, max_episode_length=None):
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
        pass

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