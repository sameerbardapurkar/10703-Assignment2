from dqn import DQNAgent

class LinearQNetwork(DQNAgent):
	def __init(self,
			   q_network,
			   preprocessor,
			   memory,
			   policy,
			   gamma,
			   target_update_freq,
			   num_burn_in,
			   train_freq,
			   batch_size):
	
