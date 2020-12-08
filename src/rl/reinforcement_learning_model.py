import random
import pickle

class ActionValueModel:

	def __init__(self, epsilon = 0.2, step_size = .5, discounting_factor = lambda a : .5**a):

		self.Q_a_s = {}
		self.N_a_s = {}
		self.epsilon = epsilon
		self.step_size = step_size
		self.discounting_factor = discounting_factor
		random.seed()


	def take_action(self, environment, state, actions, prior_state_actions, log = True):
		
		action = None

		#check if we've seen this state before
		if self.Q_a_s.get(state) is not None:  

			#randomly sample greedy vs random action
			if random.random() > self.epsilon:
				#take greedy action

				# get the dictionary of actions for our current state
				Q_a = self.Q_a_s[state]

				# find the action that maximizes q
				max_Q = 0
				
				for past_action, reward in Q_a.items():
					if reward > max_Q:
						max_Q = reward
						action = past_action

		if action is None:
			#take random action

			num_actions = len(actions)
			action = actions[int(random.random()*num_actions)]



		#take action and get reward
		reward, valid = environment(*action)

		#add action and state into our list of prior actions
		if valid:
				prior_state_actions.append((state, action))

				#if we're logging every reward, back propogate the given reward
				if log: self.back_propogate_reward(reward, prior_state_actions)

		return prior_state_actions
		
	def back_propogate_reward(self, reward, prior_state_actions):
		gamma = len(prior_state_actions)

		for prior_state, prior_action in prior_state_actions:
			prior_Q = None
			prior_dict = self.Q_a_s.get(prior_state)
			if prior_dict: prior_Q = prior_dict.get(prior_action)

			if prior_Q:
				self.Q_a_s[prior_state][prior_action] = (prior_Q*self.N_a_s[prior_state][prior_action] + self.step_size * self.discounting_factor(gamma) * (reward - prior_Q))/(self.N_a_s[prior_state][prior_action] +1)
				self.N_a_s[prior_state][prior_action] += 1 
			elif prior_dict:
				self.Q_a_s[prior_state][prior_action] = self.step_size * self.discounting_factor(gamma) * reward
				self.N_a_s[prior_state][prior_action] = 1
			else:
				self.Q_a_s[prior_state] = {prior_action : self.step_size * self.discounting_factor(gamma) * reward}
				self.N_a_s[prior_state] = {prior_action : 1}
			gamma -= 1

	def save_learning(self, filename):
		f_Q = open(filename+"Q.pkl", "wb")
		f_N = open(filename+"N.pkl", "wb")
		pickle.dump(self.Q_a_s, f_Q)
		pickle.dump(self.N_a_s, f_N)
		f_Q.close()
		f_N.close()

	def open_learning(self, filename):
		f_Q = open(filename+"Q.pkl", "rb")
		f_N = open(filename+"N.pkl", "rb")
		self.Q_a_s = pickle.load(f_Q)
		self.N_a_s = pickle.load(f_N)
		f_Q.close()
		f_N.close()


