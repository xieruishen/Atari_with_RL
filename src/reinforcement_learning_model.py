import random

class ActionValueModel:

	def __init__(self, epsilon = 0.2, discounting_factor = lambda a : .5**a):

		self.Q_a_s = {}
		self.epsilon = epsilon
		self.step_size = .5
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
				
				for action_option in actions:
					q = Q_a.get(action_option) 
					if q is not None and q > max_Q:
						max_q = Q_a[action_option]
						action = action_option

		if action is None:
			#take random action

			num_actions = len(actions)
			action = actions[int(random.random()*num_actions)]



		#take action and get reward
		reward, valid = environment(*action)

		#add action and state into our list of prior actions
		if valid:
				prior_state_actions.append((state, action))
				if log : self.back_propogate_reward(reward, prior_state_actions)

		return prior_state_actions
		
	def back_propogate_reward(self, reward, prior_state_actions):
		gamma = len(prior_state_actions)

		for prior_state, prior_action in prior_state_actions:
			prior_Q = None
			prior_dict = self.Q_a_s.get(prior_state)
			if prior_dict: prior_Q = prior_dict.get(prior_action)

			if prior_Q:
				self.Q_a_s[prior_state][prior_action] = prior_Q + self.step_size * self.discounting_factor(gamma) * (reward - prior_Q)
			elif prior_dict:
				self.Q_a_s[prior_state][prior_action] = self.step_size * self.discounting_factor(gamma) * reward
			else:
				self.Q_a_s[prior_state] = {prior_action : self.step_size * self.discounting_factor(gamma) * reward}
			gamma -= 1





