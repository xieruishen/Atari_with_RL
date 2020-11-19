import random

class value_function_RL:

	def __init__(self):

		self.Q_a_s = {}
		self.N_a_s = {}
		self.N_a = 0
		self.epsilon = 0.2
		random.seed()


	def step_size(self, n):
		return .5


	def take_action(self, environment, state, actions, prior_state_actions):
		
		action = None

		#check if we've seen this state before
		if state in Q_s_a:

			#randomly sample greedy vs random action
			if random.random() > self.epsilon:
				#take greedy action

				# get the dictionary of actions for our current state
				Q_a = self.Q_a_s[state]


				# find the action that maximizes q
				max_Q = 0
				for action_option in actions:
					if Q_a[action_option] > max_q:
						max_q = Q_a[actionaction_option]
						action = action_option

		else:
			#take random action

			num_actions = len(actions)
			action = actions[(random.random()*num_actions)]



		#take action and get reward
		reward = environment(action)

		#add action and state into our list of prior actions
		prior_state_actions.append((state, action))

		#back propogate rewards (not sure if this should be called for every action? Do we actually back propogate these rewards?)
		for prior_state, prior_action in prior_state_actions:
			prior_Q = self.Q_a_s.get(prior_state)

			if prior_Q: 
				prior_Q = prior_Q.get(prior_action)

			if prior_Q:
				self.N_a_s[prior_state][prior_action] = self.N_a_s[prior_state][prior_action]+1
				self.Q_a_s[prior_state][prior_action] = prior_Q + self.step_size(self.N_a_s[prior_state][prior_action]) * (reward - prior_Q)
				
			else:
				self.Q_a_s[prior_state][prior_action] = step_size * reward
				self.N_a_s[prior_state][prior_action] = 1




