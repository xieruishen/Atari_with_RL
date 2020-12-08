import reinforcement_learning_model as avm
import mancala as mancala

import matplotlib.pyplot as plt








if __name__ == "__main__":

	manc = mancala.Mancala()

	#create an instance of our rl agent without any discounting factor (we don't care if we win in 5 steps or 10)
	rl_agent_top = avm.ActionValueModel(epsilon = 0.2, discounting_factor = lambda a : 1)
	rl_agent_bottom = avm.ActionValueModel(epsilon = 0.2, discounting_factor = lambda a : 1)
	

	#map out valid actions for each player
	actions_bottom = [(1, "bottom"),
				(2, "bottom"),
				(3, "bottom"),
				(4, "bottom"),
				(5, "bottom"),
				(6, "bottom")]

	actions_top = [(7, "top"),
				(8, "top"),
				(9, "top"),
				(10, "top"),
				(11, "top"),
				(12, "top"),
				(13, "top"),]

	double = {"top" : False, "bottom" : False}


	def envir(pos, top_bottom):
		
		switch = {"top":"bottom",
				"bottom" : "top"}
		
		#we back propogate the reward at the end, so there's no need to track reward points as we go
		reward = 0
		valid = False

		#try to play the given position 
		valid_state = manc.play(pos, top_bottom)

		#if it's a valid move, set valid to true and set double to false
		if valid_state == 1: 
			valid = True
			double[top_bottom] = False

		#if it's a valid move that ended in a free turn, return true and set our double to true
		elif valid_state == 2:
			valid = True
			double[top_bottom] = True


		return (reward, valid)


	def hacky_user_envir(pos, top_bottom):
		
		switch = {"top":"bottom",
				"bottom" : "top"}
		
		#we back propogate the reward at the end, so there's no need to track reward points as we go
		reward = 0
		valid = False

		#try to play the given position 
		valid_state = manc.play(pos, top_bottom)

		#if it's a valid move, set valid to true and let the ai take a move
		if valid_state == 1: 
			valid = True
			

			ret = 2
			while ret == 2:
				manc.print_board()
				try:
					user_pos = int(input("what's your play?"))
					ret = manc.play(user_pos, "bottom")
				except ValueError:
					print("bad input")
					ret = 2

		#if it's a valid move that ended in a free turn, just return true 
		elif valid_state == 2:
			valid = True


		return (reward, valid)


	def train(iterations):

		prior_state_actions_top = []
		prior_state_actions_bottom = []

		for i in range(iterations):

			#while there's no winner, take action
			while manc.won is None:

				if not double["bottom"]:
					prior_state_actions_top = rl_agent_top.take_action(envir, "".join(str(manc.board)), actions_top, prior_state_actions_top, log = False)
				
				if not double["top"]:
					prior_state_actions_bottom = rl_agent_bottom.take_action(envir, "".join(str(manc.board)), actions_bottom, prior_state_actions_bottom, log = False)

			#if we won, our reward is the number of stones in our pool.
			if manc.won == 'top':
				rl_agent_top.back_propogate_reward(manc.board[0], prior_state_actions_top)
				rl_agent_bottom.back_propogate_reward(-1 * manc.board[7], prior_state_actions_bottom)
			#and if we lost, our reward is -1 * the number of stones in our oponent's pool
			elif manc.won == 'bottom': 
				rl_agent_top.back_propogate_reward(-1 * manc.board[7], prior_state_actions_top)
				rl_agent_bottom.back_propogate_reward(manc.board[0], prior_state_actions_bottom)
			#reset the board and our list of prior states/actions
			manc.reset()
			prior_state_actions_top = []
			prior_state_actions_bottom = []


	rl_agent_top.open_learning("mancala_RL")

	space_mapped =[]
	iterations = 10
	for i in range(iterations):
		perc = int(20 * i/iterations)
		loading = "[" + "X" * perc + " " * (20-perc) + "]"
		print(loading, end = "\r")
		
		train(100)

		state_action_pairs = 0
		for boards, acts in rl_agent_top.Q_a_s.items():
			state_action_pairs += len(acts)

		space_mapped.append(state_action_pairs)

	rl_agent_top.save_learning("mancala_RL")
	
	

	plt.title("epsilon :" + str(rl_agent_top.epsilon) + "\n# of different state/action pairs mapped")
	plt.plot(space_mapped)
	plt.show()



	
	prior_state_actions =[]
	manc.epsilon = 0
	while manc.won is None:
				prior_state_actions = rl_agent_top.take_action(hacky_user_envir, "".join(str(manc.board)), actions_top, prior_state_actions, log = False)
				if len(prior_state_actions) !=0 :
					state, action = prior_state_actions[-1]
					print(rl_agent_top.Q_a_s[state][action])

	





