import reinforcement_learning_model as avm
import mancala as mancala

import matplotlib.pyplot as plt







if __name__ == "__main__":

	manc = mancala.Mancala()

	#create an instance of our rl agent without any discounting factor (we don't care if we win in 5 steps or 10)
	rl_agent = avm.ActionValueModel(epsilon = 0.2, discounting_factor = lambda a : 1)
	

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



	def envir(pos, top_bottom):
		
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
			manc.dumb_ai(switch[top_bottom])

		#if it's a valid move that ended in a free turn, just return true 
		elif valid_state == 2:
			valid = True


		return (reward, valid)


	def train(iterations):

		prior_state_actions = []

		for i in range(iterations):

			#while there's no winner, take action
			while manc.won is None:
				prior_state_actions = rl_agent.take_action(envir, "".join(str(manc.board)), actions_top, prior_state_actions, log = False)
			
			#if we won, our reward is the number of stones in our pool.
			if manc.won == 'top': rl_agent.back_propogate_reward(manc.board[0], prior_state_actions)
			
			#and if we lost, our reward is -1 * the number of stones in our oponent's pool
			else: rl_agent.back_propogate_reward(-1 * manc.board[7], prior_state_actions)
			
			#reset the board and our list of prior states/actions
			manc.reset()
			prior_state_actions = []



	def test(sample_size):
		prior_state_actions = []
		wins = 0

		for i in range(sample_size):

			#while there's no winner, take action
			while manc.won is None:
				prior_state_actions = rl_agent.take_action(envir, "".join(str(manc.board)), actions_top, prior_state_actions, log = False)
			
			#if we won, add one to our wins
			if manc.won == 'top': wins +=1
			
			#reset the board and our list of prior states/actions
			manc.reset()
			prior_state_actions = []

		return (wins/sample_size)


	def test_reward(sample_size):
		prior_state_actions = []
		reward = 0

		for i in range(sample_size):

			#while there's no winner, take action
			while manc.won is None:
				prior_state_actions = rl_agent.take_action(envir, "".join(str(manc.board)), actions_top, prior_state_actions, log = False)
			
			#if we won, add one to our wins
			reward += manc.board[0] - manc.board[7]
			
			#reset the board and our list of prior states/actions
			manc.reset()
			prior_state_actions = []

		return (reward/sample_size)
		

	win_percentage = [test(100)]
	space_mapped = [0]


	iterations = 1000
	for i in range(iterations):
		perc = int(20 * i/iterations)
		loading = "[" + "X" * perc + " " * (20-perc) + "]"
		print(loading, end = "\r")
		win_percentage.append(test(100))
		
		train(100)

		state_action_pairs = 0
		for boards, acts in rl_agent.Q_a_s.items():
			state_action_pairs += len(acts)

		space_mapped.append(state_action_pairs)

	

	plt.subplot(2,1,1)
	plt.plot(win_percentage)
	plt.title("epsilon :" + str(rl_agent.epsilon) + "\nwin win_percentage over 100 rounds vs training iteration\n(1 step = 100 training rounds)")
	plt.subplot(2,1,2)
	plt.title("# of different state/action pairs mapped")
	plt.plot(space_mapped)
	plt.show()


	





