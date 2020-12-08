import reinforcement_learning_model as avm
import tic_tac_toe as ttt

import matplotlib.pyplot as plt







if __name__ == "__main__":

	ttt = ttt.TicTacToe()
	rl_agent_0 = avm.ActionValueModel(epsilon = 0.5)
	rl_agent_X = avm.ActionValueModel(epsilon = 0.01)


	actions_X = [ (0, 'X'),
				(1, 'X'),
				(2, 'X'),
				(3, 'X'),
				(4, 'X'),
				(5, 'X'),
				(6, 'X'),
				(7, 'X'),
				(8, 'X'),]

	actions_0 = [ (0, '0'),
				(1, '0'),
				(2, '0'),
				(3, '0'),
				(4, '0'),
				(5, '0'),
				(6, '0'),
				(7, '0'),
				(8, '0'),]







	def envir(pos, X_0):

		valid = ttt.play(pos, X_0)

		return (0, valid)



	def train(iterations):
		prior_state_actions_X = []
		prior_state_actions_0 = []

		for i in range(iterations):
			while ttt.won is None:
				
				prior_state_actions_X = rl_agent_X.take_action(envir, "".join(ttt.board), actions_X, prior_state_actions_X)
				prior_state_actions_0 = rl_agent_X.take_action(envir, "".join(ttt.board), actions_0, prior_state_actions_0)

			if ttt.won == 'X':
				rl_agent_X.back_propogate_reward(1, prior_state_actions_X)
				rl_agent_0.back_propogate_reward(-1, prior_state_actions_0)

			if ttt.won == '0':
				rl_agent_X.back_propogate_reward(-1, prior_state_actions_X)
				rl_agent_0.back_propogate_reward(1, prior_state_actions_0)	

			else: 
				rl_agent_X.back_propogate_reward(-1, prior_state_actions_X)
				rl_agent_0.back_propogate_reward(-1, prior_state_actions_0)	
			
			ttt.reset()
			prior_state_actions_X = []
			prior_state_actions_0 = []



	def test(sample_size):
		X_wins = 0
		draws = 0
		prior_state_actions_X = []
		prior_state_actions_0 = []

		for i in range(sample_size):
			while ttt.won is None:
				
				prior_state_actions_X = rl_agent_X.take_action(envir, "".join(ttt.board), actions_X, prior_state_actions_X)
				prior_state_actions_0 = rl_agent_X.take_action(envir, "".join(ttt.board), actions_0, prior_state_actions_0)
			if ttt.won == '0': X_wins += 1
			elif ttt.won == ' ': draws +=1

			ttt.reset()
			prior_state_actions_X = []
			prior_state_actions_0 = []
		return (X_wins/sample_size)
	

	win_percentage = [test(100)]

	for i in range(1000):
		win_percentage.append(test(50))
		train(1)	

	plt.plot(win_percentage)

	plt.show()

	action_pairs = []
	for action in rl_agent_X.Q_a_s["         "]:
		action_pairs.append((action[0], rl_agent_X.Q_a_s["         "][action]))
	action_pairs.sort()	
	print(action_pairs)


	





