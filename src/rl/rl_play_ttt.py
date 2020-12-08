import reinforcement_learning_model as avm
import tic_tac_toe as ttt

import matplotlib.pyplot as plt







if __name__ == "__main__":

	ttt = ttt.TicTacToe()
	rl_agent = avm.ActionValueModel(epsilon = 0.1)


	actions = [ (0, 'X'),
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
		reward = 0

		valid = ttt.play(pos, X_0)
		if valid: ttt.ai_play('0')

		return (reward, valid)


	def train(iterations):
		prior_state_actions = []

		for i in range(iterations):
			while ttt.won is None:
				prior_state_actions = rl_agent.take_action(envir, "".join(ttt.board), actions, prior_state_actions)
			
			if ttt.won == 'X': rl_agent.back_propogate_reward(1, prior_state_actions)
			else: rl_agent.back_propogate_reward(-1, prior_state_actions)
			ttt.reset()
			prior_state_actions = []



	def test(sample_size):
		prior_state_actions = []
		wins = 0

		for i in range(sample_size):
			while ttt.won is None:
				prior_state_actions = rl_agent.take_action(envir, "".join(ttt.board), actions, prior_state_actions)
			if ttt.won == 'X': wins +=1
			ttt.reset()
			prior_state_actions = []
		return (wins/sample_size)
		

	win_percentage = [test(100)]
	space_mapped = [0]

	for i in range(400):
		win_percentage.append(test(100))
		train(1)

		state_action_pairs = 0
		for boards, acts in rl_agent.Q_a_s.items():
			state_action_pairs += len(acts)

		space_mapped.append(state_action_pairs)

	
	

	plt.subplot(2,1,1)
	plt.plot(win_percentage)
	plt.subplot(2,1,2)
	plt.plot(space_mapped)
	plt.show()


	





