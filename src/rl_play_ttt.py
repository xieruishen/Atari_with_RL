import reinforcement_learning_model as avm
import tic_tac_toe as ttt







if __name__ == "__main__":

	ttt = ttt.TicTacToe()
	rl_agent = avm.ActionValueModel(epsilon = 0.2)


	actions = [ (0, 'X'),
				(1, 'X'),
				(2, 'X'),
				(3, 'X'),
				(4, 'X'),
				(5, 'X'),
				(6, 'X'),
				(7, 'X'),
				(8, 'X'),]






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
		

	for i in range(100):
		print(test(1000))
		train(100)
	

	print(rl_agent.Q_a_s)



	





