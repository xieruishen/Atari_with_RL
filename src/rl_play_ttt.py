import reinforcement_learning_model as avm
import tic_tac_toe as ttt







if __name__ == "__main__":

	ttt = ttt.TicTacToe()
	rl_agent = avm.ActionValueModel()


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

		if ttt.won == '0' or ttt.won == ' ': reward = -1
		if ttt.won == 'X': reward = 1

		return (reward, valid)


	prior_state_actions = []

	for i in range(100000):
		while ttt.won is None:
			prior_state_actions = rl_agent.take_action(envir, "".join(ttt.board), actions, prior_state_actions)

		ttt.reset()
		prior_state_actions = []
				
	for board, actions in rl_agent.Q_a_s.items():
		print(board + ":")
		for actions, action in actions.items():
			print("    " + str(actions) + " " +str(action))



	





