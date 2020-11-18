

class TicTacToe:

	def __init__(self):

		'''board indices: 
			0|1|2
			-----
			3|4|5
			-----
			6|7|8
			'''

		self.board = [" "]*9

		self.win_states = [
			(0,1,2), 
			(3,4,5),
			(6,7,8),
			(0,3,6),
			(1,4,7),
			(2,5,8),
			(0,4,8), 
			(2,4,6)]

		self.won = None


	def play(self, pos, X_0, display = False):
		#pos is 0-9
		#X_0 should be either "X" or "0"

		if self.won == None and pos >= 0 and pos < 9 and (X_0 == "X" or X_0 == "0") and self.board[pos] == " ":
			self.board[pos] = X_0

			self.won = self.check_win()
			if display: self.print_board()
			return True

		else: return False

	
	def check_win(self):
		for state in self.win_states:
			if self.board[state[0]] != " ":
				if (self.board[state[0]] == self.board[state[1]]) and (self.board[state[1]] == self.board[state[2]]):
					return self.board[state[0]]
		return None


	def print_board(self):
		board_visual = (
			self.board[0] + "|" + self.board[1] + "|" + self.board[2] + 
			"\n-----\n" + 
			self.board[3] + "|" + self.board[4] + "|" + self.board[5] + 
			"\n-----\n" + 
			self.board[6] + "|" + self.board[7] + "|" + self.board[8] + "\n")
		print(board_visual)
		if self.won: print(self.won)


	def ai_play(self, X_0):
		for index, pos in enumerate(self.board):
			if pos == " ":
				return self.play(index, X_0)
		return False

