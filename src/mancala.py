class Mancala:

	def __init__(self):

		'''board indices: 
			 |13|12|11|10| 9| 8|   <- Top player
			0-------------------7
			 | 1| 2| 3| 4| 5| 6|   <- Bottom player
			'''
		self.pairs={1:13,
					2:12, 
					3:11,
					4:10,
					5:9,
					6:8,
					8:6,
					9:5,
					10:4,
					11:3,
					12:2,
					13:1}	
		
		self.pool= {"top":0,
					"bottom":7}

		self.board = [0]*14
		self.board[1:6] = [4]*6
		self.board[8:13] = [4]*6

		self.won = None


	def play(self, pos, top_bottom, display = False):
		#top_bottom determines player. 
		#"top" player can play pos 8-13 and has pool at 0. 
		#"bottom" can play pos 1-6 and has pool at 7.
		

		# check if someone has won, and if the play is valid (stones in the given pocket and on the right side of the board)
		if (self.won == None
			and self.board[pos] > 0
			and ((top_bottom == "bottom" and pos > 0 and pos < 7)
				or (top_bottom == "top" and pos > 7 and pos < 14))):

			#pick up stones from the current position
			stones = self.board[pos]
			self.board[pos] = 0

			#distribute the stones one by one into the pockets ahead  
			while stones > 0:
				pos = (pos+1) % 14
				if (top_bottom == "bottom" and pos != 0) or (top_bottom == "top" and pos != 7):
					self.board[pos] +=1
					stones -= 1
			
			#check if the last stone ended up in an empty pocket
			if self.board[pos] == 1:

				#if this last empty pocket is on the player's side, pick up this stone and the stones on the opposite side and put them in the player's pool 
				if ((top_bottom == "bottom" and pos > 0 and pos < 7)
					or (top_bottom == "top" and pos > 7 and pos < 14)):

					self.board[self.pool[top_bottom]] += 1 + self.board[self.pairs[pos]]
					self.board[pos] = 0
					self.board[self.pairs[pos]] = 0

			self.check_win()
			#if the last empty pocket is the player's pool, they get another turn, so return 2. Otherwise, return 1 to indicate a valid move was made.
			if (pos == self.pool[top_bottom]): return 2
			else: return 1

		#we reach this return if the play isn't valid.
		else:
			return 0


	def reset(self):
		self.board = [0]*14
		self.board[1:6] = [4]*6
		self.board[8:13] = [4]*6

		self.won = None

	

	def check_win(self):
		sum_top = 0
		sum_bottom = 0
		for i in range(6):
			sum_top += self.board[i+8]
			sum_bottom += self.board[i+1]

		if (sum_top == 0 or sum_bottom == 0):

			self.board[self.pool["top"]] += sum_top
			self.board[self.pool["bottom"]] += sum_bottom

			if self.board[self.pool["top"]] > self.board[self.pool["bottom"]]: 
				self.won = "top"
			elif self.board[self.pool["top"]] < self.board[self.pool["bottom"]]:
				self.won = "bottom"
			else:
				self.won = "tie"

			return (self.board[self.pool["top"]], self.board[self.pool["bottom"]])
		
		return None

	def dumb_ai(self, top_bottom):
		i = 1
		around_once = False
		if top_bottom == "top": i += 7 

		valid = self.play(i, top_bottom)
		while not around_once and (valid == 0 or valid == 2):
			i = i + 1
			if i == 14:
				around_once = True
				i = 0

			valid = self.play(i, top_bottom)


	def print_board(self):
		
		board_visual = " |"
		for i in range(13, 7, -1):
			stones = str(self.board[i])
			board_visual += (2-len(stones)) * " " + stones +"|"

		stones = str(self.board[0])
		board_visual += "\n" + str(self.board[0]) + " " * (2-len(stones)) + "----------------- " + str(self.board[7]) + "\n |"
		
		for i in range(1, 7):
			stones = str(self.board[i])
			board_visual += (2-len(stones)) * " " + stones +"|"


		print(board_visual, end = "\n\n")
		if self.won: 
			print(self.won + "\n")
			


if __name__ == "__main__":
	manc = Mancala()
	manc.print_board()

	for i in range(20):
		manc.dumb_ai("bottom")
		manc.print_board()

