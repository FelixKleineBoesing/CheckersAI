from src.Board import Board
from src.Applicant import Applicant

class Game:

    def __init__(self, name: str):

        self.name = name

        # init players
        self.player_one = Applicant("One", "up")
        self.player_two = Applicant("Two", "down")

        # init board
        self.board = Board(board_length=8)
        self.board.init_stones(self.player_one, self.player_two)
        self.board.refresh_board()
        self.board.print_board()

    def get_board(self):
        pass

    def get_users(self):
        pass

    def play(self):
        pass

    

if __name__=="__main__":
    game = Game("Test")

