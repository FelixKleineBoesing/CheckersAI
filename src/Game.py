from src.Board import Board
from src.Applicant import Applicant

class Game:

    def __init__(self, name: str):

        self.name = name

        # init players
        self.player_one = Applicant("One", "up")
        self.player_two = Applicant("Two", "dpwn")

        # init board
        self.board = Board(board_length=8, )

    def get_board(self):
        pass

    def get_users(self):
        pass

    def play(self):
        pass

    

