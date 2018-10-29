from src.Board import Board
from src.Applicant import Applicant

class Restrictor:

    def __init__(self, name: str, board_size: int):
        self.name = name
        self.board_size = board_size

    def get_needed_informations(self, row, col, board: Board, player: Applicant):
        self.row = row
        self.col = col
        self.board = board
        self.player = player

    def check_valid_actions(self):
        stone_value = self.board.board[self.row, self.col]

        if abs(stone_value) > 10:
            pass
        else:
            pass





    def _check_side(self):
        pass
