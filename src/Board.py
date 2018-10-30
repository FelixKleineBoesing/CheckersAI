import numpy as np

from src.Stone import Stone
from src.Applicant import Applicant

class Board:

    def __init__(self, player_one: Applicant, player_two: Applicant, board_length: int = 8):
        assert board_length % 2 == 0, "Board length has to be an odd number!"

        self.board_length = board_length
        # 0 is no stone placed
        self.board = [[0 for x in range(self.board_length)] for y in range(self.board_length)]
        self.board = np.array(self.board)
        stones = {}
        for i in range(self.board_length):
            if i <= 2:
                player = player_one
            else:
                player = player_two
            if i <= 2 or i >= self.board_length-3:
                for j in range(self.board_length):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            self.board[i, j] = 1 if player == player_one else -1
                    else:
                        if j % 2 != 0:
                            self.board[i, j] = -1 if player == player_two else
            self.stones = stones

    def print_board(self):
        print(self.board)

    def place_stone(self, from_row: int, from_col: int, to_row: int, to_col: int, player: int):
        self.board[from_row, from_col] = 0
        self.board[to_row, to_col] = 1 if player == 1 else -1

    def remove_stone(self, row, col):
        self.board[row, col] = 0




if __name__ == "__main__":
    board = Board(board_length=8)
    board.print_board()
