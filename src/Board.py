import numpy as np


class Board:

    def __init__(self, board_length: int = 4):

        # 0 is no stone placed
        board = [[0 for x in range(board_length)] for y in range(board_length)]
        board = np.array(board)

        

    def print_board(self):
        pass

    def place_stone(self):
        pass

    def remove_stone(self):
        pass


if __name__ == "__main__":
    board = Board(board_length=8)
