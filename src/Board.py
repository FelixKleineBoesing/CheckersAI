import numpy as np


class Board:

    def __init__(self, board_length: int = 8):
        assert board_length % 2 == 0, "Board length has to be an odd number!"

        self.board_length = board_length
        # 0 is no stone placed
        self.board = [[0 for x in range(self.board_length)] for y in range(self.board_length)]
        self.board = np.array(self.board)

        for i in range(self.board_length):
            if i <= 2:
                player = 1
            else:
                player = -1
            if i <= 2 or i >= self.board_length-3:
                for j in range(self.board_length):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            self.board[i,j] = player
                    else:
                        if j % 2 != 0:
                            self.board[i,j] = player

    def print_board(self):
        print(self.board)

    def place_stone(self):
        pass

    def remove_stone(self):
        pass


if __name__ == "__main__":
    board = Board(board_length=8)
