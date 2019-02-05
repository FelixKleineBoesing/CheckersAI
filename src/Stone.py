import numpy as np

from src.Applicant import Applicant
from typing import Tuple
coord = Tuple[int, int]


class Stone:

    def __init__(self,id: int,  player: Applicant, coord: coord, status: str, value: int):
        self.id = id
        self.player = player
        self.start_row = coord[0]
        self.coord = coord
        self.status = status
        self.value = value
        self.removed = False

    def get_possible_moves(self, board_side: int, board: np.ndarray):
        moves = {}
        move = 0
        if abs(self.value) == 1:
            if self.start_row <= 2:
                directions = [(1, -1), (1, 1)]
            else:
                directions = [(-1, 1), (-1, -1)]
        else:
            directions = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        for direction in directions:
            within_board = True
            i = 1
            while(within_board):
                coord = self.coord + direction * i
                value_board = board[coord[0], coord[1]]
                within_board = self._check_if_position_on_board(coord, board_side)
                # break if first step is already out of board
                if not within_board:
                    break
                # break if there is a stone of them same player in the way
                if value_board < 0 & self.value < 0 or value_board > 0 & self.value > 0:
                    break
                # if there is no stone, than add this to move list.
                if board[coord[0], coord[1]] == 0:
                    moves[move] = {"new_coord": coord, "jumped_stones": [], "jumped_values": 0}
                    move += 1
                # if there if a stone of the enemy
                if value_board < 0 & self.value > 0 or value_board < 0 & self.value > 0:
                    # check if it can be jumped
                    coord_jump = coord + direction
                    within_board_after_jump = self._check_if_position_on_board(coord_jump, board_side)
                    if within_board_after_jump:
                        moves[move] = {"new_coord": coord, "jumped_stones": [coord], "jumped_values": 0}

                i += 1
                # break if normal stone, because they can only move one field
                if abs(self.value) == 1:
                    break

    def jump_enemy_stone(self, directions: list, board: np.ndarray, coord: tuple, board_size: int):
        for direction in directions:
            coord = coord + direction
            if not self._check_if_position_on_board(coord, board_size=board_size):
                continue






    def move_stone(self):
        pass


    @staticmethod
    def _check_if_position_on_board(coord: tuple, board_size: int):
        in_row = coord[0] in range(board_size - 1)
        in_col = coord[1] in range(board_size - 1)
        return in_row & in_col

