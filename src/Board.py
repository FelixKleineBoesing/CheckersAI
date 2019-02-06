import numpy as np
from functools import reduce

from src.Stone import Stone
from src.Applicant import Applicant


class Board:

    def __init__(self, board_length: int = 8):
        assert board_length % 2 == 0, "Board length has to be an odd number!"

        self.board_length = board_length
        self.stones = None
        # 0 is no stone placed
        self.board = [[0 for x in range(self.board_length)] for y in range(self.board_length)]
        self.board = np.array(self.board)

    def init_stones(self, player_one: Applicant, player_two: Applicant):
        n = 1
        stones = []
        for i in range(self.board_length):
            if i <= 2:
                player = player_one
                value = 1
            else:
                player = player_two
                value = -1
            if i <= 2 or i >= self.board_length - 3:
                for j in range(self.board_length):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            stone = Stone(n, player, np.array((i, j)), "normal", value)
                            stones += [stone]
                            n += 1
                    else:
                        if j % 2 != 0:
                            stone = Stone(n, player, np.array((i, j)), "normal", value)
                            stones += [stone]
                            n += 1
        self.stones = stones

    def refresh_board(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                self.board[i, j] = 0
        for stone in self.stones:
            if not stone.removed:
                self.board[stone.coord[0], stone.coord[1]] = stone.value

    def get_all_moves(self, player_name: str):
        action_space = {}
        for stone in self.stones:
            if stone.player.name == player_name:
                spec_act_space = stone.get_possible_moves(self.board_length, self.board)
                action_space[stone.id] = spec_act_space
        return action_space

    def print_board(self):
        print(self.board)

    def place_stone(self, from_row: int, from_col: int, to_row: int, to_col: int, player: int):
        self.board[from_row, from_col] = 0
        self.board[to_row, to_col] = 1 if player == 1 else -1

    def remove_stone(self, row, col):
        self.board[row, col] = 0

    def number_of_stones(self):
        number_stones = 0
        for stone in self.stones:
            number_stones += stone.removed
        return number_stones

    def check_if_player_without_stones(self):
        # returns true if one player has no stones.
        players = []
        for stone in self.stones:
            if not stone.removed:
                if stone.player.name in players:
                    continue
                else:
                    players += [stone.player.name]
            if len(players) == 2:
                return False
        return True

