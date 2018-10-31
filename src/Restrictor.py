from src.Board import Board
from src.Applicant import Applicant


class Restrictor:

    def __init__(self, name: str, board_size: int, board: Board, player_one: Applicant, player_two: Applicant):
        self.name = name
        self.board_size = board_size
        self.board = board.board
        self.player_one = player_one
        self.player_two = player_two
        self.selected_stone_row = None
        self.selected_stone_col = None

    def get_stone_position(self, row, col, board: Board, player: Applicant):
        self.selected_stone_row = row
        self.selected_stone_col = col

    def check_valid_actions(self):
        stone_value = self.board[self.selected_stone_row, self.selected_stone_col]
        possible_directions = self._possible_directions()

    def _check_player_based_on_stone(self):
        assert self.board is not None, "Deliver board please!"
        assert self.selected_stone_row is not None, "Deliver row of stone please!"
        assert self.selected_stone_col is not None, "Deliver col of stone pleaase!"
        assert self.board[self.selected_stone_row, self.selected_stone_col] != 0, "Choosen position has no stone!"

        if self.board[self.selected_stone_row, self.selected_stone_col] > 0:
            if self.player_one.side == "down":
                player = self.player_one
            else:
                player = self.player_two
        else:
            if self.player_one.side == "down":
                player = self.player_one
            else:
                player = self.player_two
        return player

    def _check_stone_status(self):
        value = self.board[self.selected_stone_row, self.selected_stone_col]
        if abs(value) == 2:
            status = "dame"
        else:
            status = "normal"

        return status

    def __possible_directions(self):
        # a stone can only go diagonally
        row = self.selected_stone_row
        col = self.selected_stone_col
        possible_positions = []
        directions = [[-1,1], [1,1], [-1,-1], [1,-1]]
        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if self._check_if_position_on_board(self.board_size, new_row, new_col):
                possible_positions = possible_positions + [[new_row, new_col]]

    @staticmethod
    def _check_if_position_on_board(board_size: int, row: int, col: int):
        in_row = row in range(board_size-1)
        in_col = col in range(board_size-1)

        return in_row & in_col