import numpy as np
import copy

from checkers.agents.Agent import Agent


class Stone:

    def __init__(self, id: int,  player: Agent, coord: np.ndarray, status: str, value: int,
                 board_size: int):
        self.id = id
        self.player = player
        self.start_row = coord[0]
        self._coord = coord
        self.status = status
        self.value = value
        self.removed = False
        self.board_size = board_size

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, coord):
        self._coord = coord
        middle = self.board_size / 2
        # if stones starts on the upper side, but get to the lower side
        if self.start_row < middle and self.coord[0] == self.board_size - 1 and abs(self.value) == 1:
            self.value = self.value * 2
        # if stones starts on the lower side, but get to the upper side
        if self.start_row > middle and self.coord[0] == 0 and abs(self.value) == 1:
            self.value = self.value * 2

    def get_possible_moves(self, board: np.ndarray):
        """
        implements the move rule set of checkers, since the possible moves are depended of the status of the chosen
        stone (whether its normal or a check)
        :param board:
        :return:
        """
        board_size = board.shape[0]
        moves = []
        if abs(self.value) == 1:
            if self.start_row <= 2:
                directions = [np.array((1, -1)), np.array((1, 1))]
            else:
                directions = [np.array((-1, 1)), np.array((-1, -1))]
        else:
            directions = [np.array((-1, 1)), np.array((1, 1)), np.array((-1, -1)), np.array((1, -1))]
        for direction in directions:
            within_board = True
            i = 1
            while within_board:
                coord = self.coord + direction * i
                within_board = _check_if_position_on_board(coord, board_size)
                # break if first step is already out of board
                if not within_board:
                    break
                value_board = board[coord[0], coord[1]]
                # break if there is a stone of them same player in the way
                if value_board < 0 and self.value < 0 or value_board > 0 and self.value > 0:
                    break
                # if there is no stone, than add this to move list.
                if value_board == 0:
                    moves += [{"old_coord": self.coord, "new_coord": coord, "jumped_stones": [], "jumped_values": 0,
                               "move_coords": [coord]}]
                # if there is a stone of the enemy
                if (value_board < 0 < self.value) or (self.value < 0 < value_board):
                    # check if it can be jumped
                    coord_jump = coord + direction
                    move_coords = [coord_jump.copy()]
                    within_board_after_jump = _check_if_position_on_board(coord_jump, board_size)
                    # break if place behind stone is out of border
                    if not within_board_after_jump:
                        break
                    value_board_jump = board[coord_jump[0], coord_jump[1]]
                    jumped_stones = []
                    # break if there is no free place
                    if value_board_jump != 0:
                        break
                    jumped_stones += [coord]
                    moves_tmp = self.jump_chain(directions, board, coord_jump, value_board, jumped_stones, move_coords)
                    if len(moves_tmp) > 0:
                        moves += moves_tmp
                    else:
                        moves += [{"old_coord": self.coord, "new_coord": coord_jump, "jumped_stones": jumped_stones,
                                   "jumped_values": abs(value_board), "move_coords": [coord_jump]}]
                i += 1
                # break if normal stone, because they can only move one field
                if abs(self.value) == 1:
                    break
        return moves

    def jump_chain(self, directions: list, board: np.ndarray, coord: tuple, value_jumped: int, jumped_stones: list,
                   move_coords: list):
        """
        checks whether there could be a stone be jumped over in a while since there may be another stone which can
        be jumped over after the first stone is jumped over
        :param directions: marks the directions a stone is able to jump. direction is delivered as -1 and 1 for rows
        and columns of board
        :param board: board representation as numpy array
        :param coord: actual coordinates
        :param value_jumped: value of stone that is jumped over before
        :param jumped_stones: coordinates of already jumped stones
        :return:
        """
        board_size = board.shape[0]
        moves = []
        coord_original = coord
        for direction in directions:
            # copy jumped stone list
            jumped_stones_tmp = copy.deepcopy(jumped_stones)
            within_board = True
            i = 1
            tmp_moves = []
            # while loop because ladies can move mutiple steps
            while within_board:
                coord = coord_original + direction * i
                within_board = _check_if_position_on_board(coord, board_size)
                # break if first step is already out of board
                if not within_board:
                    break
                value_board = board[coord[0], coord[1]]
                # break if there is a stone of them same player in the way
                if value_board < 0 and self.value < 0 or value_board > 0 and self.value > 0:
                    break
                already_jumped = False
                for jumped_coord in jumped_stones_tmp:
                    if coord[0] == jumped_coord[0] and coord[1] == jumped_coord[1]:
                        already_jumped = True
                        break
                if already_jumped:
                    break
                # if there if a stone of the enemy
                if (value_board < 0 < self.value) or (value_board > 0 > self.value):
                    # check if it can be jumped
                    coord_jump = coord + direction
                    within_board_after_jump = _check_if_position_on_board(coord_jump, board_size)
                    if not within_board_after_jump:
                        break
                    if board[coord_jump[0], coord_jump[1]] != 0:
                        break
                    jumped_stones_tmp += [coord]
                    move_coords += [coord_jump]
                    tmp_jumped_values = abs(value_jumped) + abs(value_board)
                    tmp_moves += self.jump_chain(directions, board, coord_jump, tmp_jumped_values, jumped_stones_tmp,
                                                 move_coords)
                    if len(tmp_moves) == 0:
                        moves += [{"old_coord": self.coord, "new_coord": coord_jump, "jumped_stones": jumped_stones_tmp,
                                   "jumped_values": abs(tmp_jumped_values), "move_coords": move_coords}]
                    else:
                        moves += tmp_moves
                i += 1
                # break if normal stone, because they can only move one field
                if abs(self.value) == 1:
                    break
        return moves


def _check_if_position_on_board(coord: tuple, board_size: int):
    """
    checks whether coordinates are inside of board
    :param coord:
    :param board_size:
    :return:
    """
    in_row = coord[0] in range(board_size)
    in_col = coord[1] in range(board_size)
    return in_row and in_col
