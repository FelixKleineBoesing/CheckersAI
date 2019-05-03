import numpy as np

from python_backend.src.Stone import Stone
from python_backend.src.agents.Agent import Agent
from python_backend.src.Helpers import Rewards
from python_backend.src.Helpers import ActionSpace


class Board:

    def __init__(self, board_length: int = 8):
        assert board_length % 2 == 0, "Board length has to be an odd number!"

        self.board_length = board_length
        self.stones = None
        # 0 is no stone placed
        self.board = [[0 for x in range(self.board_length)] for y in range(self.board_length)]
        self.board = np.array(self.board)
        self.id_store = np.array(self.board)

    def init_stones(self, player_one: Agent, player_two: Agent):
        n = 0
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
                            stone = Stone(n, player, np.array((i, j)), "normal", value, self.board_length)
                            stones += [stone]
                            n += 1
                    else:
                        if j % 2 != 0:
                            stone = Stone(n, player, np.array((i, j)), "normal", value, self.board_length)
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
                self.id_store[stone.coord[0], stone.coord[1]] = stone.id

    def get_all_moves(self, player_name: str):
        tmp = {}

        # because a player must jump an enemy stone we proof here if there is any move where a stone can be jumped
        jumping_possible = False
        for stone in self.stones:
            if stone.player.name == player_name and stone.removed == False:
                spec_act_space = stone.get_possible_moves(self.board_length, self.board)
                for move in spec_act_space:
                    if move["jumped_values"] != 0:
                        jumping_possible = True
                tmp[stone.id] = spec_act_space
        arr = np.zeros([self.board_length] * 4)
        action_space = ActionSpace(self.board_length)
        for id in tmp.keys():
            if len(tmp[id]) == 0:
                continue
            if jumping_possible:
                for i in range(len(tmp[id])):
                    if tmp[id][i]["jumped_values"] != 0:
                        action_space.update_dict(id, tmp[id][i])
            else:
                action_space.update_dict(id, tmp[id])
            if id in action_space.keys():
                for move in action_space.space_dict[id]:
                    action_space.update_array(self.stones[id].coord, move["new_coord"])

        return action_space

    def print_board(self):
        print(self.board)

    def move_stone(self, move: dict,  stone_id: int, rewards: Rewards):
        reward_active = 0
        reward_passive = 0

        for coord in move["jumped_stones"]:
            for stone in self.stones:
                if coord[0] == stone.coord[0] and coord[1] == stone.coord[1]:
                    stone.removed = True
                    reward_active += rewards.normal_stone_taken if abs(stone.value) == 1 else rewards.queen_stone_taken
                    reward_passive += rewards.normal_stone_loss if abs(stone.value) == 1 else rewards.queen_stone_loss
                    continue
        for stone in self.stones:
            if stone.id == stone_id:
                stone.coord = move["new_coord"]
                break
        return reward_active, reward_passive

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
                return False, players
        return True, players
