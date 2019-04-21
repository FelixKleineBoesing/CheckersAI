from python_backend.src.Board import Board
from python_backend.src.Agent import Agent
from python_backend.src.Helpers import Rewards
from python_backend.src.Helpers import DefaultRewards
import logging


class Game:

    def __init__(self, name: str, player_one: Agent, player_two: Agent,
                 board: Board, rewards: Rewards):
        assert isinstance(player_one, Agent)
        assert isinstance(player_two, Agent)
        assert isinstance(board, Board)
        if rewards is not None:
            assert isinstance(rewards, Rewards)
        else:
            logging.info("No reward function supplied! Default rewards will be used as specified in Helpers.py")
            rewards = DefaultRewards
        self.name = name

        # init players
        self.player_one = player_one
        self.player_two = player_two
        self.winner = None
        self.reward_mapping = rewards

        # init board
        self.board = board
        self.board.init_stones(self.player_one, self.player_two)
        self.board.refresh_board()

        # stats
        self.turns = 0

    def get_action_space(self, player_name: str):
        return self.board.get_all_moves(player_name)

    def play(self, verbose:bool = True):
        finished = False
        turns_without_removed_stone = 0

        while not finished:
            number_stones_before = self.board.number_of_stones()
            self.turns += 1
            if verbose:
                print("Iteration:{}".format(self.turns))
            action_space_p_one = self.get_action_space(self.player_one.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_one) == 0:
                reason = "PLayer one is blocked and can´t move!"
                winner = self.player_one.name
                break
            move, stone_id = self.player_one.play_turn(self.board.board, action_space_p_one)
            reward = self.board.move_stone(move, stone_id)
            self.board.refresh_board()
            state = self.board.board
            if verbose:
                print("--------Player One moved-----------")
                self.board.print_board()
            finished, reason, winner = self.game_finished(turns_without_removed_stone)
            if finished:
                if winner == self.player_one.name:
                    reward += 100
            if finished:
                break

            action_space_p_two = self.get_action_space(self.player_two.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_two ) == 0:
                reason = "PLayer two is blocked and can´t move!"
                winner = self.player_two.name
                break
            move, stone_id = self.player_two.play_turn(self.board.board, action_space_p_two)
            reward = self.board.move_stone(move, stone_id)
            self.board.refresh_board()
            state = self.board.board
            if verbose:
                print("--------Player two moved-----------")
                self.board.print_board()
            number_stones_after = self.board.number_of_stones()
            turns_without_removed_stone = turns_without_removed_stone + 1 if number_stones_after == \
                                                                             number_stones_before else 0
            finished, reason, winner = self.game_finished(turns_without_removed_stone)
            if finished:
                if winner == self.player_two.name:
                    reward += 100
        if verbose:
            print("Game finished with the following message: {}".format(reason))
        self.winner = winner


    def game_finished(self, turns_without_removed_stone: int):
        player_vanished, players = self.board.check_if_player_without_stones()
        if player_vanished:
            return True, "Player {} won!".format(players[0]), players[0]
        elif turns_without_removed_stone >= 20:
            return True, "Draw!", None
        else:
            return False, "", None
