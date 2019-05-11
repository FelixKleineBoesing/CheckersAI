import numpy as np

from python_backend.src.Board import Board
from python_backend.src.agents.Agent import Agent
from python_backend.src.Helpers import Rewards
from python_backend.src.Helpers import default_rewards
from python_backend.src.Helpers import ActionSpace


class Game:

    def __init__(self, name: str, agent_one: Agent, agent_two: Agent,
                 board: Board, rewards: Rewards = default_rewards,
                 save_runhistory: bool = False):
        """
        Game class which takes the players, a board and a reward class
        :param name:
        :param agent_one: agent one  that plays the game
        :param agent_two: agent two that plays the game
        :param board: object of Board class
        :param rewards: object of reward class which defines the rewards the agent get for some actions
        """
        assert isinstance(agent_one, Agent)
        assert isinstance(agent_two, Agent)
        assert isinstance(board, Board)
        assert isinstance(rewards, Rewards)
        self.name = name

        # init players
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.winner = None
        self.rewards = rewards
        self.save_runhistory = save_runhistory
        self.runhistory = []

        # init board
        self.board = board
        self.board.init_stones(self.agent_one, self.agent_two)
        self.board.refresh_board()

        # stats
        self.turns = 0
        self.cum_rewards_agent_one = None
        self.cum_rewards_agent_two = None

    def get_action_space(self, player_name: str):
        return self.board.get_all_moves(player_name)

    def play(self, verbose:bool = True):
        finished = False
        turns_without_removed_stone = 0
        cum_rewards_agent_one = 0
        cum_rewards_agent_two = 0

        while not finished:
            reward_player_one, reward_player_two = 0, 0
            number_stones_before = self.board.number_of_stones()
            self.turns += 1

            if verbose:
                print("Iteration:{}".format(self.turns))
            action_space_p_one = self.get_action_space(self.agent_one.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_one) == 0:
                reason = "Player one is blocked and can´t move!"
                winner = self.agent_two.name
                break
            action = self.agent_one.play_turn(self.board.board, action_space_p_one)
            move, stone_id = self._get_move_and_stone(action, action_space_p_one)
            rpo, rpt = self.board.move_stone(move, stone_id, self.rewards)
            reward_player_one += self.rewards.turn + rpo
            reward_player_two += rpt
            state = self.board.board
            self.board.refresh_board()
            next_state = self.board.board
            if self.save_runhistory:
                self.runhistory.append(next_state.tolist())
            if verbose:
                print("--------Player One moved-----------")
                self.board.print_board()
            finished, reason, winner = self._game_finished(turns_without_removed_stone)
            if finished:
                if winner == self.agent_one.name:
                    reward_player_one += self.rewards.win
                    reward_player_two += self.rewards.loss
            self.agent_one.get_feedback(state, action, reward_player_one, next_state, finished)
            self.agent_two.get_feedback(state, action, reward_player_two, next_state, finished)
            if finished:
                break

            cum_rewards_agent_one += reward_player_one
            cum_rewards_agent_two += reward_player_two
            reward_player_one, reward_player_two = 0, 0
            action_space_p_two = self.get_action_space(self.agent_two.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_two) == 0:
                reason = "Player two is blocked and can´t move!"
                winner = self.agent_one.name
                break
            action = self.agent_two.play_turn(self.board.board, action_space_p_two)
            move, stone_id = self._get_move_and_stone(action, action_space_p_two)
            rpt, rpo = self.board.move_stone(move, stone_id, self.rewards)
            reward_player_two += self.rewards.turn + rpt
            reward_player_one += rpo
            self.board.refresh_board()
            state = self.board.board
            if self.save_runhistory:
                self.runhistory.append(next_state.tolist())
            if verbose:
                print("--------Player two moved-----------")
                self.board.print_board()
            number_stones_after = self.board.number_of_stones()
            turns_without_removed_stone = turns_without_removed_stone + 1 if number_stones_after == \
                                                                             number_stones_before else 0
            finished, reason, winner = self._game_finished(turns_without_removed_stone)
            if finished:
                if winner == self.agent_two.name:
                    reward_player_two += self.rewards.win
                    reward_player_one += self.rewards.loss
            self.agent_two.get_feedback(state, action, reward_player_two, next_state, finished)
            self.agent_one.get_feedback(state, action, reward_player_one, next_state, finished)
            cum_rewards_agent_one += reward_player_one
            cum_rewards_agent_two += reward_player_two

        if verbose:
            print("Game finished with the following message: {}".format(reason))
        self.winner = winner
        self.cum_rewards_agent_one = cum_rewards_agent_one
        self.cum_rewards_agent_two = cum_rewards_agent_one

    def _get_move_and_stone(self, action: np.ndarray, action_space: ActionSpace):
        stone_id = self.board.id_store[action[0], action[1]]
        for move in action_space[stone_id]:
            if move["new_coord"][0] == action[2] and move["new_coord"][1] == action[3]:
                  break

        return move, stone_id

    def _game_finished(self, turns_without_removed_stone: int):
        player_vanished, players = self.board.check_if_player_without_stones()
        if player_vanished:
            return True, "Player {} won!".format(players[0]), players[0]
        elif turns_without_removed_stone >= 40:
            return True, "Draw!", None
        else:
            return False, "", None
