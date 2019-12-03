import numpy as np
import copy

from checkers.src.game.Board import Board
from checkers.src.agents.agent import Agent
from checkers.src.game.GameHelpers import Rewards, default_rewards, ActionSpace


class Game:
    """
    This class is an abstraction of the checkers game. Therefor it gets two agents, a board and a reward-function.
    I know that rewards are not part of the original game, but the actions that may return a reward are deeply
    integrated int game/board/stone-class
    """
    def __init__(self, agent_one: Agent, agent_two: Agent, board: Board, rewards: Rewards = default_rewards,
                 save_runhistory: bool = False):
        """
        Game class which takes the players, a board and a reward class
        :param agent_one: agent one  that plays the game
        :param agent_two: agent two that plays the game
        :param board: object of Board class
        :param rewards: object of reward class which defines the rewards the agent get for some actions
        """
        assert isinstance(agent_one, Agent)
        assert isinstance(agent_two, Agent)
        assert isinstance(board, Board)
        assert isinstance(rewards, Rewards)

        # init players
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.winner = None
        self.rewards = rewards
        self.save_runhistory = save_runhistory
        self.runhistory_states = []
        self.runhistory_actions = []

        # init board
        self.board = copy.deepcopy(board)
        self.board.init_stones(self.agent_one, self.agent_two)
        self.board.refresh_board()

        # stats
        self.turns = 0
        self.cum_rewards_agent_one = None
        self.cum_rewards_agent_two = None

    def get_action_space(self, player_name: str):
        return self.board.get_all_moves(player_name)

    def play(self, verbose: bool = True):
        finished = False
        turns_without_removed_stone = 0
        cum_rewards_agent_one = 0
        cum_rewards_agent_two = 0
        self.runhistory_states.append(self.board.board)
        while not finished:
            reward_player_one, reward_player_two = 0, 0
            number_stones_before = self.board.number_of_stones()
            self.turns += 1
            if verbose:
                print("Iteration:{}".format(self.turns))
            action_space_p_one = self.get_action_space(self.agent_one.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_one) > 0:
                action = self.agent_one.play_turn(self.board.board, action_space_p_one)
                move, stone_id = self._get_move_and_stone(action, action_space_p_one)
                rpo, rpt = self.board.move_stone(move, stone_id, self.rewards)
                reward_player_one += self.rewards.turn + rpo
                reward_player_two += rpt
                state = self.board.board
                self.board.refresh_board()
                next_state = self.board.board
                if self.save_runhistory:
                    self.runhistory_states.append(next_state.tolist())
                    self.runhistory_actions.append(np.array(move["move_coords"]).tolist())
                if verbose:
                    print("--------Player One moved-----------")
                    self.board.print_board()
                number_stones_after = self.board.number_of_stones()
                turns_without_removed_stone = turns_without_removed_stone + 1 if number_stones_after == \
                                                                                 number_stones_before else 0
                finished, reason, winner = self._game_finished(turns_without_removed_stone)
                if finished:
                    if winner == self.agent_one.name:
                        reward_player_one += self.rewards.win
                        reward_player_two += self.rewards.loss
                    else:
                        reward_player_one += self.rewards.loss
                        reward_player_two += self.rewards.win
                self.agent_one.get_feedback(state, action, reward_player_one, next_state, finished)
                self.agent_two.get_feedback(state, action, reward_player_two, next_state, finished)
                cum_rewards_agent_one += reward_player_one
                cum_rewards_agent_two += reward_player_two
            else:
                reason = "Player one is blocked and can´t move!"
                winner = self.agent_two.name
                finished = True
            if finished:
                break

            reward_player_one, reward_player_two = 0, 0
            number_stones_before = self.board.number_of_stones()
            self.turns += 1
            action_space_p_two = self.get_action_space(self.agent_two.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_two) > 0:
                action = self.agent_two.play_turn(self.board.board, action_space_p_two)
                move, stone_id = self._get_move_and_stone(action, action_space_p_two)
                rpt, rpo = self.board.move_stone(move, stone_id, self.rewards)
                reward_player_two += self.rewards.turn + rpt
                reward_player_one += rpo
                state = self.board.board
                self.board.refresh_board()
                next_state = self.board.board
                if self.save_runhistory:
                    self.runhistory_states.append(next_state.tolist())
                    self.runhistory_actions.append(np.array(move["move_coords"]).tolist())
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
                    else:
                        reward_player_one += self.rewards.loss
                        reward_player_two += self.rewards.win
                self.agent_two.get_feedback(state, action, reward_player_two, next_state, finished)
                self.agent_one.get_feedback(state, action, reward_player_one, next_state, finished)
                cum_rewards_agent_one += reward_player_one
                cum_rewards_agent_two += reward_player_two
            else:
                reason = "Player two is blocked and can´t move!"
                winner = self.agent_one.name
                finished = True

        if verbose:
            print("Game finished with the following message: {}".format(reason))
        self.winner = winner
        self.cum_rewards_agent_one = cum_rewards_agent_one
        self.cum_rewards_agent_two = cum_rewards_agent_one

    def _get_move_and_stone(self, action: np.ndarray, action_space: ActionSpace):
        """
        This helper function return the move and stone_id from action space base on the given action tuple
        :param action:
        :param action_space:
        :return:
        """
        stone_id = self.board.id_store[action[0], action[1]]
        for move in action_space[stone_id]:
            if move["new_coord"][0] == action[2] and move["new_coord"][1] == action[3]:
                  break

        return move, stone_id

    def _game_finished(self, turns_without_removed_stone: int):
        """
        checks if game is finished. This is the case if one of the player vanished (loss/win) from board or no stone has been
        taken since at least 25 turns (draw)
        :param turns_without_removed_stone: as it says
        :return:
        """
        player_vanished, players = self.board.check_if_player_without_stones()
        # players is a unique list containing the names of players of not removed stones.
        # Therefore the list contains only the winner at index zero if the other player lost
        if player_vanished:
            return True, "Player {} won!".format(players[0]), players[0]
        elif turns_without_removed_stone >= 25:
            return True, "Draw!", None
        else:
            return False, "", None

    def reset(self):
        self.runhistory = []
        self.turns = 0
        self.board.init_stones(self.agent_one, self.agent_two)
        self.board.refresh_board()
        self.cum_rewards_agent_one = None
        self.cum_rewards_agent_two = None
        self.winner = None

class GameWrapper:

    def __init__(self):
        pass
