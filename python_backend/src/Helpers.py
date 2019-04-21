from typing import Union


class Rewards:

    def __init__(self, win: Union[float, int],
                 loss: Union[float, int],
                 queen_stone_loss: Union[float, int],
                 normal_stone_loss: Union[float, int],
                 normal_stone_taken: Union[float, int],
                 queen_stone_taken: Union[float, int],
                 turn: Union[float, int]):
        """
        This class represents the reward function which returns a reward for a discrete set of actions
        :param win: reward for winning a game
        :param queen_stone_loss: reward for loosing a queen stone
        :param normal_stone_loss: reward for loosing a normal stone
        :param normal_stone_taken: reward for taking a normal stone
        :param queen_stone_taken: reward for taking a queen stone
        :param turn: reward for playing a turn
        """
        assert all(isinstance(arg, float) or isinstance(arg, int) for arg in [win, loss, queen_stone_loss, queen_stone_taken, normal_stone_loss,
                                                      normal_stone_taken])

        self.win = win
        self.loss = loss
        self.normal_stone_loss = normal_stone_loss
        self.queen_stone_loss = queen_stone_loss
        self.normal_stone_taken = normal_stone_taken
        self.queen_stone_taken = queen_stone_taken
        self.turn = turn


default_rewards = Rewards(100, -100, 0, 0, 10, 10, 0)