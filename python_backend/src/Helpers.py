

class Rewards:

    def __init__(self, win: float, queen_stone_loss: float, normal_stone_loss: float,
                 normal_stone_taken: float, queen_stone_taken: float, turn: float):
        """
        This class represents the reward function which returns a reward for a discrete set of actions
        :param win: reward for winning a game
        :param queen_stone_loss: reward for loosing a queen stone
        :param normal_stone_loss: reward for loosing a normal stone
        :param normal_stone_taken: reward for taking a normal stone
        :param queen_stone_taken: reward for taking a queen stone
        :param turn: reward for playing a turn
        """
        assert all(isinstance(arg, float) for arg in [win, queen_stone_loss, queen_stone_taken, normal_stone_loss,
                                                      normal_stone_taken])

        self.win = win
        self.normal_stone_loss = normal_stone_loss
        self.queen_stone_loss = queen_stone_loss
        self.normal_stone_taken = normal_stone_taken
        self.queen_stone_taken = queen_stone_taken
        self.turn = turn

DefaultRewards = Rewards(100, 0, 0, 10, 10, 0)