import abc


class Applicant(abc.ABC):

    def __init__(self, name: str, side: str = "up"):
        assert side in ["up", "down"], "Side has to be up or down"
        self.name = name
        self.side = side

    @abc.abstractmethod
    def move_stone(self):
        pass

    @abc.abstractmethod
    def play_turn(self, action_space: dict):
        pass

    @abc.abstractmethod
    def decision(self):
        pass