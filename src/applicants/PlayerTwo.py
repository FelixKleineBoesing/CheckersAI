from ..Applicant import Applicant


class PlayerTwo(Applicant):

    def __init__(self, name: str, side: str = "up"):
        super().__init__(name, side)

    def move_stone(self):
        pass

    def play_turn(self, action_space: dict):
        action_space.keys()
        pass

    def decision(self):
        pass