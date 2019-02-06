from ..Applicant import Applicant


class PlayerOne(Applicant):

    def __init__(self, name: str, side: str = "up"):
        super().__init__(name, side)

    def decision(self, action_space):
        pass