

class Applicant:

    def __init__(self, name: str, side: str = "up"):
        assert side in ["up", "down"], "Side has to be up or down"
        self.name = name
        self.side = side

    def get_board(self):
        pass

    def move_stone(self):
        pass

    def play_turn(self):
        pass