

class Applicant:

    def __init__(self, name: str, side: str = "up"):
        assert str in ["up", "down"], "Side has to be up or down"
        self.name = name
        self.side = side

    def get_board(self):
        pass

    def move_stone(self):
        pass
