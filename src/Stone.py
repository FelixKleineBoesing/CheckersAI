from src.Applicant import Applicant


class Stone:

    def __init__(self, player: Applicant, row: int, col: int, status: str):
        self.player = player
        self.row = row
        self.col = col
        self.status = status

    #def