from src.Applicant import Applicant


class Stone:

    def __init__(self,id: int,  player: Applicant, coord: tuple[int, int], status: str, value: int):
        self.id = id
        self.player = player
        self.coord = coord
        self.status = status
        self.value = value

    #def