from src.Applicant import Applicant
from typing import Tuple
coord = Tuple[int, int]


class Stone:

    def __init__(self,id: int,  player: Applicant, coord: coord, status: str, value: int):
        self.id = id
        self.player = player
        self.coord = coord
        self.status = status
        self.value = value

