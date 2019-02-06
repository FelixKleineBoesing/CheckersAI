from ..Applicant import Applicant
import random

class RandomPlayer(Applicant):

    def __init__(self, name: str, side: str = "up"):
        super().__init__(name, side)

    def decision(self, action_space: dict):

        keys = [key for key in action_space.keys()]
        ix = random.sample(range(len(keys)), 1)[0]
        stone_id = keys[ix]

        move_id = random.sample(range(len(action_space[stone_id])), 1)[0]

        return {"stone_id": stone_id, "move_id": move_id}
