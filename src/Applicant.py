import abc


class Applicant(abc.ABC):

    def __init__(self, name: str, side: str = "up"):
        assert side in ["up", "down"], "Side has to be up or down"
        self.name = name
        self.side = side

    def play_turn(self, action_space: dict):
        decision = self.decision(action_space)
        move = action_space[decision["stone_id"]][decision["move_id"]]
        stone_id = decision["stone_id"]
        return move, stone_id

    @abc.abstractmethod
    def decision(self, action_space):
        # this function must implement a decision based in the action_space and other delivered arguments
        # return must be a dictionary with the following keys: "stone_id" and "move_index" which indicates
        # the stone and move that should be executed
        pass