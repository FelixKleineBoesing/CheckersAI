from ..Agent import Agent
import random
import numpy as np

class RandomAgentWithMaxValue(Agent):

    def __init__(self, name: str, side: str = "up"):
        """
        chooses actions randomly from actions with max jumoed stones
        :param name:
        :param side:
        """
        super().__init__(name, side)

    def decision(self, state_space: np.ndarray, action_space: dict):

        max_value = 0
        for key in action_space.keys():
            for move in action_space[key]:
                if abs(move["jumped_values"]) > max_value:
                    max_value = move["jumped_values"]

        limited_actions = {}
        for key in action_space.keys():
            for move in action_space[key]:
                if move["jumped_values"] == max_value:
                    if key in limited_actions.keys():
                        limited_actions[key] += [move]
                    else:
                        limited_actions[key] = [move]

        keys = [key for key in limited_actions.keys()]
        ix = random.sample(range(len(keys)), 1)[0]
        stone_id = keys[ix]

        move_id = random.sample(range(len(limited_actions[stone_id])), 1)[0]

        return {"stone_id": stone_id, "move_id": move_id}
