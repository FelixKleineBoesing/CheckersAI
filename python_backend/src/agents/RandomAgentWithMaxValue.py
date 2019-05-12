from python_backend.src.agents.Agent import Agent
import random
import numpy as np

class RandomAgentWithMaxValue(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up"):
        """
        chooses actions randomly from actions with max jumoed stones
        :param name:
        :param side:
        """
        super().__init__(state_shape, action_shape, name, side)

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

        move = action_space[stone_id][move_id]
        decision = np.concatenate([move["old_coord"], move["new_coord"]])

        return decision
