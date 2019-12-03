from checkers.src.agents.qlearning_agent import QLearningAgent
from checkers.src.agents.random_agent import RandomAgent
from checkers.src.agents.random_agent_with_max_walue import RandomAgentWithMaxValue

random_agent = RandomAgent
random_agent_special_args = {}
random_agent_description = ""

random_agent_with_max_value = RandomAgentWithMaxValue
random_agent_with_max_value_special_args = {}
random_agent_with_max_value_description = ""

qlearning_agent = QLearningAgent
qlearning_agent_special_args = {"epsilon": 0.0, "intervall_turns_train": 10000, "intervall_turns_load": 10000}
qlearning_agent_description = {}
