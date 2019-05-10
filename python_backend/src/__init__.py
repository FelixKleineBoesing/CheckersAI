import pandas as pd

from python_backend.src.Game import Game
from python_backend.src.Board import Board
from python_backend.src.agents.Agent import Agent
from python_backend.src.agents.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from python_backend.src.agents.RandomAgent import RandomAgent
from python_backend.src.agents.QLearningAgent import QLearningAgent


def run_random_vs_random_max():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)
    agent_one = RandomAgentWithMaxValue((board_length, board_length), action_space, "One", "up")
    agent_two = RandomAgent((board_length, board_length), action_space, "Two", "down")
    for i in range(1000):
        board = Board(board_length=8)

        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        victories_player_two = 0
        victories_player_one = 0
        for winner in winners:
            if winner == "One":
                victories_player_one += 1
            if winner == "Two":
                victories_player_two += 1

    print(victories_player_one)
    print(victories_player_two)


def run_random_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)
    rewards = []
    agent_one = QLearningAgent((board_length, board_length), action_space, "One", "up", 1.0, 100, 1000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    for i in range(30000):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        if i % 50 == 0:
            print("{} Iterations".format(i))
        winners += [game.winner]
        victories_player_two = 0
        victories_player_one = 0
        for winner in winners:
            if winner == "One":
                victories_player_one += 1
            if winner == "Two":
                victories_player_two += 1
        agent_one.epsilon *= 0.99
        rewards += [game.cum_rewards_agent_one]
        if i % 1000 == 0 and i > 0:
            df = pd.DataFrame({"rewards": rewards})
            df.to_csv("rewards.txt")
    print(victories_player_one)
    print(victories_player_two)


if __name__=="__main__":
    #run_random_vs_random_max()
    run_random_vs_qlearning()