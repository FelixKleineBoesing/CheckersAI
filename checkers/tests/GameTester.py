from checkers.src.Game import Game
from checkers.src.Board import Board
from checkers.src.agents.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from checkers.src.agents.RandomAgent import RandomAgent
from checkers.src.agents.QLearningAgent import QLearningAgent
from checkers.src.agents.SARSAAgent import SARSAAgent


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

    agent_one = QLearningAgent((board_length, board_length), action_space, "One", "up", 0.0, 2000, 50000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")

    for i in range(100):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.999

    victories_player_two = 0
    victories_player_one = 0

    for winner in winners:
        if winner == "One":
            victories_player_one += 1
        if winner == "Two":
            victories_player_two += 1

    print(victories_player_one)
    print(victories_player_two)


def run_sarsa_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningAgent((board_length, board_length), action_space, "One", "up", 0.0, 2000000000, 5000000000)
    agent_two = SARSAAgent((board_length, board_length), action_space, "Two", "down", 1.0, 2000, 50000)

    for i in range(100):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.9999

    victories_player_two = 0
    victories_player_one = 0

    for winner in winners:
        if winner == "One":
            victories_player_one += 1
        if winner == "Two":
            victories_player_two += 1

    print(victories_player_one)
    print(victories_player_two)
    print(victories_player_two)


if __name__=="__main__":
    #run_random_vs_random_max()
    run_random_vs_qlearning()
    #run_sarsa_vs_qlearning()