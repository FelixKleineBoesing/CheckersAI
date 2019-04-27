from python_backend.src.Game import Game
from python_backend.src.Board import Board
from python_backend.src.Agent import Agent
from python_backend.src.agents.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from python_backend.src.agents.RandomAgent import RandomAgent
from python_backend.src.agents.QLearningAgent import QLearningAgent
import tensorflow as tf


def run_random_vs_random_max():
    winners = []
    board_length = 8
    player_one = RandomAgentWithMaxValue((board_length, board_length), (board_length, board_length), "One", "up")
    player_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    for i in range(1000):
        board = Board(board_length=8)

        game = Game("Test", player_one=player_one, player_two=player_two, board=board)
        game.play(verbose=True)
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
    player_one = QLearningAgent((board_length, board_length), (board_length, board_length), "One", "up", 1.0)
    player_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    for i in range(1000):
        board = Board(board_length=8)
        game = Game("Test", player_one=player_one, player_two=player_two, board=board)
        game.play(verbose=True)
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


if __name__=="__main__":
    #run_random_vs_random_max()
    run_random_vs_qlearning()