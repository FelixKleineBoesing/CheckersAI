import logging

from checkers.src.game.Game import Game
from checkers.src.game.Board import Board
from checkers.src.agents.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from checkers.src.agents.RandomAgent import RandomAgent
from checkers.src.agents.QLearningAgent import QLearningAgent
from checkers.src.agents.QLearningLSTMAgent import QLearningLSTMAgent
from checkers.src.agents.SARSAAgent import SARSAAgent
from checkers.src.agents.SARSALSTMAgent import SARSALSTMAgent


def run_random_vs_random_max():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)
    agent_one = RandomAgentWithMaxValue((board_length, board_length), action_space, "One", "up")
    agent_two = RandomAgent((board_length, board_length), action_space, "Two", "down")
    iterations = 1000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "One":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1

            logging.info("PLayer One: {}".format(str(victories_player_one)))
            logging.info("PLayer One: {}".format(str(victories_player_two)))


def run_random_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningAgent((board_length, board_length), action_space, "qlearning", "up", 1.0, 2500, 50000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 50000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.9999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "qlearning":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1

            logging.info("PLayer One: {}".format(str(victories_player_one)))
            logging.info("PLayer One: {}".format(str(victories_player_two)))


def run_sarsa_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningAgent((board_length, board_length), action_space, "qlearning", "up", 0.0, 2000000000, 5000000000)
    agent_two = SARSAAgent((board_length, board_length), action_space, "sarsa", "down", 0.0, 20000000, 500000000)
    iterations = 100
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.9999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "qlearning":
                    victories_player_one += 1
                if winner == "sarsa":
                    victories_player_two += 1

            logging.info("PLayer One: {}".format(str(victories_player_one)))
            logging.info("PLayer One: {}".format(str(victories_player_two)))


def run_sarsa_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = SARSAAgent((board_length, board_length), action_space, "sarsa", "up", 0.0, 2000, 50000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 100
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.9999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "sarsa":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1

            logging.info("PLayer One: {}".format(str(victories_player_one)))
            logging.info("PLayer Two: {}".format(str(victories_player_two)))


def run_sarsa_lstm_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = SARSALSTMAgent((board_length, board_length), action_space, "sarsa_lstm", "up", 0.0, 2000, 50000,
                               caching=False)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 100
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.99999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "sarsa_lstm":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1
            logging.info("Current epsilon: {}".format(agent_one.epsilon))
            logging.info("PLayer One: {}".format(str(victories_player_one)))
            logging.info("PLayer Two: {}".format(str(victories_player_two)))


def run_q_lstm_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningLSTMAgent((board_length, board_length), action_space, "q_lstm", "up", 1.0, 2000, 50000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 200000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game("Test", agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.99999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "sarsa_lstm":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1
            logging.info("Current epsilon: {}".format(agent_one.epsilon))
            logging.info("PLayer One: {}".format(str(victories_player_one)))
            logging.info("PLayer Two: {}".format(str(victories_player_two)))


if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    #run_random_vs_random_max()
    #run_random_vs_qlearning()
    #run_sarsa_vs_qlearning()
    #run_sarsa_vs_random()
    run_sarsa_lstm_vs_random()
    #run_q_lstm_vs_random()