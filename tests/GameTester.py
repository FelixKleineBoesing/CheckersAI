import logging

from game.Game import Game
from game import Board
from agents.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from agents import RandomAgent
from agents import QLearningAgent
from agents.QLearningLSTMAgent import QLearningLSTMAgent
from agents import SARSAAgent
from agents.SARSALSTMAgent import SARSALSTMAgent
from agents import A2C


def run_random_vs_random_max():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)
    agent_one = RandomAgentWithMaxValue((board_length, board_length), action_space, "One", "up")
    agent_two = RandomAgent((board_length, board_length), action_space, "Two", "down")
    iterations = 1000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
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

            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))#


def run_random_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningAgent((board_length, board_length), action_space, "qlearning", "up", 1.0, 2500, 100000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 50000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
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

            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))


def run_sarsa_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningAgent((board_length, board_length), action_space, "qlearning", "up", 0.0, 250000000, 10000000)
    agent_two = SARSAAgent((board_length, board_length), action_space, "sarsa", "down", 0.0, 25000000, 10000000)
    iterations = 10000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.9999
        agent_two.epsilon *= 0.9999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "qlearning":
                    victories_player_one += 1
                if winner == "sarsa":
                    victories_player_two += 1

            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))


def run_sarsa_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = SARSAAgent((board_length, board_length), action_space, "sarsa", "up", 0.0, 2000000, 100000000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 100
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
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

            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))


def run_sarsa_lstm_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = SARSALSTMAgent((board_length, board_length), action_space, "sarsa_lstm", "up", 1.0, 2000, 100000,
                               caching=False)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 200000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
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
            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))


def run_q_lstm_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningLSTMAgent((board_length, board_length), action_space, "q_lstm", "up", 1.0, 2000, 100000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 200000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
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
            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))


def run_a2c_vs_random():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = A2C((board_length, board_length), action_space, "a3c", "up", 1.0, 2000, 100000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
    iterations = 200000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.99999
        if (i % 5000 == 0 and i > 0) or (iterations - 1 == i):
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "a3c":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1
            logging.info("Current epsilon: {}".format(agent_one.epsilon))
            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))

def run_sarsa_vs_sarsa():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = SARSAAgent((board_length, board_length), action_space, "sarsa", "up", 0.0, 25000000, 100000000)
    agent_two = SARSAAgent((board_length, board_length), action_space, "sarsa_two", "down", 0.0, 25000000, 10000000,
                           save_path="../data/modeldata/sarsa_two/model.ckpt")
    iterations = 100
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.9999
        agent_two.epsilon *= 0.9999
        if (i % 5000 == 0 and i > 0) or iterations - 1 == i:
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "sarsa":
                    victories_player_one += 1
                if winner == "sarsa_two":
                    victories_player_two += 1

            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))


def run_a2c_vs_sarsa():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = A2C((board_length, board_length), action_space, "a3c", "up", 1.0, 2000, 100000)
    agent_two = SARSAAgent((board_length, board_length), action_space, "sarsa_two", "down", 1.0, 2000, 100000,
                           save_path="../data/modeldata/sarsa_two/model.ckpt")
    iterations = 200000
    for i in range(iterations):
        board = Board(board_length=8)
        game = Game(agent_one=agent_one, agent_two=agent_two, board=board)
        game.play(verbose=False)
        winners += [game.winner]
        agent_one.epsilon *= 0.99999
        if (i % 5000 == 0 and i > 0) or (iterations - 1 == i):
            victories_player_two = 0
            victories_player_one = 0
            for winner in winners:
                if winner == "a3c":
                    victories_player_one += 1
                if winner == "Two":
                    victories_player_two += 1
            logging.info("Current epsilon: {}".format(agent_one.epsilon))
            logging.info("Player One: {}".format(str(victories_player_one)))
            logging.info("Player Two: {}".format(str(victories_player_two)))
            logging.info("Mean Rewards Agent One: {}".format(agent_one.moving_average_rewards[-1]))
            logging.info("Mean Rewards Agent Two: {}".format(agent_two.moving_average_rewards[-1]))

if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    #run_random_vs_random_max()
    #run_random_vs_qlearning()
    #run_sarsa_vs_qlearning()
    #run_sarsa_vs_random()
    #run_sarsa_lstm_vs_random()
    #run_q_lstm_vs_random()
    run_a2c_vs_sarsa()
    #run_sarsa_vs_sarsa()
