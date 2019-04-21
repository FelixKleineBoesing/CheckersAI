from python_backend.src.Game import Game
from python_backend.src.Board import Board
from python_backend.src.Agent import Agent
from python_backend.src.applicants.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from python_backend.src.applicants.RandomAgent import RandomAgent
from python_backend.src.applicants.QLearningAgent import QLearningAgent
from python_backend.src.applicants.QLearningAgent import load_weigths_into_target_network
import tensorflow as tf


if __name__=="__main__":
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    winners = []
    board_length = 8
    player_one = QLearningAgent((board_length, board_length), (board_length, board_length), "One", "up", 1.0)
    sess.run(tf.global_variables_initializer())

    target_network = QLearningAgent((board_length, board_length), (board_length, board_length), "target_network", "up", 0.0)
    load_weigths_into_target_network(player_one, target_network)
    sess.run([tf.assert_equal(w, w_target) for w, w_target in zip(player_one.weights, target_network.weights)]);
    for i in range(1000):
        board = Board(board_length=8)
        player_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")
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
        player_one.epsilon *= 0.98


    print(victories_player_one)
    print(victories_player_two)