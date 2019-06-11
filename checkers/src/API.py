from checkers.src.game.Board import Board
from checkers.src.game.Game import Game

from flask import Flask
from flask import request
from waitress import serve
import json
import sys

import multiprocessing as mp
managed_dict = mp.Manager().dict



app = Flask(__name__)
agents = [{"name": "RandomAgentWithMaxValue", "description": "Agent that randomly chooses actions that have the "
                                                             "maximum value based on the jumped stones"},
          {"name": "RandomAgent", "description": "Agent that randomly chooses actions"},
          {"name": "QLearningAgent", "description": "Q Learning Agent that greedily chooses the actions with the "
                                                    "highest Q Values"},
          {"name": "QLearningLSTMAgent", "description": "Q Learning Agent that uses LSTM Memory Units instead of "
                                                        "usual Dense Layers!"},
          {"name": "SARSAAgent", "description": "SARSA Agent that considers which action is done in the next state, "
                                                "instead of just taking the max value of all actions!"},
          {"name": "SARSALSTMAgent", "description": "SARSA Agent that considers which action is done in the next state,"
                                                    "instead of just taking the max value of all actions! "
                                                    "It uses LSTMs instead of Dense Layers!"},
          {"name": "A2C", "description": "This agent is a implementation of the actor critic algorithm that deepmind "
                                         "uses!"}]


@app.route("/get_agents", methods=["GET"])
def get_agents():
    return json.dumps(agents)


@app.route("/game_with_two_agents", methods=["GET"])
def run_game():
    try:
        agent_one = getattr(sys.modules["__main__"], request.args.get("agent_one"))
    except AttributeError:
        return json.dumps({"status": "error", "error_type": "NoPlayerFound", "text": "The specified agent one could not"
                                                                                     " be found"})
    try:
        agent_two = getattr(sys.modules["__main__"], request.args.get("agent_two"))
    except AttributeError:
        return json.dumps({"status": "error", "error_type": "NoPlayerFound", "text": "The specified agent two could not"
                                                                                     " be found"})

    board_len = 8
    agent_one = agent_one((board_len, board_len), (board_len, board_len), "One", "up")
    agent_two = agent_two((board_len, board_len), (board_len, board_len), "Two", "down")
    board = Board(board_length=board_len)

    game = Game(agent_one=agent_one, agent_two=agent_two, board=board, save_runhistory=True)
    game.play(verbose=True)
    return json.dumps({"status": "ok", "runhistory": game.runhistory, "text": "Player {0} wins!".format(game.winner),
                       "board_values": {"player_one": "positive", "player_two": "negative"}})


@app.route("/start_game", methods=["GET"])
def start_game():
    pass




serve(app, port=5001, threads=4)

