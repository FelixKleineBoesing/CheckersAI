from python_backend.src.Helpers import default_rewards
from python_backend.src.Board import Board
from python_backend.src.Game import Game
from python_backend.src.Stone import Stone
from python_backend.src.agents.RandomAgentWithMaxValue import RandomAgentWithMaxValue
from python_backend.src.agents.RandomAgent import RandomAgent
from python_backend.src.agents.QLearningAgent import QLearningAgent

from flask import Flask
from flask import request
from waitress import serve
import json
import sys

app = Flask(__name__)
agents = [{"name": "RandomAgentWithMaxValue", "description": "Agent that randomly chooses actions that have the "
                                                             "maximum value based on the jumped stones"},
          {"name": "RandomAgent", "description": "Agent that randomly chooses actions"},
          {"name": "QLearningAgent", "description": "Q Learning Agent that greedily chooses the actions with the "
                                                    "highest Q Values"}]


@app.route("/get_agents", methods=["GET"])
def get_agents():
    return json.dumps(agents)


@app.route("/run_game", methods=["GET"])
def run_game():
    try:
        agent_one = getattr(sys.modules["__main__"], request.args.get("agent_one"))
    except AttributeError:
        return json.dumps({"status": "error", "error_type": "NoPlayerFound", "text": "The specified agent one could not "
                                                                                     "be found"})
    try:
        agent_two = getattr(sys.modules["__main__"], request.args.get("agent_two"))
    except AttributeError:
        return json.dumps({"status": "error", "error_type": "NoPlayerFound", "text": "The specified agent two could not "
                                                                                     "be found"})

    board_len = 8
    agent_one = agent_one((board_len, board_len), (board_len, board_len), "One", "up")
    agent_two = agent_two((board_len, board_len), (board_len, board_len), "Two", "down")
    board = Board(board_length=board_len)

    game = Game("Prod", agent_one=agent_one, agent_two=agent_two, board=board, save_runhistory=True)
    game.play(verbose=True)
    return json.dumps({"status": "ok", "runhistory": game.runhistory, "text": "Player {0} wins!".format(game.winner)})


serve(app, port=5001, threads=4)

