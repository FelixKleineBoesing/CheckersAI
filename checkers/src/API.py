from flask import Flask
from flask import request
from waitress import serve
import json
import sys
import numpy as np
import time
import multiprocessing as mp
import os

print(os.getcwd())
print(os.listdir("."))

from checkers.src.game.Board import Board
from checkers.src.game.Game import Game
from checkers.src.game.GameASync import GameASync
from checkers.src.agents.random_agent import RandomAgent
from checkers.src.agents.random_agent_with_max_walue import RandomAgentWithMaxValue
from checkers.src.agents.qlearning_lstm_agent import QLearningLSTMAgent
from checkers.src.agents.qlearning_agent import QLearningAgent
from checkers.src.agents.SARSALSTMAgent import SARSALSTMAgent
from checkers.src.agents.SARSAAgent import SARSAAgent
from checkers.src.agents.a2c import A2C
from checkers.src.Helpers import update_managed_dict


def fill_mdict(mdict, game_id):
    mdict[game_id] = {"running": False, "action": None, "action_space": None,
                      "board": None, "enemy_moves": None}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


managed_dict = mp.Manager().dict()
managed_dict["game_id"] = 0


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

@app.route("/get-agents", methods=["GET"])
def get_agents():
    return json.dumps(agents)


@app.route("/game-with-two-agents", methods=["GET"])
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
    return json.dumps({"status": "ok", "states": game.runhistory_states, "text": "Player {0} wins!".format(game.winner),
                       "board_values": {"player_one": "positive", "player_two": "negative"},
                       "actions": game.runhistory_actions})


@app.route("/start-game", methods=["GET"])
def start_game():
    try:
        agent_two = getattr(sys.modules["__main__"], request.args.get("agent_one"))
    except AttributeError:
        return json.dumps({"status": "error", "error_type": "NoPlayerFound", "text": "The specified agent one could not"
                                                                                     " be found"})
    board = Board(board_length=8)

    game_id = managed_dict["game_id"]
    game_id += 1
    managed_dict["game_id"] = game_id
    fill_mdict(managed_dict, game_id)

    board_len = 8
    agent_two = agent_two((board_len, board_len), (board_len, board_len), "Two", "down")
    game = GameASync(agent_two=agent_two, board=board, save_runhistory=True, game_id=game_id)
    process = mp.Process(target=game.play, args=(True, managed_dict))
    process.start()
    return json.dumps({"status": "started", "board": game.board.board.tolist(),
                       "game_id": game_id})


@app.route("/get-possible-actions", methods=["GET"])
def get_possible_actions():
    try:
        coords = request.args.get("coord")
    except:
        return json.dumps({"status": "error", "description": "Coord not delivered"})
    try:
        game_id = int(request.args.get("game_id"))
    except:
        return json.dumps({"status": "error", "description": "game_id not delivered"})
    coords = tuple(int(item) for item in coords[1:-1] if item != ",")
    action_space = managed_dict[game_id]["action_space"]
    board = action_space.space_array[coords[0], coords[1], :, :]
    moves = []
    for stone_id, move in action_space.space_dict.items():
        old_coord = move[0]["old_coord"]
        if old_coord[0] == coords[0] and old_coord[1] == coords[1]:
            moves += move

    return json.dumps({"status": "ok", "board": board.tolist(), "moves": moves}, cls=NumpyEncoder)


@app.route("/do-action", methods=["GET"])
def do_action():
    """
    action must be an array of length 4 with the coordinate [x_from, y_from, x_to, y_to]
    :return:
    """
    try:
        action = request.args.get("action")
    except:
        return json.dumps({"status": "error", "description": "action not delivered"})
    try:
        game_id = int(request.args.get("game_id"))
    except:
        return json.dumps({"status": "error", "description": "game_id not delivered"})
    action = tuple(int(item) for item in action[1:-1] if item != ",")
    if len(action) != 4:
        return json.dumps({"status": "error", "message": "action must be an array of length 4!"})
    update_managed_dict(managed_dict, game_id, "action", action)
    time.sleep(3)
    while managed_dict[game_id]["action_space"] is None:
        time.sleep(1)

    enemy_moves = managed_dict[game_id]["enemy_moves"]
    board = managed_dict[game_id]["board"]

    return json.dumps({"status": "ok", "board": board, "enemy_moves": enemy_moves}, cls=NumpyEncoder)


if __name__=="__main__":
    serve(app, port=5001, threads=4)

