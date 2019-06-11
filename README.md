# CheckersAI - in development
 


Finished:
- Checkers Framework
- Random Agents
- Q Learning Agent (Dense & LSTM Networks)
- SARSA Agent (Dense & LSTM Networks)
- A2C Agent (Dense & LSTM Networks)

In Development:
- React Frontend for Vizualisation
- Python Rest Interface
- New Exploration Strategies

# CheckersAI

This is a small learning project for Reinforcement Learning.
I wanted to develop a everything from scratch (except the deep learnning framework) 
since you don´t get a perfect existing enviroment for real-world problems.
Therefore one has to think about it oneself how states and actions are represented and 
what rewards to give for certain (State, Action, Next State) tuples.
Feel free to contribute and help debugging!

## Getting Started

These instructions show you how to use this project to set up a 
checkes environment as a package iin dockerized environment. The docker 
solution will support a react frontend for display of games and gaming 
against an agent while it will also support statistics during learning.

### Prerequisites (if docker solution is preferred)

If you haven´t already installed docker, just download the installer from docs.docker.com
[Docker for Windows](https://docs.docker.com/docker-for-windows/install/)
[Docker for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
[Docker for Mac OS](https://docs.docker.com/docker-for-mac/install/)


### Prerequisites (if docker solution is preferred)

Assuree that you have installed python 3.X. Python 2 is not supported 

### Installing (Docker)

Running the environment is fairly simple. Switch to the cloned repository and build the images

```
git clone https://github.com/FelixKleineBoesing/gymTraining.git
cd gymTraining
docker-compose up --build
```

### Installing (Package)

Running the environment is fairly simple. Switch to the cloned repository and build the images

```
git clone https://github.com/FelixKleineBoesing/gymTraining.git
cd gymTraining
pip3 install .
```

### Content

Game class which implements the environment.
Agent abstract class which defines the acting interface to the game. 

### Usage

Example function (documented in GameTester):
```
def run_random_vs_qlearning():
    winners = []
    board_length = 8
    action_space = (board_length, board_length, board_length, board_length)

    agent_one = QLearningAgent((board_length, board_length), action_space, "One", "up", 1.0, 10, 1000)
    agent_two = RandomAgent((board_length, board_length), (board_length, board_length), "Two", "down")

    for i in range(30000):
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
```

### Contributions

Feel free to fork and enhance the set of notebooks.
