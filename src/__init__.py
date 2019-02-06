from src.Game import Game
from src.Board import Board
from src.Applicant import Applicant
from src.applicants.PlayerOne import PlayerOne
from src.applicants.PlayerTwo import PlayerTwo
from src.applicants.RandomPlayer import RandomPlayer


if __name__=="__main__":
    winners = []
    for i in range(10):
        board = Board(board_length=8)
        player_one = RandomPlayer("One", "up")
        player_two = RandomPlayer("Two", "down")
        game = Game("Test", player_one=player_one, player_two=player_two, board=board)
        game.play()
        winners += [game.winner]
