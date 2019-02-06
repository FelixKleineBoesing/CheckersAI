from src.Game import Game
from src.Board import Board
from src.Applicant import Applicant
from src.applicants.PlayerOne import PlayerOne
from src.applicants.PlayerTwo import PlayerTwo


if __name__=="__main__":
    board = Board(board_length=8)
    player_one = PlayerOne("One", "up")
    player_two = PlayerTwo("Two", "down")
    game = Game("Test", player_one=player_one, player_two=player_two, board=board)
    game.play()