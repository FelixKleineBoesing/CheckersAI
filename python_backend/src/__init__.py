from src.Game import Game
from src.Board import Board
from src.Applicant import Applicant
from src.applicants.RandomPlayerWithMaxValue import RandomPlayerWithMaxValue
from src.applicants.RandomPlayer import RandomPlayer


if __name__=="__main__":
    winners = []
    for i in range(1000):
        board = Board(board_length=8)
        player_one = RandomPlayerWithMaxValue("One", "up")
        player_two = RandomPlayer("Two", "down")
        game = Game("Test", player_one=player_one, player_two=player_two, board=board)
        game.play()
        winners += [game.winner]
        print(winners)
        victories_player_two = 0
        victories_player_one = 0
        for winner in winners:
            if winner == "One":
                victories_player_one += 1
            if winner == "Two":
                victories_player_two += 1

    print(victories_player_one)
    print(victories_player_two)