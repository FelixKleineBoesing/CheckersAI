from src.Board import Board
from src.Applicant import Applicant
from src.applicants.PlayerOne import PlayerOne
from src.applicants.PlayerTwo import PlayerTwo


class Game:

    def __init__(self, name: str, player_one: Applicant, player_two: Applicant,
                 board: Board):
        assert isinstance(player_one, Applicant)
        assert isinstance(player_two, Applicant)
        assert isinstance(board, Board)
        self.name = name

        # init players
        self.player_one = player_one
        self.player_two = player_two

        # init board
        self.board = board
        self.board.init_stones(self.player_one, self.player_two)
        self.board.refresh_board()
        self.board.print_board()

        # stats
        self.turns = 0

    def get_action_space(self, player_name: str):
        return self.board.get_all_moves(player_name)

    def play(self):
        unfinished = True
        turns_without_removed_stone = 0

        while unfinished:
            number_stones_before = self.board.number_of_stones()

            self.turns += 1
            action_space_p_one = self.get_action_space(self.player_one.name)
            self.player_one.play_turn(action_space_p_one)
            self.board.refresh_board()
            unfinished = self.game_running(turns_without_removed_stone)
            if not unfinished:
                break

            action_space_p_two =self.get_action_space(self.player_two.name)
            self.player_two.play_turn(action_space_p_two)
            self.board.refresh_board()
            number_stones_after = self.board.number_of_stones()
            self.game_running(turns_without_removed_stone)
            turns_without_removed_stone = turns_without_removed_stone + 1 if number_stones_after == \
                                                                             number_stones_before else 0

        #end game here

    def game_running(self, turns_without_removed_stone: int):
        if self.board.check_if_player_without_stones() or turns_without_removed_stone >= 20:
            return False
        else:
            return True
