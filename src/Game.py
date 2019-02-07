from src.Board import Board
from src.Applicant import Applicant


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
        self.winner = None

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
        finished = False
        turns_without_removed_stone = 0

        while not finished:
            number_stones_before = self.board.number_of_stones()
            self.turns += 1
            print("Iteration:{}".format(self.turns))
            action_space_p_one = self.get_action_space(self.player_one.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_one) == 0:
                reason = "PLayer one is blocked and can´t move!"
                winner = self.player_one.name
                break
            move, stone_id = self.player_one.play_turn(action_space_p_one)
            self.board.move_stone(move, stone_id)
            self.board.refresh_board()
            print("--------Player One moved-----------")
            self.board.print_board()
            finished, reason, winner = self.game_finished(turns_without_removed_stone)
            if finished:
                break

            action_space_p_two =self.get_action_space(self.player_two.name)
            # if player is blocked and can´t do any moves, than he has lost the game
            if len(action_space_p_two ) == 0:
                reason = "PLayer two is blocked and can´t move!"
                winner = self.player_two.name
                break
            move, stone_id = self.player_two.play_turn(action_space_p_two)
            self.board.move_stone(move, stone_id)
            self.board.refresh_board()
            print("--------Player two moved-----------")
            self.board.print_board()
            number_stones_after = self.board.number_of_stones()
            turns_without_removed_stone = turns_without_removed_stone + 1 if number_stones_after == \
                                                                             number_stones_before else 0
            finished, reason, winner = self.game_finished(turns_without_removed_stone)

        print("Game finished with the following message: {}".format(reason))
        self.winner = winner
        #end game here

    def game_finished(self, turns_without_removed_stone: int):
        player_vanished, players = self.board.check_if_player_without_stones()
        if player_vanished:
            return True, "Player {} won!".format(players[0]), players[0]
        elif turns_without_removed_stone >= 20:
            return True, "Draw!", None
        else:
            return False, "", None
