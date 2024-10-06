import chess
from collections.abc import Generator

class Movepick:
    def __init__(self, tt_move: chess.Move, moves_list: chess.PseudoLegalMoveGenerator):
        """Initializes the MovePick object with ttMove and the board's legal move generator"""
        self.ttMove = tt_move
        self.moves_list = moves_list
        self.stage = 0

    def moves(self) -> Generator:
        """Generator function that yields the next move to explore"""
        if self.ttMove is not None:
            yield self.ttMove

        for move in self.moves_list:
            if move == self.ttMove:
                continue
            yield move