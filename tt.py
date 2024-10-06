import chess

class TranspositionTable:
    """Transposition Table: https://www.chessprogramming.org/Transposition_Table

    Records the best move for every position"""
    def __init__(self):
        self.table = {}

    def probe(self, board: chess.Board):
        """Look up information from the transpositon table"""
        return self.table.get(board._transposition_key(), None)

    def store(self, board: chess.Board, move: chess.Move | None):
        """Stores information to the transposition table"""
        self.table[board._transposition_key()] = move