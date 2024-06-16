#!/Users/shawn/Documents/development/lichess-bot-python-final-project/venv/bin/python
from collections.abc import Generator
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import chess


# NNUE Evaluation
class NNUE:
    def __init__(self, filename: str, feature_size: int = 768, hidden_size: int = 1024):
        data = Path(filename).read_bytes()
        self.raw = np.frombuffer(data, dtype="<i2")
        self.ft = self.raw[: feature_size * hidden_size].reshape(
            feature_size, hidden_size
        )
        self.ftBiases = self.raw[
            feature_size * hidden_size : feature_size * hidden_size + hidden_size
        ].reshape(hidden_size)
        self.outputWeights = self.raw[
            feature_size * hidden_size
            + hidden_size : feature_size * hidden_size
            + hidden_size * 3
        ].reshape(hidden_size * 2)
        self.outputBias = self.raw[feature_size * hidden_size + hidden_size * 3]

        self.hidden_size = hidden_size
        self.feature_size = feature_size

    def feature_index(self, piece: chess.Piece, square: chess.Square, flipped: bool):
        if flipped:
            side_is_black = piece.color
            square = square ^ 0x38
        else:
            side_is_black = not piece.color
        return square + int(piece.piece_type - 1) * 64 + (384 if side_is_black else 0)

    def visualize1(self, board: chess.Board, neuron_index: int = 0):

        intensity = np.zeros((8, 8))

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            intensity[chess.square_rank(square)][chess.square_file(square)] = self.ft[
                self.feature_index(piece, square, False)
            ][neuron_index]

        self.display(intensity)

    def visualize2(self, piecetype, color, neuron_index: int = 0):

        intensity = np.zeros((8, 8))

        for square in chess.SQUARES:
            piece = chess.Piece(piecetype, color)
            if piece is None:
                continue
            intensity[chess.square_file(square)][chess.square_rank(square)] = self.ft[
                self.feature_index(piece, square, False)
            ][neuron_index]

        self.display(intensity)

    def display(self, intensity):

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(intensity, cmap="magma", interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])

    def full_evaluate(self, board: chess.Board):
        accumulatorWhite = self.ftBiases.copy()
        accumulatorBlack = self.ftBiases.copy()

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            accumulatorWhite += self.ft[self.feature_index(piece, square, False)]
            accumulatorBlack += self.ft[self.feature_index(piece, square, True)]

        if board.turn == chess.WHITE:
            total = np.sum(
                accumulatorWhite.clip(0, 181).astype(np.int32) ** 2
                * self.outputWeights[: self.hidden_size]
            ) + np.sum(
                accumulatorBlack.clip(0, 181).astype(np.int32) ** 2
                * self.outputWeights[self.hidden_size :]
            )
        else:
            total = np.sum(
                accumulatorBlack.clip(0, 181).astype(np.int32) ** 2
                * self.outputWeights[: self.hidden_size]
            ) + np.sum(
                accumulatorWhite.clip(0, 181).astype(np.int32) ** 2
                * self.outputWeights[self.hidden_size :]
            )

        value = (total // 181 + self.outputBias) * 400 // (181 * 64)

        return value


class TranspositionTable:
    def __init__(self):
        self.table = {}

    def probe(self, board: chess.Board):
        return self.table.get(board._transposition_key(), None)

    def store(self, board: chess.Board, move: chess.Move | None):
        self.table[board._transposition_key()] = move


class Movepick:
    def __init__(self, ttMove: chess.Move, moves_list: chess.PseudoLegalMoveGenerator):
        self.ttMove = ttMove
        self.moves_list = moves_list
        self.stage = 0

    def moves(self) -> Generator:

        if self.ttMove != None:
            yield self.ttMove

        for move in self.moves_list:
            if move == self.ttMove:
                continue
            yield move


class Searcher:
    def __init__(self):
        self.board = chess.Board()
        self.MAX = 100000
        self.MATE = 40000
        self.eval = NNUE("simple.nnue")
        self.tt = TranspositionTable()

    def start_search(self, depth=3, board: chess.Board = chess.Board()) -> None:
        self.board = board
        for i in range(1, depth + 1):
            best_value, best_move = self.search(i, 0, self.board, -self.MAX, self.MAX)
            print(
                f"info depth {i} score cp {best_value} pv {best_move.uci() if best_move is not None else '(none)'}"
            )

        print("bestmove", best_move.uci() if best_move is not None else "(none)")

    def search(
        self, depth: int, ply: int, board: chess.Board, alpha, beta
    ) -> tuple[int, chess.Move | None]:

        if depth <= 0:
            return self.eval.full_evaluate(self.board), None
            # return self.qsearch(ply, self.board, alpha, beta), None

        best_value: int = -self.MAX
        best_move: chess.Move | None = None
        old_alpha = alpha
        move_count = 0

        tt_move = self.tt.probe(board)
        mp = Movepick(tt_move, board.pseudo_legal_moves)

        for move in mp.moves():
            if not board.is_legal(move):
                continue

            move_count += 1
            self.board.push(move)
            score: int = -self.search(depth - 1, ply + 1, board, -beta, -alpha)[0]
            self.board.pop()

            if score > best_value:
                best_value = score
                if best_value > alpha:
                    alpha = best_value
                    best_move = move
                    if best_value >= beta:
                        return best_value, None

        if move_count == 0:
            return ((-40000 + ply) if board.is_check() else 0), None

        self.tt.store(board, tt_move if old_alpha == alpha else best_move)

        return best_value, best_move

    def qsearch(self, ply, board: chess.Board, alpha, beta) -> int:

        best_value: int = -self.MAX
        move_count = 0

        if not board.is_check():
            stand_pat: int = self.eval.full_evaluate(self.board)
            alpha = max(alpha, stand_pat)
            if stand_pat >= beta:
                return stand_pat

        for move in board.legal_moves:
            if not board.is_check() and not board.is_capture(move):
                continue
            move_count += 1

            self.board.push(move)
            score: int = -self.qsearch(ply + 1, board, -beta, -alpha)
            self.board.pop()

            if score > best_value:
                best_value = score
                if best_value > alpha:
                    alpha = best_value
                    if best_value >= beta:
                        return best_value

        if move_count == 0:
            return (
                (-40000 + ply)
                if board.is_check()
                else self.eval.full_evaluate(self.board)
            )

        return best_value


class UCI:
    def __init__(self):
        self.board = chess.Board()
        self.searcher = Searcher()

    def receive(self) -> bool:
        cmd_input = input()
        if not cmd_input:
            return True
        args = cmd_input.split()

        match args[0]:
            case "isready":
                print("readyok")
            case "uci":
                print("id name Python-Chess-Engine")
                print("id author Shawn Xu")
                print("uciok")
            case "ucinewgame":
                self.board = chess.Board()
            case "quit":
                return False
            case "position":
                if args[1] == "startpos":
                    self.board.set_fen(
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    )
                else:
                    self.board.set_fen(" ".join(args[2:8]))
                if "moves" in cmd_input:
                    for move in cmd_input.split("moves")[1].split():
                        self.board.push(chess.Move.from_uci(move))
            case "go":
                self.searcher.start_search(board=self.board)

        return True

    def loop(self) -> None:
        while self.receive():
            pass


if __name__ == "__main__":
    uci = UCI()
    uci.loop()
