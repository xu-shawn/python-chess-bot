from collections.abc import Generator
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import chess
import datetime
from sys import argv

from torch.backends.cudnn import benchmark

# Use for node count
nodes: int = 0

# NNUE Evaluation
class NNUE:
    """Class to store and evaluate neural networks"""
    def __init__(self, filename: str, feature_size: int = 768, hidden_size: int = 512):
        """Reads in NNUE file to memory"""
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
        """Calculates the corresponding index on the input layer of a board feature"""
        if flipped:
            side_is_black = piece.color
            square = square ^ 0x38
        else:
            side_is_black = not piece.color
        return square + int(piece.piece_type - 1) * 64 + (384 if side_is_black else 0)

    def visualize1(self, board: chess.Board, neuron_index: int = 0):
        """Plots the intensity of the weights from every sqaure of the board to a specific neuron"""
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
        """For each piece, plot the weight of that piece on each square to a specific neuron"""
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
        """Plot a 8x8 array in a square grid, colored by intensity"""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(intensity, cmap="magma", interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])

    def full_evaluate(self, board: chess.Board):
        """Compute NNUE output without incremental updates"""
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


class Searcher:
    def __init__(self):
        """Initializes the searcher with board and constants"""
        self.board = chess.Board()
        self.MAX = 100000
        self.MATE = 40000
        self.eval = NNUE("simple.nnue")
        self.tt = TranspositionTable()

    def start_search(self, depth=3, board: chess.Board = chess.Board()) -> None:
        """Iterative deepening loop, also handles uci output"""
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
        """Main search, utilitzes fail-soft alpah-beta pruning inside a negamax framework"""
        global nodes
        nodes += 1
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
        """Quiescient search, the idea is to analyze all captures before returning the evaluation to maintain search stability"""
        best_value: int = -self.MAX
        move_count = 0
        global nodes
        nodes += 1

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


class Benchmark:
    def __init__(self):
        """Initialize board and searcher"""
        self.board = chess.Board()
        self.searcher = Searcher()

        # The bench depth
        self.bench_depth: int = 3

    def run_benchmark(self):
        """Reset the nodes"""
        global nodes
        nodes = 0

        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
			"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
			"4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
			"rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
			"r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
			"r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
			"r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
			"r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
			"4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
			"2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
			"r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
			"3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
			"r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
			"4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
			"3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
			"6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
            		"3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
			"8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
            		"7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
			"8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
           		"8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
			"8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
			"8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
			"5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
			"6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
			"1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
			"6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
			"8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
			"5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
			"4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
			"r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
			"3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
			"4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1"
        ]

        start_time = datetime.datetime.now()

        for position in test_positions:
            # Update the position
            self.board.set_fen(position)

            # Search the position with the bench depth
            self.searcher.start_search(self.bench_depth, self.board)

        duration = datetime.datetime.now() - start_time
        elapsed = int(duration.total_seconds() * 1000)

        # Report final stats
        print("Time  : " + str(elapsed) + " ms")
        print("Nodes : " + str(nodes))
        print("NPS   : " + str(int(nodes / elapsed * 1000)))



class UCI:
    def __init__(self):
        """Initialize board and searcher"""
        self.board = chess.Board()
        self.searcher = Searcher()
        self.benchmark = Benchmark()

    def receive(self) -> bool:
        """Main UCI function, handles one line of command at a time"""

        if  argv[1] == "bench":
            self.benchmark.run_benchmark()
            return False

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
            case "bench":
                self.benchmark.run_benchmark()

        return True

    def loop(self) -> None:
        """Continuously receive UCI communication from input"""
        while self.receive():
            pass


if __name__ == "__main__":
    uci = UCI()
    uci.loop()
