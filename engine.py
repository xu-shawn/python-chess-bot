import chess
import datetime

from sympy.codegen.ast import continue_

import consts

from sys import argv
from eval import NNUE
from tt import TranspositionTable
from moveorder import Movepick
from timemanager import Timemanager

# Use for node count
nodes: int = 0

# Used for time management
time: int = -40000
increment: int = 0

class Searcher:

    def __init__(self):
        """Initializes the searcher with board and constants"""
        self.board = chess.Board()
        self.MAX = 100000
        self.MATE = 40000
        self.eval = NNUE("simple.nnue")
        self.tt = TranspositionTable()
        self.start = datetime.datetime.now()
        self.tm = Timemanager()

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

        global nodes, time, increment
        """Main search, utilitzes fail-soft alpah-beta pruning inside a negamax framework"""
        nodes += 1

        """duration = datetime.datetime.now() - start_time
        elapsed = int(duration.total_seconds() * 1000)"""

        if time != -40000 and int((datetime.datetime.now() - self.start).total_seconds() * 1000 >= self.tm.getTimeForMove(time, increment)):
            return beta, None

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

        start_time = datetime.datetime.now()

        for position in consts.test_positions:
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

        if len(argv) > 1:
            if argv[1] == "bench":
                self.benchmark.run_benchmark()
                return False

        cmd_input = input()
        if not cmd_input:
            return True
        args = cmd_input.split()

        global time, increment

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

                if len(args) > 8:
                    if args[1] == "wtime":
                        time = int(args[2])

                    if args[3] == "btime":
                        time = int(args[4])

                    if args[5] == "winc":
                        increment = int(args[6])

                    if args[7] == "btime":
                        increment = int(args[8])

                    print("The wtime is" + str(time))
                    self.searcher.start_search(depth=256, board=self.board)
                    return True
                else:
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
