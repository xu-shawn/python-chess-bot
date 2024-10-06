import numpy as np
import matplotlib.pyplot as plt
import chess

from pathlib import Path

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
        """Plots the intensity of the weights from every square of the board to a specific neuron"""
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

