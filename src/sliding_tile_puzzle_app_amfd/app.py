import itertools
import os
import time

# from importlib.resources import files
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, random
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PuzzleImage:
    def __init__(self, puzzle_size, fname):
        self.puzzle_size = puzzle_size
        self.qimage = (
            QImage(fname).convertToFormat(QImage.Format.Format_RGB888)
        )
        img_q = self.qimage  # readability

        # check height and width for the need to be corrected (addition of a
        # border) to achieve integer divisibility
        img_height_border = (
            (self.puzzle_size - (img_q.height() % self.puzzle_size))
            % self.puzzle_size
        )
        img_width_border = (
            (self.puzzle_size - (img_q.width() % self.puzzle_size))
            % self.puzzle_size
        )

        # add border, if necessary
        if img_height_border > 0 or img_width_border > 0:
            # buf = img_q.bits()
            # buf.setsize(img_q.sizeInBytes())
            # print(buf, buf.getsize())
            # print(img_q.height(), img_q.width(), img_q.sizeInBytes(), img_q.bytesPerLine())
            # img_np = np.frombuffer(buf, np.uint8).reshape(
            #    img_q.height(), img_q.width(), 3)
            buf = img_q.constBits()
            buf.setsize(img_q.sizeInBytes())

            img_np = np.frombuffer(buf, dtype=np.uint8)
            img_np = img_np.reshape(img_q.height(), img_q.bytesPerLine())
            img_np = img_np[:, :img_q.width() * 3].reshape(
                img_q.height(), img_q.width(), 3
            )

            img_np_corr = np.ones(
                (
                    img_np.shape[0] + img_height_border,
                    img_np.shape[1] + img_width_border,
                    img_np.shape[2]
                ),
                np.uint8,
            )*255  # white border

            img_height_border_half = img_height_border//2
            img_width_border_half = img_width_border//2
            img_np_corr[
                img_height_border_half:img_height_border_half+img_np.shape[0],
                img_width_border_half:img_width_border_half+img_np.shape[1],
                :
            ] = img_np[:, :, :]

            img_q = QImage(
                img_np_corr.data,
                img_np_corr.shape[1],
                img_np_corr.shape[0],
                img_np_corr.strides[0],
                QImage.Format.Format_RGB888,
            )

        self.pixmap = QPixmap.fromImage(img_q)

        # TODO: should be defined as properties (in a Python sense)
        self.height, self.width = self.pixmap.height(), self.pixmap.width()


class Puzzle:
    def __init__(self, size, missing_tile=None):
        # XXXXX: getter/property for size and size squared
        self.size = size
        self.size_sq = size**2

        self.rng = random.default_rng(seed=4)
        self.num = 0
        self.is_solvable = None

        self.missingTileRow = self.size-1
        self.missingTileCol = self.size-1

    # XXXXX: should be class or instance method?
    def compute_solvability(self):
        """
        check is puzzle is solvable by calculating:
        1) the parity of the permutation (cell_positions)
        2) the taxicab distance between the initial
        position of the missing tile and the final
        position of such tile
        https://en.wikipedia.org/wiki/15_puzzle#Solvability
        https://en.wikipedia.org/wiki/Parity_of_a_permutation
        the taxicab/Manhattan distance is also
        the 1-norm or l1-norm of a vector
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm
        """

        positionCombinations = list(
            itertools.combinations(range(self.size_sq), 2)
        )
        numInversions = 0
        for comb in positionCombinations:
            if self.tile_pos[comb[0]] > self.tile_pos[comb[1]]:
                numInversions += 1

        taxicab_dist = np.linalg.norm(
            x=[
                self.missingTileScrambledRow-self.missingTileRow,
                self.missingTileScrambledCol-self.missingTileCol,
            ],
            ord=1,
            axis=None
        )

        self.is_solvable = (numInversions+taxicab_dist) % 2 == 0

    def generate_puzzle(self, solvable_only: bool = True, puzzle_num: int = 0):
        """
        generate a new puzzle (solvable, by default)
        """
        iter = 0
        while True:
            self.tile_pos = self.rng.permutation(self.size_sq)
            self.missingTileScrambledIdx = int(self.tile_pos.argmax())
            self.missingTileScrambledRow = (
                self.missingTileScrambledIdx // self.size
            )
            self.missingTileScrambledCol = (
                self.missingTileScrambledIdx % self.size
            )
            self.compute_solvability()

            if solvable_only and not self.is_solvable:
                iter += 1

                # to prevent an infinite loop (in theory)
                if iter >= 500:
                    raise Exception(
                        'could not generate solvable puzzle within 500 iterations')

                continue

            self.num += 1
            if puzzle_num != 0 and self.num < puzzle_num:
                iter = 0
                continue

            break

        # backup original tile positions (unsolved puzzle)
        self.tile_pos_initial = self.tile_pos.copy()
        self.tile_pos_final = np.arange(self.size_sq)
        self.missing_tile_position_initial = (
            self.missingTileScrambledIdx,
            self.missingTileScrambledRow,
            self.missingTileScrambledCol,
        )

        self.tile_dist = -1*np.ones(self.size_sq, dtype=float)
        self.update_dist()

    def switch_tiles(self, pos):
        self.tile_pos[pos] = np.flip(self.tile_pos[pos])
        # update_dist should go into a setter method for Puzzle.tile_pos?
        self.update_dist(pos)

    def update_dist(self, idxs=None):
        """
        calculate tile distances from their current positions to their
        final positions (puzzle solved) for the specified tiles (idxs)
        """
        if idxs is None:
            idxs = np.arange(self.size_sq)

        tile_pos_row_diff = (
            self.tile_pos[idxs] // self.size
            - self.tile_pos_final[idxs] // self.size
        )
        tile_pos_col_diff = (
            self.tile_pos[idxs] % self.size
            - self.tile_pos_final[idxs] % self.size
        )

        self.tile_dist[idxs] = np.sqrt(
            tile_pos_row_diff**2 + tile_pos_col_diff**2
        )


class ThreadPuzzle(QThread):
    """
    Worker thread that takes care of handling the computations concerning
    the puzzle object, such as generating and solving the puzzle, and also
    of receiving input/orders and emitting signals to the GUI thread
    """
    # why are signals in the book as class attributes and not
    # instance attributes?
    data = pyqtSignal(list)
    update_solvable = pyqtSignal(str)
    update_puzzle_num = pyqtSignal(str)
    enable_keys_draw_puzzle_heatmap = pyqtSignal()

    move_missing_tile = pyqtSignal(str, bool)
    finished_solving = pyqtSignal()
    error_solving = pyqtSignal()
    stopped_solving = pyqtSignal()

    def __init__(self, puzzle):
        super().__init__()
        self.puzzle = puzzle

        self.stop = False

        self.solve = False
        self.pending_moves = 0

        self.stop_solve = False

    # needs @pyqtSlot() decorator or not?
    def new_puzzle(self):
        print('[ThreadPuzzle] Generating new puzzle (#', end='')
        # XXXXX: move puzzle generation/label setting to thread
        # (emit signal to change label text)
        self.puzzle.generate_puzzle()
        print(f'{self.puzzle.num})')

        self.update_solvable.emit("Yes" if self.puzzle.is_solvable else "No")
        self.update_puzzle_num.emit(str(self.puzzle.num))
        self.enable_keys_draw_puzzle_heatmap.emit()

    def reset_puzzle(self):
        self.puzzle.tile_pos = self.puzzle.tile_pos_initial.copy()
        # update_dist should go into a setter method for Puzzle.tile_pos
        self.puzzle.update_dist()

        (
            self.puzzle.missingTileScrambledIdx,
            self.puzzle.missingTileScrambledRow,
            self.puzzle.missingTileScrambledCol,
        ) = self.puzzle.missing_tile_position_initial

        self.enable_keys_draw_puzzle_heatmap.emit()

    # @pyqtSlot() # ?
    def solve_puzzle(self):
        self.solve = True

    def stop_solving(self):
        self.stop_solve = True

    # @pyqtSlot() # ?
    def move_performed(self):
        self.pending_moves -= 1
        # print('[t_puzzle--] pending: ', self.pending_moves)

    @pyqtSlot()
    def run(self):
        print("[ThreadPuzzle] started")
        self.new_puzzle()
        # TODO: process data in worker thread,
        # emit signal when done, only then use
        # the GUI thread to plot
        while True:
            if self.solve:
                self.solve = False
                self.solving_algorithm()

            if self.stop:
                print("[ThreadPuzzle] stopping...")
                break

            time.sleep(0.1)

        print("[ThreadPuzzle] stopped")

    def solving_algorithm(self):
        # entender variaveis/algoritmo
        # rever funcao self.executeMoveSequenceFinalMove
        # e remover funcoes preexistentes/ja nao possiveis
        # de utilizar

        # solve puzzle from top to bottom, left to right
        currentIdx, currentRow, currentCol = (-1, -1, -1)
        while currentIdx < self.puzzle.size_sq-1:
            if self.stop_solve:
                break

            currentIdx += 1
            currentRow = currentIdx // self.puzzle.size
            currentCol = currentIdx % self.puzzle.size

            # print('current idx, row, col: ', currentIdx, currentRow, currentCol)
            if (
                self.puzzle.tile_pos[currentIdx]
                == self.puzzle.tile_pos_final[currentIdx]
                and currentCol < self.puzzle.size-2
                and currentRow < self.puzzle.size-2
            ):
                continue

            # if all the tiles considered so far are at their final
            # positions, continue, otherwise the algorithm is not
            # correctly solving the puzzle
            if not (
                (
                    self.puzzle.tile_pos[:currentIdx]
                    == self.puzzle.tile_pos_final[:currentIdx]
                ).all()
            ):
                break

            # current position: the position currently being solved
            #                   currentIdx, currentRow, currentCol
            # set tile currentIdx as tile of interest
            # tile of interest: the tile that is being considered will be moved
            #                   (to the target position)
            #                   self.tileIdx, self.tileRow, self.tileCol
            # target position: the position to which the tile of interest shall
            #                  be moved
            self.tileIdx = int(
                np.where(self.puzzle.tile_pos == currentIdx)[0][0]
            )
            self.tileRow = self.tileIdx // self.puzzle.size
            self.tileCol = self.tileIdx % self.puzzle.size
            if currentRow < self.puzzle.size-2:
                # solve all the rows except the last two in this way
                if currentCol < self.puzzle.size-2:
                    # for the same row, solve all the columns except
                    # the last two in this way

                    # first element: horizontal movement type (left/right)
                    # second element: vertical movement type (up/down)
                    # third element: switch direction (left/right)

                    # XXXXX: correct this while loop by considering that
                    # the sequence of moves will be different depending on
                    # whether self.tileCol < currentCol or
                    # self.tileCol > currentCol
                    while self.tileCol != currentCol:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledRow > self.tileRow:
                            if self.tileRow > currentRow:
                                if self.tileCol < currentCol:
                                    if (
                                        self.puzzle.missingTileScrambledCol
                                        <= self.tileCol
                                    ):
                                        moveSequence.append(
                                            ("right", self.tileCol+1)
                                        )
                                    else:
                                        moveSequence.append(
                                            ("left", self.tileCol+1)
                                        )
                                    moveSequence.append(("up", self.tileRow))
                                    final_move = 'left'
                                elif self.tileCol > currentCol:
                                    if (
                                        self.puzzle.missingTileScrambledCol
                                        < self.tileCol
                                    ):
                                        moveSequence.append(
                                            ("right", self.tileCol-1)
                                        )
                                    else:
                                        moveSequence.append(
                                            ("left", self.tileCol-1)
                                        )
                                    moveSequence.append(("up", self.tileRow))
                                    final_move = 'right'
                                else:
                                    raise Exception('should not happen')
                            elif self.tileRow == currentRow:
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    < self.tileCol
                                ):
                                    moveSequence.append(
                                        ("right", self.tileCol-1)
                                    )
                                else:
                                    # self.puzzle.missingTileScrambledCol
                                    # >= self.tileCol
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                moveSequence.append(("up", self.tileRow))
                                final_move = 'right'
                            else:  # self.tileRow < currentRow
                                raise Exception('should not happen!')
                        elif (
                            self.puzzle.missingTileScrambledRow < self.tileRow
                        ):
                            # implies self.tileRow > currentRow
                            if self.tileRow <= currentRow:
                                raise Exception('should not happen!')

                            if self.tileCol < currentCol:
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    <= self.tileCol
                                ):
                                    moveSequence.append(
                                        ("right", self.tileCol+1)
                                    )
                                else:
                                    # XXXXX: why the next line?!
                                    moveSequence.append(
                                        (
                                            "down",
                                            self.puzzle.missingTileScrambledRow+1
                                        )
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol+1)
                                    )
                                final_move = 'left'
                            elif self.tileCol > currentCol:
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    < self.tileCol
                                ):
                                    moveSequence.append(
                                        ("right", self.tileCol-1)
                                    )
                                else:
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                final_move = 'right'
                            moveSequence.append(("down", self.tileRow))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # == self.tileRow
                            if self.tileRow > currentRow:
                                if self.tileCol < currentCol:
                                    if (
                                        self.puzzle.missingTileScrambledCol
                                        < self.tileCol
                                    ):
                                        # if missing tile and tile of interest
                                        # are in the last row, move missing
                                        # tile up as first move, otherwise move
                                        # the missing tile down
                                        if (
                                            self.puzzle.missingTileScrambledRow
                                            == self.puzzle.size-1
                                        ):
                                            moveSequence.append(
                                                ("up", self.tileRow-1)
                                            )
                                            moveSequence.append(
                                                ("right", self.tileCol+1)
                                            )
                                            moveSequence.append(
                                                ("down", self.tileRow)
                                            )
                                        else:
                                            moveSequence.append(
                                                ("down", self.tileRow+1)
                                            )
                                            moveSequence.append(
                                                ("right", self.tileCol+1)
                                            )
                                            moveSequence.append(
                                                ("up", self.tileRow)
                                            )
                                    elif (
                                        self.puzzle.missingTileScrambledCol
                                        > self.tileCol
                                    ):
                                        moveSequence.append(
                                            ("left", self.tileCol+1)
                                        )
                                    else:
                                        raise Exception('should not happen!')
                                    final_move = 'left'
                                elif self.tileCol > currentCol:
                                    if (
                                        self.puzzle.missingTileScrambledCol
                                        < self.tileCol
                                    ):
                                        moveSequence.append(
                                            ("right", self.tileCol-1)
                                        )
                                    elif (
                                        self.puzzle.missingTileScrambledCol
                                        > self.tileCol
                                    ):
                                        # if missing tile and tile of interest
                                        # are in the last row, move missing
                                        # tile up as first move, otherwise move
                                        # missing tile down
                                        if (
                                            self.puzzle.missingTileScrambledRow
                                            == self.puzzle.size-1
                                        ):
                                            moveSequence.append(
                                                ("up", self.tileRow-1)
                                            )
                                            moveSequence.append(
                                                ("left", self.tileCol-1)
                                            )
                                            moveSequence.append(
                                                ("down", self.tileRow)
                                            )
                                        else:
                                            moveSequence.append(
                                                ("down", self.tileRow+1)
                                            )
                                            moveSequence.append(
                                                ("left", self.tileCol-1)
                                            )
                                            moveSequence.append(
                                                ("up", self.tileRow)
                                            )
                                    else:
                                        raise Exception('should not happen!')
                                    final_move = 'right'
                                else:
                                    raise Exception('should not happen')
                            elif self.tileRow == currentRow:
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    < self.tileCol
                                ):
                                    moveSequence.append(
                                        ("right", self.tileCol-1)
                                    )
                                elif (
                                    self.puzzle.missingTileScrambledCol
                                    > self.tileCol
                                ):
                                    moveSequence.append(
                                        ("down", self.tileRow+1)
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                    moveSequence.append(("up", self.tileRow))
                                else:
                                    # self.puzzle.missingTileScrambledCol
                                    # == self.tileCol
                                    raise Exception('should not happen!')
                                final_move = 'right'
                            else:  # self.tileRow < currentRow
                                raise Exception('should not happen!')

                        # execute determined move sequence
                        self.executeMoveSequenceFinalMove(
                            moveSequence, final_move
                        )

                    # now that self.tileCol == currentCol, the tile of interest
                    # should be moved up in order to achieve
                    # self.tileRow == currentRow
                    while self.tileRow != currentRow:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledRow > self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                if self.tileRow == currentRow+1:
                                    moveSequence.append(
                                        ("right", self.tileCol+1)
                                    )
                                    moveSequence.append(
                                        ("up", self.tileRow-1)
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol)
                                    )
                                elif self.tileRow > currentRow+1:
                                    moveSequence.append(
                                        ("up", self.tileRow-1)
                                    )
                                    moveSequence.append(
                                        ("right", self.tileCol)
                                    )
                                else:
                                    raise Exception('should not happen!')
                            else:
                                # self.puzzle.missingTileScrambledCol
                                # >= self.tileCol:
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    == self.tileCol
                                ):
                                    moveSequence.append(
                                        ("right", self.tileCol+1)
                                    )
                                moveSequence.append(("up", self.tileRow-1))
                                moveSequence.append(("left", self.tileCol))
                        elif (
                            self.puzzle.missingTileScrambledRow < self.tileRow
                        ):
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol))
                            elif (
                                self.puzzle.missingTileScrambledCol
                                > self.tileCol
                            ):
                                moveSequence.append(("left", self.tileCol))
                            moveSequence.append(("down", self.tileRow-1))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # == self.tileRow
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                if self.tileRow == currentRow+1:
                                    moveSequence.append(
                                        ("down", self.tileRow+1)
                                    )
                                    moveSequence.append(
                                        ("right", self.tileCol+1)
                                    )
                                    moveSequence.append(("up", self.tileRow-1))
                                    moveSequence.append(("left", self.tileCol))
                                elif self.tileRow > currentRow+1:
                                    moveSequence.append(("up", self.tileRow-1))
                                    moveSequence.append(
                                        ("right", self.tileCol)
                                    )
                                else:
                                    raise Exception('should not happen!')
                            elif (
                                self.puzzle.missingTileScrambledCol
                                > self.tileCol
                            ):
                                moveSequence.append(("up", self.tileRow-1))
                                moveSequence.append(("left", self.tileCol))

                            else:
                                raise NotImplementedError

                        self.executeMoveSequenceFinalMove(moveSequence, 'down')

                else:
                    # the last two tiles of a row are solved jointly:
                    # 1) first, by moving the penultimate tile of the row
                    #    (currentRow) to the last position of such (a?) row,
                    #    i.e., by moving the tile of position
                    #    (currentRow, self.puzzle.size-2) to position
                    #    (currentRow, self.puzzle.size-1)
                    # 2) then, by moving the last tile of the row to the last
                    #    position of the *next* row, i.e., by moving the tile
                    #    of position (currentRow, self.puzzle.size-1) to
                    #    position (currentRow+1, self.puzzle.size-1)
                    # 3) finally, by carrying out a predefined sequence of
                    #    moves so that the two last tiles of the row are moved
                    #    to their correct positions, thus completing that row

                    # set next (last) column of the row as target position
                    currentIdx += 1
                    currentCol += 1

                    # check if the two last tiles of the row are already in
                    # their final positions, to avoid unnecessary moves
                    if (
                        np.where(self.puzzle.tile_pos == currentIdx-1)[0][0]
                        == currentIdx-1
                        and np.where(self.puzzle.tile_pos == currentIdx)[0][0]
                        == currentIdx
                    ):
                        continue

                    # move the penultimate tile of the row, horizontally,
                    # to the desired column
                    while self.tileCol != self.puzzle.size-1:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledCol <= self.tileCol:
                            if (
                                self.puzzle.missingTileScrambledRow
                                == self.tileRow
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    != self.puzzle.size-1
                                ):
                                    moveSequence.append(
                                        ("down", self.tileRow+1)
                                    )
                                else:
                                    moveSequence.append(("up", self.tileRow-1))
                            moveSequence.append(("right", self.tileCol+1))
                        else:
                            # self.puzzle.missingTileScrambledCol
                            # > self.tileCol
                            if (
                                self.puzzle.missingTileScrambledRow
                                < self.tileRow
                            ):
                                # to prevent displacing tiles previously
                                # moved to their correct positions
                                moveSequence.append(("down", currentRow+1))
                            moveSequence.append(("left", self.tileCol+1))

                        if self.puzzle.missingTileScrambledRow > self.tileRow:
                            moveSequence.append(("up", self.tileRow))
                        elif (
                            self.puzzle.missingTileScrambledRow < self.tileRow
                        ):
                            moveSequence.append(("down", self.tileRow))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # == self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    != self.puzzle.size-1
                                ):
                                    moveSequence.append(("up", self.tileRow))
                                else:
                                    moveSequence.append(("down", self.tileRow))

                        # carry out determined move sequence
                        self.executeMoveSequenceFinalMove(moveSequence, 'left')

                    # move the penultimate tile of the row, vertically,
                    # to the desired row
                    while self.tileRow != currentRow:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledRow >= self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol-1))
                            elif (
                                self.puzzle.missingTileScrambledCol
                                == self.tileCol
                            ):
                                moveSequence.append(("left", self.tileCol-1))
                            else:
                                raise Exception('should not happen')
                            moveSequence.append(("up", self.tileRow-1))
                            moveSequence.append(("right", self.tileCol))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # < self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol))
                            moveSequence.append(("down", self.tileRow-1))

                        # execute determined move sequence
                        self.executeMoveSequenceFinalMove(moveSequence, 'down')

                    # move the last tile of the row to the last column of
                    # the row *next* to the row being solved

                    # set the tile of interest as the last tile of the row
                    self.tileIdx = int(np.where(
                        self.puzzle.tile_pos == currentIdx)[0][0]
                    )
                    self.tileRow = self.tileIdx // self.puzzle.size
                    self.tileCol = self.tileIdx % self.puzzle.size

                    # *** special case ***: if the last tile of the row being
                    # solved (current tile of interest) is "cornered", position
                    # (currentRow, self.puzzle.size-2), index
                    # currentRow+self.puzzle.size-2=currentIdx-1,
                    # the missing tile is moved to immediately below the
                    # tile of interest and switched with the latter in order to
                    # "uncorner" it
                    if self.tileIdx == currentIdx-1:
                        moveSequence = []
                        if (
                            self.puzzle.missingTileScrambledCol
                            < self.tileCol
                        ):
                            moveSequence.append(("right", self.tileCol))
                        elif (
                            self.puzzle.missingTileScrambledCol > self.tileCol
                        ):
                            moveSequence.append(("left", self.tileCol))
                        else:
                            # missing tile does not need to be moved
                            # horizontally (it is already in the same
                            # column as the last tile of the row), only
                            # vertically (to row currentRow+1)
                            pass
                        if (
                            self.puzzle.missingTileScrambledRow
                            > self.tileRow+1
                        ):
                            moveSequence.append(("up", self.tileRow+1))

                        # execute determined move sequence plus final move
                        # self.moveMissingTileUp()
                        # self.tileIdx += self.puzzle.size
                        # self.tileRow += 1
                        self.executeMoveSequenceFinalMove(moveSequence, 'up')

                    # *** special cases: as a result of the moves so far
                    # (also possibly due to the previous special case), the
                    # missing tile is in position
                    # (currentRow, self.puzzle.size-2)

                    # XXXXX: replace row/col comparisons with index
                    # comparisons
                    if (
                        (
                            self.puzzle.missingTileScrambledRow,
                            self.puzzle.missingTileScrambledCol
                        ) == (currentRow, self.puzzle.size-2)
                    ):
                        # if, additionally, the tile of interest is
                        # immediately below the missing tile, a predefined
                        # sequence of moves is carried out to move the tile
                        # of interest to the target position (after that,
                        # the conditions of the two next while loops will
                        # evaluate to false)
                        if (
                            (self.tileRow, self.tileCol)
                            == (currentRow+1, self.puzzle.size-2)
                        ):
                            # self.moveMissingTileRight()
                            # self.moveMissingTileDown()
                            # self.moveMissingTileLeft()
                            # self.tileIdx += 1
                            # self.tileCol += 1
                            moveSequence = []
                            moveSequence.append(
                                (
                                    "right",
                                    self.puzzle.missingTileScrambledCol+1
                                )
                            )
                            moveSequence.append(
                                ("down", self.puzzle.missingTileScrambledRow+1)
                            )
                            self.executeMoveSequenceFinalMove(
                                moveSequence, 'left'
                            )

                            # self.moveMissingTileDown()
                            # self.moveMissingTileRight()
                            # self.moveMissingTileUp()
                            # self.tileIdx += self.puzzle.size
                            # self.tileRow += 1
                            moveSequence = []
                            moveSequence.append(
                                ("down", self.puzzle.missingTileScrambledRow+1)
                            )
                            moveSequence.append(
                                (
                                    "right",
                                    self.puzzle.missingTileScrambledCol+1
                                )
                            )
                            self.executeMoveSequenceFinalMove(
                                moveSequence, 'up'
                            )

                            # self.moveMissingTileUp()
                            # self.moveMissingTileLeft()
                            # self.moveMissingTileDown()
                            # self.moveMissingTileRight()
                            # self.moveMissingTileDown()
                            # self.tileIdx -= self.puzzle.size
                            # self.tileRow -= 1
                            moveSequence = []
                            moveSequence.append(
                                ("up", self.puzzle.missingTileScrambledRow-1)
                            )
                            moveSequence.append(
                                ("left", self.puzzle.missingTileScrambledCol-1)
                            )
                            moveSequence.append(
                                ("down", self.puzzle.missingTileScrambledRow)
                            )
                            moveSequence.append(
                                ("right", self.puzzle.missingTileScrambledCol)
                            )
                            self.executeMoveSequenceFinalMove(
                                moveSequence, "down"
                            )

                            if (
                                (self.tileRow, self.tileCol)
                                != (currentRow+1, self.puzzle.size-1)
                            ):
                                # XXXXX: test/debug, remove this later
                                raise ValueError(
                                    'unexpected/something went wrong!'
                                )
                        else:
                            # just move the missing tile down, since the
                            # logic in the next while loop assumes the
                            # missing tile is not in the row currently
                            # being solved (not doing this may lead to the
                            # puzzle not being correctly solved by this
                            # algorithm)
                            # self.moveMissingTileDown()
                            moveSequence = []
                            moveSequence.append(
                                ("down", self.puzzle.missingTileScrambledRow+1)
                            )
                            self.executeMoveSequenceFinalMove(moveSequence)

                    # set last column of the next row as target position
                    currentIdx += self.puzzle.size
                    currentRow += 1

                    # move the last tile of the row, horizontally, to the
                    # desired column
                    while self.tileCol != self.puzzle.size-1:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledCol <= self.tileCol:
                            if (
                                self.puzzle.missingTileScrambledRow
                                == self.tileRow
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    != self.puzzle.size-1
                                ):
                                    moveSequence.append(
                                        ("down", self.tileRow+1)
                                    )
                                else:
                                    moveSequence.append(("up", self.tileRow-1))
                            moveSequence.append(("right", self.tileCol+1))
                        else:
                            moveSequence.append(("left", self.tileCol+1))

                        if self.puzzle.missingTileScrambledRow > self.tileRow:
                            moveSequence.append(("up", self.tileRow))
                        elif (
                            self.puzzle.missingTileScrambledRow < self.tileRow
                        ):
                            moveSequence.append(("down", self.tileRow))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # == self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    != self.puzzle.size-1
                                ):
                                    moveSequence.append(("up", self.tileRow))
                                else:
                                    moveSequence.append(("down", self.tileRow))

                        # carry out determined move sequence
                        self.executeMoveSequenceFinalMove(moveSequence, 'left')

                    # move the last tile of the row, vertically, to the desired
                    # row
                    while self.tileRow != currentRow:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledRow >= self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                <= self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    == self.tileCol
                                ):
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                moveSequence.append(("up", self.tileRow-1))
                                moveSequence.append(("right", self.tileCol))
                            else:
                                raise Exception('should not happen')
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # < self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                <= self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledCol
                                    < self.tileCol
                                ):
                                    moveSequence.append(
                                        ("right", self.tileCol)
                                    )
                                moveSequence.append(("down", self.tileRow-1))
                            else:
                                raise Exception('should not happen!')

                        # execute determined move sequence
                        self.executeMoveSequenceFinalMove(moveSequence, 'down')

                    # at this point, the tile of interest should be at target
                    # position (currentRow, currentCol)
                    # the missing tile is then moved horizontally to
                    # column self.puzzle.size-2, and then vertically to
                    # row current_row

                    # set penultimate column of the current row as target
                    # position of the missing tile
                    currentIdx -= self.puzzle.size + 1
                    currentRow -= 1
                    currentCol -= 1

                    moveSequence = []
                    if (
                        self.puzzle.missingTileScrambledCol
                        != self.puzzle.size-2
                    ):
                        if (
                            self.puzzle.missingTileScrambledCol
                            < self.puzzle.size-2
                        ):
                            moveSequence.append(("right", self.puzzle.size-2))
                        elif (
                            self.puzzle.missingTileScrambledCol
                            > self.puzzle.size-2
                        ):
                            moveSequence.append(("left", self.puzzle.size-2))
                    if self.puzzle.missingTileScrambledRow != currentRow:
                        moveSequence.append(("up", currentRow))

                    # carry out a final, predefined sequence of moves to
                    # move the two last tiles of the row to their correct
                    # positions
                    # self.moveMissingTileRight()
                    moveSequence.append(("right", self.puzzle.size-1))
                    # self.moveMissingTileDown()
                    # self.tileIdx -= self.puzzle.size
                    # self.tileRow -= 1
                    self.executeMoveSequenceFinalMove(moveSequence, 'down')

                    currentIdx += 1
                    currentCol += 1
            else:
                # the two last rows are solved jointly, column by column, from
                # left to right:
                # 1) first, by moving the tile of the lower (last) row to
                #    its final column, but to the upper (penultimate) row,
                # 2) and by moving the tile of the upper row to its final
                #    row, but to the next column
                # 3) finally, a predefined sequence of moves is carried out
                #    so that the tiles of the two last rows and of the
                #    same column (currentCol) are moved to their correct
                #    positions, thus completing that column
                # 4) the two last columns are solved differently, either by
                #    moving ("rotating") the missing tile in a clockwise or
                #    counter-clockwise fashion
                if currentCol < self.puzzle.size-2:

                    # check if the two last tiles of the column are already in
                    # their correct positions, to avoid unnecessary moves
                    if (
                        np.where(
                            self.puzzle.tile_pos == currentIdx
                        )[0][0] == currentIdx
                        and
                        np.where(
                            self.puzzle.tile_pos == currentIdx
                            + self.puzzle.size
                        )[0][0] == currentIdx + self.puzzle.size
                    ):
                        continue

                    # set tile of the lower (last) row as tile of interest
                    self.tileIdx = int(np.where(
                        self.puzzle.tile_pos == currentIdx+self.puzzle.size
                    )[0][0])
                    self.tileRow = self.tileIdx // self.puzzle.size
                    self.tileCol = self.tileIdx % self.puzzle.size

                    # move the tile of the lower row to the desired column
                    while self.tileCol != currentCol:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledRow != self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol-1))
                            else:
                                moveSequence.append(("left", self.tileCol-1))

                            if (
                                self.puzzle.missingTileScrambledRow
                                > self.tileRow
                            ):
                                moveSequence.append(("up", self.tileRow))
                            else:
                                moveSequence.append(("down", self.tileRow))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # == self.tileRow
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol-1))
                            elif (
                                self.puzzle.missingTileScrambledCol
                                > self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    == self.puzzle.size-1
                                ):
                                    moveSequence.append(
                                        ("up", self.puzzle.size-2)
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                    moveSequence.append(
                                        ("down", self.puzzle.size-1)
                                    )
                                else:
                                    moveSequence.append(
                                        ("down", self.puzzle.size-1)
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                    moveSequence.append(
                                        ("up", self.puzzle.size-2)
                                    )
                            else:
                                raise Exception('should not happen!')

                        self.executeMoveSequenceFinalMove(
                            moveSequence, 'right'
                        )

                    # move the tile of the lower row to the desired row
                    while self.tileRow != currentRow:
                        moveSequence = []
                        if self.puzzle.missingTileScrambledCol > self.tileCol:
                            if (
                                self.puzzle.missingTileScrambledRow
                                == self.tileRow
                            ):
                                moveSequence.append(("up", self.puzzle.size-2))
                            moveSequence.append(("left", self.tileCol))
                        elif (
                            self.puzzle.missingTileScrambledCol < self.tileCol
                        ):
                            raise Exception('should not happen!')
                        self.executeMoveSequenceFinalMove(moveSequence, 'down')

                    # set tile of the upper (penultimate) row as tile of
                    # interest
                    self.tileIdx = int(np.where(
                        self.puzzle.tile_pos == currentIdx)[0][0]
                    )
                    self.tileRow = self.tileIdx // self.puzzle.size
                    self.tileCol = self.tileIdx % self.puzzle.size

                    # move the tile of the upper row to the desired column
                    # *** the next while loop handles the (two) special cases
                    # where the tile of the upper row is in position
                    # (currentRow+1, currentCol) =
                    # (self.puzzle.size-1, currentCol), i.e., "cornered"
                    # this tile should, however, end up in position
                    # (currentRow, currentCol+1) before the final,
                    # predefined sequence of moves is carried out

                    # set next column of the upper row as target position
                    currentIdx += 1
                    currentCol += 1
                    while self.tileCol != currentCol:
                        moveSequence = []
                        final_move = 'right'
                        if self.puzzle.missingTileScrambledRow != self.tileRow:
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol-1))
                            else:  # >=
                                if self.tileCol > currentCol:
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                elif self.tileCol < currentCol:
                                    # tile of interest is "cornered" and
                                    # missing tile is in the upper row
                                    moveSequence.append(
                                        ("down", self.tileRow)
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol+1)
                                    )
                                    final_move = 'left'
                                else:
                                    raise Exception('should not happen!')
                            if (
                                self.puzzle.missingTileScrambledRow
                                > self.tileRow
                            ):
                                moveSequence.append(("up", self.tileRow))
                            else:
                                moveSequence.append(("down", self.tileRow))
                        else:
                            # self.puzzle.missingTileScrambledRow
                            # == self.tileRow
                            if (
                                self.puzzle.missingTileScrambledCol
                                < self.tileCol
                            ):
                                moveSequence.append(("right", self.tileCol-1))
                            elif (
                                self.puzzle.missingTileScrambledCol
                                > self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    == self.puzzle.size-1
                                ):
                                    if self.tileCol > currentCol:
                                        moveSequence.append(
                                            ("up", self.puzzle.size-2)
                                        )
                                        moveSequence.append(
                                            ("left", self.tileCol-1)
                                        )
                                        moveSequence.append(
                                            ("down", self.puzzle.size-1)
                                        )
                                    elif self.tileCol < currentCol:
                                        # tile of interest is "cornered" and
                                        # missing tile is in the lower row
                                        moveSequence.append(
                                            ("left", self.tileCol+1)
                                        )
                                        final_move = 'left'
                                    else:
                                        raise Exception('should not happen!')
                                else:
                                    moveSequence.append(
                                        ("down", self.puzzle.size-1)
                                    )
                                    moveSequence.append(
                                        ("left", self.tileCol-1)
                                    )
                                    moveSequence.append(
                                        ("up", self.puzzle.size-2)
                                    )
                            else:
                                raise Exception('should not happen!')

                        # execute determined move sequence
                        self.executeMoveSequenceFinalMove(
                            moveSequence, final_move
                        )

                    # the tile of the upper row should now be in the
                    # desired column

                    # move the tile of the upper row to the desired row
                    # *** the next while loop handles the special cass
                    # where the missing tile is "cornered", i.e., in
                    # position (currentRow+1, currentCol-1)
                    while self.tileRow != currentRow:
                        moveSequence = []
                        if (
                            self.puzzle.missingTileScrambledCol >= self.tileCol
                        ):
                            if (
                                self.puzzle.missingTileScrambledCol
                                > self.tileCol
                            ):
                                if (
                                    self.puzzle.missingTileScrambledRow
                                    == self.tileRow
                                ):
                                    moveSequence.append(
                                        ("up", self.puzzle.size-2)
                                    )
                                moveSequence.append(
                                    ("left", self.tileCol)
                                )
                            else:
                                # (
                                #     self.puzzle.missingTileScrambledCol
                                #     == self.tileCol
                                # )
                                pass
                            self.executeMoveSequenceFinalMove(
                                moveSequence, "down"
                            )
                        else:
                            # (
                            #     self.puzzle.missingTileScrambledCol
                            #     < self.tileCol:
                            # )
                            # special case, carry out a predefined sequence
                            # of moves

                            # self.moveMissingTileUp()
                            # self.moveMissingTileRight()
                            # self.moveMissingTileDown()
                            # self.tileIdx -= self.puzzle.size
                            # self.tileRow -= 1
                            moveSequence = []
                            moveSequence.append(
                                ("up", self.puzzle.missingTileScrambledRow-1)
                            )
                            moveSequence.append(
                                (
                                    "right",
                                    self.puzzle.missingTileScrambledCol+1,
                                )
                            )
                            self.executeMoveSequenceFinalMove(
                                moveSequence, "down"
                            )

                            # self.moveMissingTileRight()
                            # self.moveMissingTileUp()
                            # self.moveMissingTileLeft()
                            # self.tileIdx += 1
                            # self.tileCol += 1
                            moveSequence = []
                            moveSequence.append(
                                (
                                    "right",
                                    self.puzzle.missingTileScrambledCol+1
                                )
                            )
                            moveSequence.append(
                                ("up", self.puzzle.missingTileScrambledRow-1)
                            )
                            self.executeMoveSequenceFinalMove(
                                moveSequence, "left"
                            )

                            # self.moveMissingTileLeft()
                            # self.moveMissingTileDown()
                            # self.moveMissingTileRight()
                            # self.moveMissingTileUp()
                            # self.moveMissingTileRight()
                            # self.tileIdx -= 1
                            # self.tileCol -= 1
                            moveSequence = []
                            moveSequence.append(
                                ("left", self.puzzle.missingTileScrambledCol-1)
                            )
                            moveSequence.append(
                                ("down", self.puzzle.missingTileScrambledRow+1)
                            )
                            moveSequence.append(
                                ("right", self.puzzle.missingTileScrambledCol)
                            )
                            moveSequence.append(
                                ("up", self.puzzle.missingTileScrambledRow)
                            )
                            self.executeMoveSequenceFinalMove(
                                moveSequence, "right"
                            )

                    # reset target position to previous column of the
                    # penultimate row
                    currentIdx -= 1
                    currentCol -= 1

                    # move the missing tile to the position that will
                    # allow completing the (whole) current column of the
                    # puzzle
                    moveSequence = []
                    if (
                        self.puzzle.missingTileScrambledRow
                        != self.puzzle.size-1
                    ):
                        moveSequence.append(("down", self.puzzle.size-1))
                    moveSequence.append(("left", currentCol))
                    self.executeMoveSequenceFinalMove(moveSequence)

                    # now move the missing tile up, then right
                    # self.moveMissingTileUp()
                    # self.moveMissingTileRight()
                    # self.tileIdx -= 1
                    # self.tileCol -= 1
                    moveSequence = []
                    moveSequence.append(
                        ("up", self.puzzle.missingTileScrambledRow-1)
                    )
                    self.executeMoveSequenceFinalMove(moveSequence, "right")

                else:
                    # check column of tile of index self.puzzle.size_sq-2
                    # if column is self.puzzle.size-2, rotate clockwise,
                    # otherwise rotate counter-clockwise
                    self.tileIdx = int(np.where(
                        self.puzzle.tile_pos == self.puzzle.size_sq-2)[0][0]
                    )
                    self.tileRow = self.tileIdx // self.puzzle.size
                    self.tileCol = self.tileIdx % self.puzzle.size

                    # determination of rotation direction and first
                    # movement
                    if self.tileCol == self.puzzle.size-2:
                        # clockwise
                        rotation_direction = +1

                        if (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-2,
                                    self.puzzle.size-2,
                                )
                        ):
                            next_movement = 'right'
                        elif (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-2,
                                    self.puzzle.size-1,
                                )
                        ):
                            next_movement = 'down'
                        elif (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-1,
                                    self.puzzle.size-1,
                                )
                        ):
                            next_movement = 'left'
                        elif (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-1,
                                    self.puzzle.size-2,
                                )
                        ):
                            next_movement = 'up'
                        else:
                            raise ValueError(
                                'missing tile in unexpected position'
                            )
                    else:
                        # counter-clockwise
                        rotation_direction = -1

                        if (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-2,
                                    self.puzzle.size-2,
                                )
                        ):
                            next_movement = 'down'
                        elif (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-2,
                                    self.puzzle.size-1
                                )
                        ):
                            next_movement = 'left'
                        elif (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-1,
                                    self.puzzle.size-1,
                                )
                        ):
                            next_movement = 'up'
                        elif (
                                (
                                    self.puzzle.missingTileScrambledRow,
                                    self.puzzle.missingTileScrambledCol,
                                )
                                ==
                                (
                                    self.puzzle.size-1,
                                    self.puzzle.size-2,
                                )
                        ):
                            next_movement = 'right'
                        else:
                            raise ValueError(
                                'missing tile in unexpected position'
                            )

                    # rotation stop criteria: both tiles
                    # self.puzzle.size_sq-2 and self.puzzle.size_sq-1
                    # are in the correct positions
                    while (
                        (
                            self.puzzle.tile_pos[-2:]
                            != (self.puzzle.size_sq-2, self.puzzle.size_sq-1)
                        )
                        .any()
                    ):
                        self.executeMoveSequenceFinalMove(
                            final_move=next_movement
                        )
                        if rotation_direction == +1:
                            if next_movement == 'right':
                                next_movement = 'down'
                            elif next_movement == 'down':
                                next_movement = 'left'
                            elif next_movement == 'left':
                                next_movement = 'up'
                            elif next_movement == 'up':
                                next_movement = 'right'
                        elif rotation_direction == -1:
                            if next_movement == 'right':
                                next_movement = 'up'
                            elif next_movement == 'up':
                                next_movement = 'left'
                            elif next_movement == 'left':
                                next_movement = 'down'
                            elif next_movement == 'down':
                                next_movement = 'right'

                    currentIdx = self.puzzle.size_sq-1
                    currentRow = self.puzzle.size-1
                    currentCol = self.puzzle.size-1

        if not self.stop_solve:
            # at this point, all the tiles are expected to be at their final
            # positions
            if (self.puzzle.tile_pos == self.puzzle.tile_pos_final).all():
                self.finished_solving.emit()
            else:
                self.error_solving.emit()
                raise Exception(
                    f'puzzle did not get solved! error at tile {currentIdx}'
                    f' (row {currentRow}, column {currentCol})')
        else:
            self.stop_solve = False
            self.stopped_solving.emit()

    def executeMoveSequenceFinalMove(self, move_sequence=(), final_move=None):
        for move in move_sequence:
            if move[0] == 'left':
                while self.puzzle.missingTileScrambledCol > move[1]:
                    self.send_move_decision("l")
                    # if self.moveMissingTileLeft() == -1:
                    #     raise ValueError('calculated invalid move!1')
            elif move[0] == 'up':
                while self.puzzle.missingTileScrambledRow > move[1]:
                    self.send_move_decision("u")
                    # if self.moveMissingTileUp() == -1:
                    #     raise ValueError('calculated invalid move!3')
            elif move[0] == 'right':
                while self.puzzle.missingTileScrambledCol < move[1]:
                    self.send_move_decision("r")
                    # if self.moveMissingTileRight() == -1:
                    #     raise ValueError('calculated invalid move!2')
            elif move[0] == 'down':
                while self.puzzle.missingTileScrambledRow < move[1]:
                    self.send_move_decision("d")
                    # if self.moveMissingTileDown() == -1:
                    #     raise ValueError('calculated invalid move!4')

        # move missing tile left/right/up/down, switching the missing tile
        # with the tile of interest and effectively moving the latter
        # right/left/down/up
        # print('final_move: ', final_move)
        if final_move:
            if final_move == 'left':
                self.send_move_decision("l")
                # self.moveMissingTileLeft()
                deltaRow = 0
                deltaCol = +1
            elif final_move == 'up':
                self.send_move_decision("u")
                # self.moveMissingTileUp()
                deltaRow = +1
                deltaCol = 0
            elif final_move == 'right':
                self.send_move_decision("r")
                # self.moveMissingTileRight()
                deltaRow = 0
                deltaCol = -1
            elif final_move == 'down':
                self.send_move_decision("d")
                # self.moveMissingTileDown()
                deltaRow = -1
                deltaCol = 0
            else:
                raise ValueError(
                    "\"final_move\" must take the values "
                    "{{left, right, up, down}}"
                )
            self.tileIdx += deltaRow*self.puzzle.size + deltaCol
            self.tileRow += deltaRow
            self.tileCol += deltaCol

    def send_move_decision(self, direction: str):
        self.pending_moves += 1
        self.move_missing_tile.emit(direction, True)
        # print('[t_puzzle] pending: ', self.pending_moves)
        if self.pending_moves < 0:
            raise ValueError(
                'number of pending moves should *never* be negative')
        while self.pending_moves > 0:
            time.sleep(0.01)


class SlidingTilePuzzleGUI(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.button_new = QPushButton(text="Generate new")
        self.button_reset = QPushButton(text="Reset moves")
        self.button_solve = QPushButton(text="Solve current")
        self.button_solve_inf = QPushButton(text="Solve infinitely")

        self.checkbox_wraparound = QCheckBox(text="Wraparound")

        label_puzzle_size = QLabel()
        label_puzzle_size.setStyleSheet('font-weight: bold')
        label_puzzle_size.setText('Puzzle size: ')
        self.spinbox_size_select = QSpinBox()
        self.spinbox_size_select.setRange(3, 20)
        self.spinbox_size_select.setValue(3)
        self.button_size_set = QPushButton(text='Set')

        layout_puzzle_size = QHBoxLayout()
        layout_puzzle_size.addWidget(self.spinbox_size_select)
        layout_puzzle_size.addWidget(self.button_size_set)
        layout_puzzle_size.setAlignment(Qt.AlignmentFlag.AlignTop)

        # keyboard focus settings
        widgets_buttons_checkbox = (
            self.button_new,
            self.button_reset,
            self.button_solve,
            self.button_solve_inf,
            self.checkbox_wraparound,
        )
        for widget in widgets_buttons_checkbox:
            widget.setFocusPolicy(
                # remove keyboard/tab and mouse click focus/navigation through
                # bitwise operations (clear the two least significant bits of
                # focusPolicy), so the buttons do not show as selected/
                # highlighted when solving the puzzle using the keyboard arrow
                # keys
                widget.focusPolicy() & ~Qt.FocusPolicy.TabFocus
                & ~Qt.FocusPolicy.ClickFocus
            )
        self.button_size_set.setFocusPolicy(
            widget.focusPolicy() & ~Qt.FocusPolicy.TabFocus
            & ~Qt.FocusPolicy.ClickFocus
        )
        self.spinbox_size_select.setFocusPolicy(
            widget.focusPolicy() & ~Qt.FocusPolicy.TabFocus
            & ~Qt.FocusPolicy.ClickFocus
        )

        self.keys_enabled = False  # respond to (arrow) keyboard key presses
        self.solving = False
        self.solve_infinite = False
        self.changing_puzzle_size = False

        # review/make diagrams: Git
        # understand/learn: QPixmap
        # https://duckduckgo.com/?q=qpixmap&ia=web
        # https://doc.qt.io/qt-6/qpixmap.html#copy
        #     -> copy
        # https://stackoverflow.com/questions/10307860/what-is-the-difference-between-qimage-and-qpixmap

        # horizontal and vertical spacing of the QGridLayout widgets
        h_spacing = 1
        v_spacing = 1

        # puzzle solution
        self.layoutSolution = QGridLayout()
        self.layoutSolution.setHorizontalSpacing(h_spacing)
        self.layoutSolution.setVerticalSpacing(v_spacing)
        self.layoutSolution.setAlignment(Qt.AlignmentFlag.AlignTop)

        # puzzle being solved
        self.layoutPuzzle = QGridLayout()
        self.layoutPuzzle.setHorizontalSpacing(h_spacing)
        self.layoutPuzzle.setVerticalSpacing(v_spacing)
        self.layoutPuzzle.setAlignment(Qt.AlignmentFlag.AlignTop)

        # distance-to-RGB mapping for the puzzle being solved
        self.layoutRGB = QGridLayout()
        self.layoutRGB.setHorizontalSpacing(h_spacing)
        self.layoutRGB.setVerticalSpacing(v_spacing)
        self.layoutRGB.setAlignment(Qt.AlignmentFlag.AlignTop)

        labelPuzzleNumTxt = QLabel()
        labelPuzzleNumTxt.setStyleSheet('font-weight: bold')
        labelPuzzleNumTxt.setText('Puzzle number: ')
        labelPuzzleNumTxt.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        labelMovesText = QLabel()
        labelMovesText.setStyleSheet('font-weight: bold')
        labelMovesText.setText('Moves: ')
        labelMovesText.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.puzzle_num = QLabel()
        self.puzzle_num.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.label_move_count = QLabel()
        self.label_move_count.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        labelQSolvableTxt = QLabel()
        labelQSolvableTxt.setStyleSheet('font-weight: bold')
        labelQSolvableTxt.setText('Solvable?')
        labelQSolvableTxt.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.label_solvable = QLabel()
        self.label_solvable.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        layoutButtons = QVBoxLayout()
        layoutButtons.setAlignment(Qt.AlignmentFlag.AlignTop)

        for widget in widgets_buttons_checkbox:
            layoutButtons.addWidget(widget)

        layoutButtons.addWidget(label_puzzle_size)
        layoutButtons.addLayout(layout_puzzle_size)
        layoutButtons.addWidget(labelPuzzleNumTxt)
        layoutButtons.addWidget(self.puzzle_num)
        layoutButtons.addWidget(labelMovesText)
        layoutButtons.addWidget(self.label_move_count)
        layoutButtons.addWidget(labelQSolvableTxt)
        layoutButtons.addWidget(self.label_solvable)

        layoutApp = QHBoxLayout()
        layoutApp.addLayout(self.layoutSolution)
        layoutApp.addLayout(self.layoutPuzzle)
        layoutApp.addLayout(self.layoutRGB)
        layoutApp.addLayout(layoutButtons)

        widgetApp = QWidget()
        widgetApp.setLayout(layoutApp)
        self.setCentralWidget(widgetApp)


        self.size = self.spinbox_size_select.value()
        self.puzzle = Puzzle(size=self.size)
        self.load_puzzle_image()
        self.calc_puzzle_cell_dims()
        self.drawPuzzleSolution()

        print(f'[GUI Thread] Puzzle size is {self.size}')

        self.thread_puzzle = ThreadPuzzle(self.puzzle)
        self.thread_puzzle.update_solvable.connect(
            self.label_solvable.setText
        )
        self.thread_puzzle.update_puzzle_num.connect(
            self.puzzle_num.setText
        )
        self.thread_puzzle.enable_keys_draw_puzzle_heatmap.connect(
            self.enable_keys_draw_puzzle_heatmap
        )
        self.thread_puzzle.move_missing_tile.connect(
            self.moveMissingTile
        )
        self.thread_puzzle.finished_solving.connect(
            self.solve_finished
        )
        self.thread_puzzle.stopped_solving.connect(
            self.solve_stopped
        )

        self.button_new.clicked.connect(self.thread_puzzle.new_puzzle)
        self.button_reset.clicked.connect(self.thread_puzzle.reset_puzzle)
        self.button_solve.clicked.connect(self.solve_puzzle)
        self.solve_puzzle_inf = lambda: self.solve_puzzle(infinite=True)
        self.button_solve_inf.clicked.connect(
            self.solve_puzzle_inf
        )
        self.button_size_set.clicked.connect(self.change_puzzle_size)

        self.thread_puzzle.start()

        self.move_history = None
        self.distancesHistory = None
        self.distancesSumHistory = None
        self.completionHistory = None

        self.plot_figs = None
        self.plot_axs = None


        self.size_change_confirmation_dialog = QMessageBox(
            QMessageBox.Icon.Question,
            'Puzzle size change confirmation',
            (
                'Do you really want to change the puzzle size? '
                'All progress will be reset.'
            ),
            (
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ),
        )
        self.size_change_confirmation_dialog.setDefaultButton(QMessageBox.StandardButton.No)

        self.setWindowTitle('Sliding Tile Puzzle')

        # TODO: should only show the window and enable responding to key
        # presses after the puzzle has been loaded and shown on the main window
        # not everything is guaranteed to be ready (due to threading/
        # asynchronous)
        # e.g., self.puzzle object attributes
        # self.puzzle.tile_pos_initial, self.puzzle.tile_pos_final
        # print(self.puzzle.tile_pos_final)

        self.show()

    def enable_keys_draw_puzzle_heatmap(self):
        self.keys_enabled = True
        self.drawPuzzleRGBMap()
        self.resetMovesHistory()

    def solve_puzzle(self, infinite=False):
        self.solving = True
        # disable buttons
        self.button_new.setDisabled(True)
        self.button_reset.setDisabled(True)
        self.button_solve.setDisabled(True)
        self.button_solve_inf.setDisabled(True)

        # disable responding to keyboard keys
        self.keys_enabled = False

        if infinite:
            self.solve_infinite = True
        self.thread_puzzle.solve_puzzle()

        # change slot for the clicked signal
        # change button text to stop,
        if not infinite:
            fun = self.solve_puzzle
            button = self.button_solve
        else:
            fun = self.solve_puzzle_inf
            button = self.button_solve_inf

        button.clicked.disconnect(fun)
        button.clicked.connect(self.stop_solving)
        button.setText('Solving... click to stop')
        button.setDisabled(False)

    def stop_solving(self):
        # set button to "stopping..." and disable it
        if not self.solve_infinite:
            button = self.button_solve
        else:
            button = self.button_solve_inf
        button.setDisabled(True)
        button.setText('Stopping...')

        button.clicked.disconnect(self.stop_solving)
        if not self.solve_infinite:
            self.button_solve.clicked.connect(self.solve_puzzle)
        else:
            self.button_solve_inf.clicked.connect(self.solve_puzzle_inf)

        self.thread_puzzle.stop_solving()

    def solve_finished(self):
        self.plot_solved()

        if self.solve_infinite:
            self.thread_puzzle.new_puzzle()
            self.thread_puzzle.solve_puzzle()

            return

        self.button_solve.clicked.disconnect(self.stop_solving)
        self.button_solve.clicked.connect(self.solve_puzzle)
        self.button_solve.setText('Solve')
        self.enable_buttons()
        self.solving = False
        
        if self.changing_puzzle_size:
            self.change_puzzle_size_finish()

    def solve_stopped(self):
        if not self.solve_infinite:
            self.button_solve.setText('Solve')
        else:
            self.button_solve_inf.setText('Solve (infinite)')
            self.solve_infinite = False

        self.enable_buttons()
        self.solving = False

        if self.changing_puzzle_size:
            self.change_puzzle_size_finish()

    def enable_buttons(self):
        self.button_new.setDisabled(False)
        self.button_reset.setDisabled(False)
        self.button_solve.setDisabled(False)
        self.button_solve_inf.setDisabled(False)

    def solveDP(self):
        solutionsHistory = []

        solutionsHistory.append([])
        solutionsHistory[0].append(self.puzzle.tile_pos)
        solutionsHistory[0].append(
            [
                self.puzzle.missingTileScrambledIdx,  # idx
                self.puzzle.missingTileScrambledRow,  # row
                self.puzzle.missingTileScrambledCol,  # col
            ]
        )
        # move history leading to solution
        solutionsHistory[0].append([])

        missingTileIdx = self.puzzle.size_sq - 1
        while True:
            # careful with the for loop:
            # the length of solutionsHistory will very
            # likely increase
            newSolutions = 0
            for i in range(len(solutionsHistory)):
                if (
                    (solutionsHistory[i][0] == self.puzzle.tile_pos_initial)
                    .sum() == self.puzzle.size_sq
                ):
                    # if this solution is final, no need to branch on it :)
                    continue
                sol = solutionsHistory[i]

                # for each current partial solution, find the
                # directions in which the missing tile can be
                # moved
                possibleMoves = []
                _, row, col = sol[1]
                if row > 0 and (len(sol[2]) == 0 or sol[2][-1] != -1):
                    # can move up
                    possibleMoves.append(1)
                if (
                    row < self.puzzle.size-1
                    and (len(sol[2]) == 0 or sol[2][-1] != 1)
                ):
                    # can move down
                    possibleMoves.append(-1)
                if col > 0 and (len(sol[2]) == 0 or sol[2][-1] != -2):
                    # can move left
                    possibleMoves.append(2)
                if (
                    col < self.puzzle.size-1
                    and (len(sol[2]) == 0 or sol[2][-1] != 2)
                ):
                    # can move right
                    possibleMoves.append(-2)

                # generate new solutions based on the allowed
                # movements of the missing tile
                for move in possibleMoves:
                    # anti-simmetry: -2, -1, 1, 2 instead of
                    # 0, 1, 2, 3
                    newSolutions += 1
                    if move == 1:
                        # move up
                        solutionsHistory.append([])
                        solNew = solutionsHistory[-1]
                        # solutionsHistory.append(sol.copy()) # does this work?
                        # solNew.append(sol[0].copy())
                        # solNew.append(sol[1].copy()) # shallow or deep copy?!?
                        # check how copy method of list works
                        # solNew.append(sol[2].copy()) # shallow or deep copy?!?

                        # tmpIdx = sol[1][0] - self.puzzle.size
                        # tmpTile = sol[0][tmpIdx]
                        # solNew[0][tmpIdx]    = sol[1][0]
                        # solNew[0][sol[1][0]] = sol[1][0]
                        cell_positions = sol[0].copy()
                        cell_positions[
                            sol[1][0]
                        ] = cell_positions[sol[1][0] - self.puzzle.size]
                        cell_positions[
                            sol[1][0] - self.puzzle.size
                        ] = missingTileIdx

                        missingTileIdxRowCol = sol[1].copy()
                        missingTileIdxRowCol[0] -= self.puzzle.size
                        missingTileIdxRowCol[1] -= 1
                    elif move == -1:
                        # move down
                        solutionsHistory.append([])
                        solNew = solutionsHistory[-1]

                        cell_positions = sol[0].copy()
                        # if sol[1][0] == (
                        #     29 or (sol[1][0] + self.puzzle.size) == 29
                        # ):
                        #     pass
                        cell_positions[
                            sol[1][0]
                        ] = cell_positions[sol[1][0] + self.puzzle.size]
                        cell_positions[
                            sol[1][0] + self.puzzle.size
                        ] = missingTileIdx

                        missingTileIdxRowCol = sol[1].copy()
                        missingTileIdxRowCol[0] += self.puzzle.size
                        missingTileIdxRowCol[1] += 1
                    elif move == 2:
                        # move left
                        solutionsHistory.append([])
                        solNew = solutionsHistory[-1]

                        cell_positions = sol[0].copy()
                        cell_positions[sol[1][0]] = (
                            cell_positions[sol[1][0] - 1]
                        )
                        cell_positions[sol[1][0] - 1] = missingTileIdx

                        missingTileIdxRowCol = sol[1].copy()
                        missingTileIdxRowCol[0] -= 1
                        missingTileIdxRowCol[2] -= 1
                    elif move == -2:
                        # move right
                        solutionsHistory.append([])
                        solNew = solutionsHistory[-1]

                        cell_positions = sol[0].copy()
                        cell_positions[sol[1][0]] = (
                            cell_positions[sol[1][0] + 1]
                        )
                        cell_positions[sol[1][0] + 1] = missingTileIdx

                        missingTileIdxRowCol = sol[1].copy()
                        missingTileIdxRowCol[0] += 1
                        missingTileIdxRowCol[2] += 1
                    else:
                        raise ValueError(
                            'only values 1, -1, 2 and -2 are foreseen'
                        )

                    # move sequence or moves sequence?
                    movesSequence = sol[2].copy()
                    movesSequence.append(move)

                    solNew.append(cell_positions)
                    solNew.append(missingTileIdxRowCol)
                    solNew.append(movesSequence)

            # prune non-optimal (partial) solutions:
            # for solutions leading to equal cell/tile
            # positions, keep only the solutions with
            #
            solutionsToKeep = np.ones(
                shape=(len(solutionsHistory),),
                dtype=bool
            )
            numFullSolutions = 0
            for i in range(len(solutionsHistory)):
                if (
                    (solutionsHistory[i][0] == self.puzzle.tile_pos_initial)
                    .sum() == self.puzzle.size_sq
                ):
                    numFullSolutions += 1
                for j in range(len(solutionsHistory)):
                    if (
                        j != i and bool(solutionsToKeep[j])
                        and (solutionsHistory[i][0] == solutionsHistory[j][0])
                        .sum() == self.puzzle.size_sq  # replace with .all()?
                    ):
                        if (
                            len(solutionsHistory[i][2])
                            < len(solutionsHistory[j][2])
                        ):
                            # mark solution j for deletion
                            solutionsToKeep[j] = False
                        elif (
                            len(solutionsHistory[i][2])
                            > len(solutionsHistory[j][2])
                        ):
                            # mark solution i for deletion
                            solutionsToKeep[i] = False
                            # no need to keep comparing solution i to
                            # other solutions, since a better solution
                            # has already been found!
                            continue

            print(f'{numFullSolutions} full solution(s)')

            solutionsHistoryNew = []
            count = 0
            for i in range(len(solutionsHistory)):
                if bool(solutionsToKeep[i]):
                    solutionsHistoryNew.append(solutionsHistory[i])
                    count += 1

                    if count == 19000:
                        break
            solutionsHistory = solutionsHistoryNew
            print(
                f"kept {solutionsToKeep.sum()}/{len(solutionsToKeep)}"
                "solutions"
            )

            print(len(solutionsHistory), newSolutions)

    def change_puzzle_size(self):
        puzzle_size_new = self.spinbox_size_select.value()
        if puzzle_size_new == self.size:
            return

        answer = self.size_change_confirmation_dialog.exec()
        if answer == QMessageBox.StandardButton.No:
            self.spinbox_size_select.setValue(self.size)
        elif answer == QMessageBox.StandardButton.Yes:
            self.changing_puzzle_size = True
            # save value as attribute for method change_puzzle_size_finish
            self.puzzle_size_new = puzzle_size_new

            # if solving a puzzle, stop solving
            if self.solving:
                self.stop_solving()
            else:
                self.change_puzzle_size_finish()

    def change_puzzle_size_finish(self):
        # close figures from solving puzzles
        if self.plot_figs:
            for fig in self.plot_figs:
                plt.close(fig)
            self.plot_figs = None

        # at this point:
        # self.size - old puzzle size
        # self.puzzle_size_new - new puzzle size

        # clear puzzle solution for the old puzzle size
        self.clear_solution_puzzle_heatmap(previous_size=self.size)

        # update puzzle size variable
        self.size = self.puzzle_size_new

        # create a new Puzzle object for the new puzzle size
        # update cell dimensions for the new puzzle size
        # draw puzzle solution for the new puzzle size
        self.puzzle = Puzzle(size=self.puzzle_size_new)
        self.calc_puzzle_cell_dims()
        self.drawPuzzleSolution()

        # update the ThreadPuzzle object with the new Puzzle object
        # generate a new puzzle (and plot it)
        self.thread_puzzle.puzzle = self.puzzle
        self.thread_puzzle.new_puzzle()

        # finish changing puzzle size
        del self.puzzle_size_new
        self.changing_puzzle_size = False

        print(f'[GUI Thread] Finished changing puzzle size to {self.size}')

    def load_puzzle_image(self):
        self.img = PuzzleImage(
            self.size,
            os.path.join(
                os.path.dirname(__file__),
                'images',
                'pexels-mavihnt-36752495_reduced.jpg',
            )
            # files("sliding_tile_puzzle").joinpath(
            #     'images',
            #     'pexels-mavihnt-36752495_reduced.jpg',
            # )
        )

    def clear_solution_puzzle_heatmap(self, previous_size):
        layouts_to_clear = (
            self.layoutSolution,
            self.layoutPuzzle,
            self.layoutRGB,
            )

        for layout in layouts_to_clear:
            for row in range(previous_size):
                for col in range(previous_size):
                    layout_item = layout.itemAtPosition(row, col)
                    if layout_item and layout_item.widget():
                        layout.removeWidget(layout_item.widget())

    def calc_puzzle_cell_dims(self):
        self.cell_height = self.img.height // self.size
        self.cell_width = self.img.width // self.size

    def drawPuzzleSolution(self):
        for row in range(self.puzzle.size):
            for col in range(self.puzzle.size):
                if (
                    row != self.puzzle.missingTileRow
                    or col != self.puzzle.missingTileCol
                ):
                    pixmapCell = self.img.pixmap.copy(
                        self.img.width//self.puzzle.size*col,
                        self.img.height//self.puzzle.size*row,
                        self.cell_width,
                        self.cell_height
                    )
                else:
                    pixmapCell = QPixmap(self.cell_width, self.cell_height)
                    pixmapCell.fill(QColor('yellow'))
                label = QLabel()
                label.setPixmap(pixmapCell)
                self.layoutSolution.addWidget(label, row, col)

    def drawPuzzleRGBMap(self):
        cell_positionsRows = self.puzzle.tile_pos // self.puzzle.size
        cell_positionsCols = self.puzzle.tile_pos % self.puzzle.size
        i = -1
        for row in range(self.puzzle.size):
            for col in range(self.puzzle.size):
                i += 1
                # retrieve pixmap from the puzzle solution
                pixmapCell = self.layoutSolution.itemAtPosition(
                    cell_positionsRows[i],
                    cell_positionsCols[i]
                ).widget().pixmap()
                # if no widget at position (row, col), create widget QLabel and
                # set the pixmap, otherwise just replace the pixmap
                # XXXXX: why does creating and adding a new widget when there
                #        is already a widget have unintended effects?
                if self.layoutPuzzle.itemAtPosition(row, col) is None:
                    label = QLabel()
                    label.setPixmap(pixmapCell)
                    self.layoutPuzzle.addWidget(label, row, col)
                else:
                    self.layoutPuzzle.itemAtPosition(
                        row,
                        col).widget().setPixmap(pixmapCell)

        self.tile_posRGB = 255*np.ones(
            shape=(self.puzzle.size_sq, 3),
            dtype=np.uint8
        )
        self.updateRGBMap()

    def updateRGBMap(self, idxs=None):
        """
        updates the RGB map for the tiles specified by the mask
        """
        if idxs is None:
            idxs = np.arange(self.puzzle.size_sq)

        # remove these following two lines after implementing the mask
        # cell_positions = self.puzzle.tile_pos
        # self.puzzle.tile_pos_final = arange(self.puzzle.size_sq)
        # print(cell_positions, type(cell_positions))
        # print(self.puzzle.tile_pos_final, type(self.puzzle.tile_pos_final))

        # calculate new RGB values for the tiles specified in the mask
        tile_color = np.zeros(shape=(len(idxs), 3), dtype=np.uint8)
        # distance can go from 0 to sqrt(2)*(puzzle_size-1), and
        # 1) 0 corresponds to green (RGB: 0, 255, 0)
        # 2) 1/2*sqrt(2)*(puzzle_size-1) to yellow (RGB: 255, 255, 0)
        # 3) and sqrt(2)*(puzzle_size-1) to red (RGB: 255, 0, 0)
        tile_dist_rel_half = (
            self.puzzle.tile_dist[idxs]/(1/2*pow(2, .5)*(self.puzzle.size-1))
        )

        # calculate red value (index 0) and then green value (index 1)
        tile_color[:, 0] = (
            (np.round(np.minimum(tile_dist_rel_half, 1)*255, decimals=0))
            .astype(np.uint8)
        )
        tile_color[:, 1] = (
            255
            - np.round(
                (np.maximum(tile_dist_rel_half, 1)-1)*255,
                decimals=0
            ).astype(np.uint8)
        )

        # assign black color to tiles in the correct positions
        tile_color[
            self.puzzle.tile_pos[idxs] == self.puzzle.tile_pos_final[idxs],
            :
        ] = 0

        self.tile_posRGB[idxs] = tile_color

        # update the heat map tile colors
        rows = idxs // self.puzzle.size
        cols = idxs % self.puzzle.size
        for i in range(len(idxs)):
            pixmapCellRGB = QPixmap(self.cell_width, self.cell_height)
            r_val, g_val, b_val = self.tile_posRGB[idxs[i]]
            pixmapCellRGB.fill(QColor(r_val, g_val, b_val))

            if self.layoutRGB.itemAtPosition(rows[i], cols[i]) is None:
                labelRGB = QLabel()
                labelRGB.setPixmap(pixmapCellRGB)
                self.layoutRGB.addWidget(labelRGB, rows[i], cols[i])
            else:
                (
                    self.layoutRGB.itemAtPosition(rows[i], cols[i])
                    .widget().setPixmap(pixmapCellRGB)
                )

    def moveMissingTile(self, direction: str, puzzle_thread_signal: bool = False):
        # verify if the missing tile can be moved up/down/left/right
        # within the limits of the grid
        valid_move = False
        if direction == "u":
            if self.puzzle.missingTileScrambledRow > 0:
                valid_move = True
                missingTileFinalRow = self.puzzle.missingTileScrambledRow-1
                missingTileFinalCol = self.puzzle.missingTileScrambledCol
            else:  # == 0
                if self.checkbox_wraparound.isChecked():
                    valid_move = True
                    missingTileFinalRow = self.puzzle.size-1
                    missingTileFinalCol = self.puzzle.missingTileScrambledCol
        elif direction == "d":
            if self.puzzle.missingTileScrambledRow < self.puzzle.size-1:
                valid_move = True
                missingTileFinalRow = self.puzzle.missingTileScrambledRow+1
                missingTileFinalCol = self.puzzle.missingTileScrambledCol
            else:  # == self.puzzle.size-1
                if self.checkbox_wraparound.isChecked():
                    valid_move = True
                    missingTileFinalRow = 0
                    missingTileFinalCol = self.puzzle.missingTileScrambledCol
        elif direction == "l":
            if self.puzzle.missingTileScrambledCol > 0:
                valid_move = True
                missingTileFinalRow = self.puzzle.missingTileScrambledRow
                missingTileFinalCol = self.puzzle.missingTileScrambledCol-1
            else:  # == 0
                if self.checkbox_wraparound.isChecked():
                    valid_move = True
                    missingTileFinalRow = self.puzzle.missingTileScrambledRow
                    missingTileFinalCol = self.puzzle.size-1
        elif direction == "r":
            if self.puzzle.missingTileScrambledCol < self.puzzle.size-1:
                valid_move = True
                missingTileFinalRow = self.puzzle.missingTileScrambledRow
                missingTileFinalCol = self.puzzle.missingTileScrambledCol+1
            else:  # == self.puzzle.size-1
                if self.checkbox_wraparound.isChecked():
                    valid_move = True
                    missingTileFinalRow = self.puzzle.missingTileScrambledRow
                    missingTileFinalCol = 0

        if not valid_move:
            # play alert sound
            print('\a', end='')
            return -1
        else:
            pixmapMissingTile = self.layoutPuzzle.itemAtPosition(
                self.puzzle.missingTileScrambledRow,
                self.puzzle.missingTileScrambledCol
            ).widget().pixmap()

            pixmapTileToSwitch = self.layoutPuzzle.itemAtPosition(
                missingTileFinalRow,
                missingTileFinalCol
            ).widget().pixmap()

            widgetMissingTile = self.layoutPuzzle.itemAtPosition(
                missingTileFinalRow,
                missingTileFinalCol
            ).widget()
            widgetMissingTile.setPixmap(pixmapMissingTile)

            widgetTileToSwitch = self.layoutPuzzle.itemAtPosition(
                self.puzzle.missingTileScrambledRow,
                self.puzzle.missingTileScrambledCol
            ).widget()
            widgetTileToSwitch.setPixmap(pixmapTileToSwitch)

            widgetMissingTile.repaint()
            widgetTileToSwitch.repaint()

            switchedTilesRowCol = []
            switchedTilesRowCol.append(
                (
                    self.puzzle.missingTileScrambledRow,
                    self.puzzle.missingTileScrambledCol
                )
            )
            self.puzzle.missingTileScrambledRow = missingTileFinalRow
            self.puzzle.missingTileScrambledCol = missingTileFinalCol
            self.puzzle.missingTileScrambledIdx = (
                self.puzzle.missingTileScrambledRow*self.puzzle.size
                + self.puzzle.missingTileScrambledCol
            )
            switchedTilesRowCol.append(
                (
                    self.puzzle.missingTileScrambledRow,
                    self.puzzle.missingTileScrambledCol,
                )
            )

            positionTileToSwitch = (
                switchedTilesRowCol[0][0]*self.puzzle.size
                + switchedTilesRowCol[0][1]
            )
            positionMissingTile = (
                switchedTilesRowCol[1][0]*self.puzzle.size
                + switchedTilesRowCol[1][1]
            )

            switch_pos = np.array(
                [positionTileToSwitch, positionMissingTile],
                dtype=self.puzzle.tile_pos.dtype
            )

            # XXXXX: should by done by ThreadPuzzle thread,
            # not the GUI thread
            self.puzzle.switch_tiles(switch_pos)

            # update RGB map for the two tiles switched
            self.updateRGBMap(switch_pos)

            if direction == "u":
                self.move_history[-1].append(1)
            elif direction == "d":
                self.move_history[-1].append(-1)
            elif direction == "l":
                self.move_history[-1].append(2)
            elif direction == "r":
                self.move_history[-1].append(-2)
            self.update_move_count()

            self.distancesHistory[-1].append(self.puzzle.tile_dist.copy())
            self.distancesSumHistory[-1].append(self.puzzle.tile_dist.sum())
            self.completionHistory[-1].append(
                (self.puzzle.tile_pos == arange(self.puzzle.size_sq)).sum()
                / self.puzzle.size_sq
            )

            time.sleep(.01)

        if puzzle_thread_signal:
            self.thread_puzzle.move_performed()

        return 0

    def moveMissingTileUp(self):
        return self.moveMissingTile("u")

    def moveMissingTileDown(self):
        return self.moveMissingTile("d")

    def moveMissingTileLeft(self):
        return self.moveMissingTile("l")

    def moveMissingTileRight(self):
        return self.moveMissingTile("r")

    def plot_solved(self):
        # TODO: (check/review)
        #       only plot if it hasn't been solved before?
        #       also check vs reset/new functionality

        if self.plot_figs is None:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()

            plt.show()

            self.plot_figs = (fig1, fig2)
            self.plot_axs = (ax1, ax2)

        x = list(range(len(self.move_history[-1])))
        y = self.distancesSumHistory[-1]
        z = self.completionHistory[-1]
        self.plot_axs[0].plot(x, y, 'o-', linewidth=2)
        self.plot_axs[1].plot(x, z, 'o-', linewidth=2)

        for fig in self.plot_figs:
            fig.canvas.draw_idle()

    def resetMovesHistory(self):
        # XXXXX: needs to be reviewed
        # reset and new buttons should have different move history resets
        # (clear vs addition)
        if self.move_history is None:
            self.move_history = [[]]
            self.distancesHistory = [[]]
            self.distancesSumHistory = [[]]
            self.completionHistory = [[]]
        else:
            self.move_history.append([])
            self.distancesHistory.append([])
            self.distancesSumHistory.append([])
            self.completionHistory.append([])

        self.update_move_count()

    def update_move_count(self):
        self.label_move_count.setText(str(len(self.move_history[-1])))

    def keyReleaseEvent(self, event):
        if isinstance(event, QKeyEvent) and self.keys_enabled:
            keyInt = event.key()

            if keyInt == Qt.Key.Key_Left:
                self.moveMissingTileLeft()
            elif keyInt == Qt.Key.Key_Up:
                self.moveMissingTileUp()
            elif keyInt == Qt.Key.Key_Right:
                self.moveMissingTileRight()
            elif keyInt == Qt.Key.Key_Down:
                self.moveMissingTileDown()


class SlidingTilePuzzleApp(QApplication):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # reference to SlidingTilePuzzleGUI/QMainWindow needs to be kept, otherwise window
        # object is destroyed shortly after creation due to garbage collection
        self.window = SlidingTilePuzzleGUI()
