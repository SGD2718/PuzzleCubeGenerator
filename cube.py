import itertools
import numpy as np


class Cube:
    def __init__(self, piece, position, numPieces, state=np.full((3, 3, 3), -1, dtype=int).tolist()):
        """
        :param piece: the piece to which the cube belongs
        :param position: the cube's position on the puzzle cube
        :param numPieces: number of pieces the cube is made of
        :param state: the puzzle cube's state
        """
        self.position = tuple(position)
        self.piece = piece
        self.adjacent = self.find_adjacent(numPieces, state)

    def __int__(self):
        return self.piece

    def __repr__(self):
        return '🟥🟧🟩🟦🟪🟨⬛️🟫⬜️'[self.piece]

    def __str__(self):
        return '🟥🟧🟩🟦🟪🟨⬛️🟫⬜️'[self.piece]

    def find_adjacent(self, numPieces, state: list[list[list[int]]]):
        """
        Gets the positions and pieces of the adjacent cubes to the cube.
        index -1 is list of cubes that are not part of a piece yet.

        :param numPieces: number of pieces the cube is made of
        :param state: the state of the puzzle cube
        :return: A list of adjacent cubes, where the index of each element in the list represents the piece to which it
        belongs. Pieces are represented as dictionaries containing the list of positions of each cube and the axes on
        which they were offset.
        """

        adjacent = [{'positions': [], 'axes': []} for _ in range(numPieces+1)]
        pos = list(self.position)

        # x, y, and z axes = (0, 1, 2), respectively
        for axis, offset in itertools.product(range(3), (-1, 1)):
            pos[axis] += offset  # offset position

            # ensure that the offset position is valid
            if 0 <= pos[axis] < 3:
                # get the piece of which the cube in question is a part
                piece = int(state[pos[0]][pos[1]][pos[2]])

                # update adjacent
                adjacent[piece]['positions'].append(tuple(pos))
                adjacent[piece]['axes'].append(axis + (3 if offset == 1 else 0))

            pos[axis] -= offset  # undo offset

        return adjacent

    def get_adjacent_positions(self, numPieces=None, state: list[list[list[int]]] = None):
        """
        :param numPieces: number of pieces the cube is made of
        :param state: the state of the puzzle cube. Leave as None if the state did not change.
        :return: a list of pieces, where each piece is represented by a list of the positions of its cubes.
        """
        # update adjacent, if necessary
        if state is not None:
            self.adjacent = self.find_adjacent(numPieces, state)

        # get the positions of the adjacent pieces and return.
        return [piece['positions'] for piece in self.adjacent]

    def get_adjacent_axes(self, numPieces=None, state: list[list[list[int]]] = None):
        """
        :param numPieces: number of pieces the cube is made of
        :param state: the state of the puzzle cube. Leave as None if the state did not change.
        :return: a list of pieces, where each piece is represented by a list of the axes on which its cubes are offset
        from this cube.
        """
        # update adjacent, if necessary
        if state is not None:
            self.adjacent = self.find_adjacent(numPieces, state)

        # get the positions of the adjacent pieces and return.
        return [piece['axes'] for piece in self.adjacent]

    def get_connections(self) -> list:
        """
        Gets the connections to adjacent cubes that are part of the same piece.
        :return: a list of the adjacent cubes' coordinates
        """
        return self.adjacent[self.piece]