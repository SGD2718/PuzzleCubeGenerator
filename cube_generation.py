from cube import *
import numpy as np
import random


class PuzzleCube:
    """
    Puzzle Cube class
    """

    def __init__(self,
                 state: list[list[list[int]]] = None,
                 size: int = 3,
                 numPieces: int = 5,
                 pieceMinCubes: int = 4,
                 pieceMaxCubes: int = 6
                 ):
        """
        PuzzleCube class constructor.

        :param state: the state of the puzzle cube
        :param size: the size of the cube
        :param numPieces: number of pieces to make the cube with
        :param pieceMinCubes: minimum number of cubes per piece
        :param pieceMaxCubes: maximum number of cubes per piece
        """

        self.size = size
        self.minCubes = pieceMinCubes
        self.maxCubes = pieceMaxCubes
        self.numPieces = numPieces

        if state is None:
            self.generate_state()
        elif isinstance(state, str):
            self.from_str(state)
        else:
            self.state = state

        self.pieces = self.find_pieces()

    def __repr__(self):
        puzzleCube = ''
        for layer in self.state:
            for row in layer:
                puzzleCube += ' '.join(map(str, row)) + '\n'
            puzzleCube += '\n'

        return puzzleCube[:-2]

    def from_str(self, string):
        """
        thing
        :param string: string representation of the cube
        :return:
        """
        for i, color in enumerate(list('🟥🟧🟩🟦🟪🟨⬛️🟫⬜️')):
            string = string.replace(color, str(i))

        try:
            state = list(map(lambda layer:
                             list(map(lambda row:
                                      list(map(int, row.split(' '))),
                                      layer.split('\n'))),
                             string.split('\n\n')))
            self.state = state

            for x, y, z in itertools.product(range(self.size), range(self.size), range(self.size)):
                self.state[x][y][z] = Cube(self.state[x][y][z], (x, y, z), self.numPieces, self.state)

        except:
            print("INVALID INPUT.\nPlease use the format outputted after pressing 'P'.")

        return self.state

    def distribute_cubes(self) -> list:
        """
        Distributes the number of cubes that goes to each piece
        :return: a list of the number of cubes each piece should contain.
        """
        numCubes = self.size ** 3

        # make sure that the puzzle can be made with the cube limits
        assert (self.minCubes <= numCubes / self.numPieces <= self.maxCubes)

        # generate minimum distribution

        '''
        (volume - max * numMax) / min - 1 <= numMin <= (volume - max*(1 + numMax)) / min
        numMin <= (volume - max * numPieces) / (min - max)
        numMax = numPieces - numMin - 1

        volume = size^3
        filler: volume - min * numMin - max * numMax 
        '''

        numMin = min((numCubes - self.maxCubes * self.numPieces) // (self.minCubes - self.maxCubes), self.numPieces - 1)
        numMax = max(self.numPieces - numMin - 1, 0)
        filler = numCubes - numMin * self.minCubes - numMax * self.maxCubes
        distribution = [self.maxCubes for _ in range(numMax)] + [filler] + [self.minCubes for _ in range(numMin)]

        # ensure that all the values are in range
        assert (sum(distribution) == numCubes)
        assert (min(distribution) >= self.minCubes)
        assert (max(distribution) <= self.maxCubes)

        # generate all remaining possible distributions
        distributions = [distribution]

        if distribution == [distribution[0]] * self.numPieces:
            return distributions

        def generate(distr):
            distr = distr[:]
            # find smaller parts of exchange
            indexMinRight = distr.index(min(distr))
            indexMinLeft = indexMinRight + 0
            tmp = indexMinRight + 0
            while distr[0] > distr[indexMinLeft - 1] + 1:
                indexMinLeft -= 1
                # remember the first time this minimum occurred
                if distr[tmp] != distr[indexMinLeft]:
                    tmp = indexMinLeft

            # use the rightmost occurrence of this minimum
            indexMinLeft = tmp
            del tmp

            # find larger part of exchange
            indexMax = self.numPieces - distr[::-1].index(max(distr)) - 1

            # use this thing
            if distr[indexMax] > distr[indexMinLeft] + 1:
                distr1 = distr[:]
                distr1[indexMax] -= 1
                distr1[indexMinLeft] += 1
                if distr1 not in distributions:
                    distributions.append(distr1)
                    generate(distr1)

            if distr[indexMax] > distr[indexMinRight] + 1:
                distr[indexMax] -= 1
                distr[indexMinRight] += 1
                if distr not in distributions:
                    distributions.append(distr)
                    generate(distr)

        generate(distribution)
        return random.choice(distributions)

    def find_pieces(self) -> list[list[tuple[int, int, int]]]:
        """
        Gets the positions of each cube in each piece.
        The last element is the list of available spots.

        :return: a list of pieces, each represented as a list of positions corresponding to each cube in the piece.
        """

        pieces = [[] for _ in range(self.numPieces + 1)]

        for i, j, k in itertools.product(range(self.size), range(self.size), range(self.size)):
            pieces[int(self.state[i][j][k])].append((i, j, k))

        return pieces

    def append_cube(self, piece: int, free: list = None) -> list:
        """
        Appends a cube to a piece of the puzzle cube in a random available spot and updates the list of available spots.

        :param piece: the piece to which the cube will belong
        :param free: the list of free spaces for the piece. If free == None, then any unoccupied spot can be chosen.
        :return: the updated list of free spaces after placing the cube.
        """

        # check if the cube is the first in the piece
        isRoot = False
        if free is None:
            free = self.pieces[-1]
            isRoot = True

        if not free:
            return None

        # pick a random free space
        index = random.randrange(0, len(free))

        # remove spot from lists of free spots
        pos = tuple(free.pop(index))

        if not isRoot:
            self.pieces[-1].remove(pos)

        # add the cube to the piece
        self.pieces[piece].append(pos)

        # update the puzzle cube state
        cube = self.state[pos[0]][pos[1]][pos[2]] = Cube(piece, pos, self.numPieces, self.state)

        # update list of free spots
        if isRoot:
            return cube.get_adjacent_positions()[-1]

        free += cube.get_adjacent_positions()[-1]

        return list(set(free))

    def is_congruent(self, piece1, piece2):
        if isinstance(piece1, int):
            piece1 = self.pieces[piece1]
        if isinstance(piece2, int):
            piece2 = self.pieces[piece2]

        # make sure they have the same number of pieces
        if len(piece1) != len(piece2):
            return False

        # convert to numpy array for more operations
        piece1 = np.asarray(piece1, dtype=float)
        piece2 = np.asarray(piece2, dtype=float)

        # calculate the sizes of the pieces
        size1 = np.max(piece1, axis=0) - np.min(piece1, axis=0)
        size2 = np.max(piece2, axis=0) - np.min(piece2, axis=0)

        # make sure that the sizes are the same
        if sorted(size1) != sorted(size2):
            return False

        # translate the pieces so that their centroids are at the origin

        numCubes = len(piece1)

        def squared_norm(piece, i, j):
            delta = piece[i] - piece[j]
            return delta.dot(delta)

        norms1 = []
        norms2 = []

        for i in range(numCubes-1):
            for j in range(i, numCubes):
                norms1.append(squared_norm(piece1, i, j))
                norms2.append(squared_norm(piece2, i, j))

        return sorted(norms1) == sorted(norms2)

    def all_unique(self):
        for i in range(self.numPieces-1):
            for j in range(i+1, self.numPieces):
                if self.is_congruent(i, j):
                    return False
        return True

    def is_interlocked(self) -> bool:
        """
        Checks that at least two pieces are interlocked together
        :return: True if interlock was found, False otherwise.
        """
        '''
        Interlocking: when a piece contacts another
        piece on at least one set of parallel sides.
        '''
        singleInterlock = False

        for i, piece in enumerate(self.pieces[:-1]):
            # checklist of contacts for each axis.
            axisContacts = np.full((self.numPieces, 6), False, dtype=bool)
            multiInterlock = False

            # loop through each cube in the piece
            for pos in piece:
                # each piece must be interlocked with something so that it's not just sitting there.
                cube = self.state[pos[0]][pos[1]][pos[2]]
                pieceAxes = cube.get_adjacent_axes(self.numPieces, self.state)

                # loop through the pieces that the cube contacts
                for j, axes in enumerate(pieceAxes):
                    if i != j:
                        # loop through the contacts
                        for axis in axes:
                            # update number of contacts to the piece on the axis
                            axisContacts[j][axis] = True
                            axis2 = axis + 3 if axis < 3 else axis - 3

                            # check if we found a sandwich
                            if axisContacts[j][axis2]:
                                singleInterlock = True
                                multiInterlock = True
                            elif (not multiInterlock) and [x.any() for x in list(axisContacts.T)][axis2]:
                                multiInterlock = True

                            # check if we're done with this piece
                            if singleInterlock and multiInterlock:
                                break

                        # avoid unnecessary checks
                        else:
                            continue
                        break
                else:
                    continue
                break
            else:
                # make sure that the piece interlocks with something
                if not multiInterlock:
                    # a piece that falls off is unacceptable
                    return False

        return singleInterlock

    def generate_state(self):
        # determine how many cubes each piece will have
        print('Generating...')
        cubeCounts = self.distribute_cubes()

        while True:
            # reset the puzzle cube
            self.state = np.full((self.size, self.size, self.size), -1, dtype=int).tolist()
            self.pieces = self.find_pieces()

            for piece in range(self.numPieces):
                # make the piece by adding cubes
                free = None
                for _ in range(cubeCounts[piece]):
                    free = self.append_cube(piece, free)
                    # stop if we ran out of free spaces and aren't done
                    if free is None:
                        break

                # break if we must restart
                else:
                    continue
                break
            else:
                # if the cube is complete, then check interlocking and uniqueness
                if self.is_interlocked() and self.all_unique():
                    return self.state
