import random
import itertools
import time

from _3d_utils import *


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

        except RuntimeError:
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


class PuzzleCube3D(PuzzleCube):
    def __init__(self,
                 state: list[list[list[int]]] = None,
                 size: int = 3,
                 numPieces: int = 5,
                 pieceMinCubes: int = 4,
                 pieceMaxCubes: int = 6):
        """
        PuzzleCube3D class constructor.

        :param state: the state of the puzzle cube
        :param size: the size of the cube
        :param numPieces: number of pieces to make the cube with
        :param pieceMinCubes: minimum number of cubes per piece
        :param pieceMaxCubes: maximum number of cubes per piece
        """
        PuzzleCube.__init__(self, state, size, numPieces, pieceMinCubes, pieceMaxCubes)

        self.cubes = []
        self.pieces = []
        self.make_3d()

        self.offsets = []
        self.exploded = False
        self.explode()

        self.camera = Camera3D(pos=(0, -25, -1000), theta=(0, 0, 0), FoV=1000)

        self.rotMat = np.identity(3, dtype=float)
        self.rotVel = np.asarray([0, 0, 0], dtype=np.float32)

    def make_3d(self):
        scale = 40
        vertices = np.array([(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
                             (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)])

        faces = np.array([[0, 1, 2, 3],
                          [4, 5, 6, 7],
                          [0, 1, 5, 4],
                          [1, 2, 6, 5],
                          [2, 3, 7, 6],
                          [3, 0, 4, 7]], dtype=int)

        colors = [(255, 0, 0), (255, 128, 0), (0, 255, 0), (0, 0, 255),
                  (128, 0, 255), (255, 255, 0), (0, 0, 0), (128, 64, 0),
                  (128, 128, 128), (0, 255, 255), (255, 0, 128), (0, 128, 255)]

        self.cubes = []
        self.pieces = []
        startPos = -0.5*(self.size-1)
        coordRange = range(self.size)

        for x, y, z in itertools.product(coordRange, coordRange, coordRange):
            cube = self.state[x][y][z]
            color = colors[cube.piece]
            cubeVertices = (vertices + 2 * (np.asarray([x, y, z]) + startPos)) * scale
            self.cubes.append(Solid3D(cubeVertices, faces, color))
            self.pieces.append(cube.piece)

    def regenerate(self, state=None):
        if state is None:
            self.generate_state()
        elif isinstance(state, str):
            self.state = self.from_str(state)
        else:
            self.state = state

        self.make_3d()
        self.explode()

    def explode(self):
        avgSize = self.size ** 2 / self.numPieces
        self.offsets = [Point3D(0, 0, 0) for _ in range(self.numPieces)]

        for i, piece in enumerate(self.pieces):
            self.offsets[piece] += self.cubes[i].center

        for piece in range(self.numPieces):
            self.offsets[piece] /= avgSize

    def render(self):
        pygame.init()

        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('3D Puzzle Cube')

        clock = pygame.time.Clock()

        running = True

        buttons = [False for _ in range(self.numPieces)]
        visible = [True for _ in range(self.numPieces)]
        visibleKeys = [pygame.K_1, pygame.K_2, pygame.K_3,
                       pygame.K_4, pygame.K_5, pygame.K_6,
                       pygame.K_7, pygame.K_8, pygame.K_9,
                       pygame.K_0][:self.numPieces]

        explodeTime = time.time()
        explodeFrame = 0

        clickX = 0
        clickY = 0
        clicked = False

        while running:
            # reduce rotation velocity
            self.rotVel *= 0 if clicked else 0.999
            keys = pygame.key.get_pressed()
            self.rotVel[0] += 0.01 * (keys[pygame.K_UP] - keys[pygame.K_DOWN])
            self.rotVel[1] += 0.01 * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            self.rotVel[2] += 0.01 * (keys[pygame.K_RSHIFT] - keys[pygame.K_SLASH])
            self.rotVel = np.clip(self.rotVel, -0.07, 0.07)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    clickX, clickY = event.pos
                    clicked = True
                if event.type == pygame.MOUSEBUTTONUP:
                    clicked = False
                    self.rotVel = np.asarray([0, 0, 0], dtype=float)
                if event.type == pygame.MOUSEMOTION:
                    if clicked:
                        x, y = event.pos
                        self.rotVel = 0.008*np.asarray([clickY - y, x - clickX, 0], dtype=float)
                        clickX = x
                        clickY = y
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.regenerate()
                    if event.key == pygame.K_p:
                        print(self)
                    if event.key == pygame.K_l:
                        state = ''
                        print('New state:')
                        for i in range(self.size * self.size + 2):
                            state += input('') + '\n'
                        self.regenerate(state[:-1])
                    if event.key == pygame.K_SPACE or event.key == pygame.K_x:
                        explodeTime = time.time()
                        self.exploded = not self.exploded

            screen.fill((255, 255, 255))

            self.camera.forward_step((20 * (keys[pygame.K_a] - keys[pygame.K_d]),
                                      20 * (keys[pygame.K_q] - keys[pygame.K_e]),
                                      20 * (keys[pygame.K_w] - keys[pygame.K_s])))

            self.rotMat = Point3D(0, 0, 0).rotation_matrix((1 - explodeFrame * 0.2) * self.rotVel).dot(self.rotMat)

            for i, key in enumerate(visibleKeys):
                keydown = keys[key]
                # key down now but not before = just pressed

                if keydown and not buttons[i]:
                    visible[i] = not visible[i]

                buttons[i] = keydown

            explodeFrame += min(time.time() - explodeTime, 1) * (self.exploded - explodeFrame)

            rotated = sorted([cube.get_shifted(self.offsets[self.pieces[i]] * 0.875 * explodeFrame)
                             .get_matrix_rotated(self.rotMat, Point3D(0, 0, 0)) for i, cube in enumerate(self.cubes)
                              if visible[self.pieces[i]]],
                             key=lambda n: n.min_z() + n.center.dist_to(self.camera),
                             reverse=True)

            for cube in rotated:
                if cube.center.z > self.camera.z:
                    cube.draw(screen,
                              self.camera.get_shifted(np.asarray((0, -50, -750)) * explodeFrame),
                              shadow=True,
                              shadowOffset=(0, -cube.center.y - (85+70*explodeFrame) * self.size, 0),
                              cor=Point3D(0, 0, 0)
                              )

            pygame.display.flip()
            clock.tick(120)

        pygame.quit()


if __name__ == '__main__':
    print("\033[1;38;2;0;192;255mHOW TO USE:\n"
          "1. Camera Movement:\033[0m\n"
          "\t- use WASD to move forward, left, backward, and right\n"
          "\t- use Q and E to move up and down\n\n"
          "2. \033[1;38;2;0;192;255mPuzzle Cube Appearance:\033[0m\n"
          "\t- use arrow keys, forward slash (/) and RSHIFT or drag with the mouse to rotate the cube\n"
          "\t- use 1-5 to toggle the visibility of each piece\n"
          "\t- use 'SPACE' or 'X' to toggle exploded view\n\n"
          "3. \033[1;38;2;0;192;255mPuzzle Cube Generation:\033[0m\n"
          "\t- use 'R' to regenerate the puzzle cube\n"
          "\t- use 'P' to print out the cube's layers in the console\n"
          "\t\t- this can be used to load the cube in the future\n"
          "\t- use 'L' to load a previously generated cube\n"
          "\t\t- copy and paste the cube's layers from when you pressed 'P'")
    puzzle = PuzzleCube3D(None)
    puzzle.render()
