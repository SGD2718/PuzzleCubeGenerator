from cube_generation import *
from _3d_utils import *
from pygame import *
import time
import math


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
        self.projection = 'perspective'

        self.outlined = True

        self.camera = Camera3D(pos=(0, -25, -1000), theta=(0, 0, 0), FoV=1000)

        self.rotMat = Point3D.rotation_matrix(None, (np.pi*0.25, 0, np.pi*0.25))
        self.rotVel = np.asarray([0, 0, 0], dtype=np.float32)
        self.explode()

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

        colors = [(255, 0, 0), (255, 128, 0), (0, 255, 0), (0, 128, 255),
                  (128, 0, 200), (255, 255, 0), (0, 0, 0), (128, 64, 0),
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

        if self.projection == 'perspective':
            self.explode()
        else:
            self.orthographic()

    def explode(self):
        avgSize = self.numPieces / self.size ** 2
        self.offsets = [Point3D(0, 0, 0) for _ in range(self.numPieces)]

        for i, piece in enumerate(self.pieces):
            self.offsets[piece] += self.cubes[i].center * avgSize

    def orthographic(self):
        searchFactor = 40
        self.rotMat = Point3D.rotation_matrix(None, (math.pi * 0.25, 0, math.pi * 0.25))
        self.rotVel = np.asarray([0.0, 0.0, 0.0])
        offsets = [Point3D(self.rotMat.dot((0, math.cos(θ), math.sin(θ)))) * searchFactor
                   for θ in np.arange(0, math.tau, math.tau / (self.numPieces-1))+math.pi/3-math.pi/4] + \
                  [Point3D(0, 0, 0)]

        #self.rotMat = self.rotMat.dot(Point3D.rotation_matrix(None, (.0*math.pi, 0*math.pi, 1*math.pi)))

        centers = [Point3D(0, 0, 0) for _ in range(self.numPieces)]

        for i, piece in enumerate(self.pieces):
            centers[piece] = self.cubes[i].center / self.pieces.count(self.pieces[i])

        self.offsets = []
        for center in centers:
            nearest = min(offsets, key=lambda pt: pt.dist_to(center))
            offsets.remove(nearest)
            self.offsets.append(nearest * 400 / searchFactor + center)

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

        def new_offsets():
            newOffsets = list(self.offsets)
            sortedOffsets = [newOffsets.pop()]

            for _ in range(len(newOffsets)):
                nearest = 0
                minDist = float('inf')
                target = sortedOffsets[-1]
                for i, offset in enumerate(newOffsets):
                    distance = target.dist_to(offset)
                    if distance < minDist:
                        minDist = distance
                        nearest = i
                sortedOffsets.append(newOffsets.pop(nearest))

            return Solid3D(sortedOffsets, np.asarray([range(len(sortedOffsets))]), color=None)

        offsets = new_offsets()

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
                        print(self.rotMat / math.pi)
                    if event.key == pygame.K_l:
                        state = ''
                        print('New state:')
                        for i in range(self.size * self.size + 2):
                            state += input('') + '\n'
                        self.regenerate(state[:-1])
                    if event.key == pygame.K_SPACE or event.key == pygame.K_x:
                        explodeTime = time.time()
                        self.exploded = not self.exploded
                    if event.key == pygame.K_o:
                        self.outlined = not self.outlined
                    if event.key == pygame.K_m:
                        if self.projection == 'perspective':
                            self.projection = 'orthographic'
                            self.orthographic()
                        else:
                            self.projection = 'perspective'
                            self.explode()

                        offsets = new_offsets()

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
                              shadow=self.projection == 'perspective',
                              shadowOffset=(0, -cube.center.y - (85+70*explodeFrame) * self.size, 0),
                              outline=2 * self.outlined,
                              cor=Point3D(0, 0, 0),
                              perspective=self.projection == 'perspective')

            '''offsets.get_matrix_rotated(self.rotMat).draw(
                screen,
                self.camera.get_shifted(np.asarray((0, -50, -750)) * explodeFrame),
                outline=5,
                cor=Point3D(0, 0, 0),
                perspective=self.projection == 'perspective')'''

            pygame.display.flip()
            clock.tick(120)

        pygame.quit()
