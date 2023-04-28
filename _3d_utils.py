import math
import numpy as np
import pygame

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 750


class Point3D:
    def __init__(self, *args):
        if isinstance(args[0], Point3D):
            self.point = args[0].point
        elif len(args) == 3:
            self.point = np.array(args, dtype=np.float32)
        else:
            self.point = np.array(args[0], dtype=np.float32)

        self.x = self.point[0]
        self.y = self.point[1]
        self.z = self.point[2]

    def __tuple__(self):
        return tuple(self.point)

    def __list__(self):
        return list(self.point)

    def __sub__(self, point2):
        return Point3D(self.point - Point3D(point2).point)

    def __add__(self, point2):
        return Point3D(self.point + Point3D(point2).point)

    def __mul__(self, scalar):
        return Point3D(self.point * scalar)

    def __truediv__(self, scalar):
        return Point3D(self.point / scalar)

    def __div__(self, scalar):
        return Point3D(self.point // scalar)

    def get_rotated(self, theta: tuple, cor=(0, 0, 0)):
        cor = Point3D(cor)
        # offset, rotate, then un-offset point
        return Point3D(np.dot(self.rotation_matrix(theta), self.point - cor.point)) + cor.point

    def rotate(self, theta, cor):
        p = self.get_rotated(theta, cor)
        self.set(p.point)
        return p

    def rotation_matrix(self, theta):
        """Point3D.rotation_matrix(theta) -> np.array(3,3)
        creates a rotation matrix."""

        # helper method to make a Rodrigues rotation matrix
        def axis_matrix(u, phi):
            W = np.array([[0, -u[2], u[1]],
                          [u[2], 0, -u[0]],
                          [-u[1], u[0], 0]])
            return np.eye(3, dtype=np.float32) + math.sin(phi) * W + (1 - math.cos(phi)) * (W @ W)

        # calculate sin and cos of thetaX
        cosY = math.cos(theta[1])
        sinY = math.sin(theta[1])

        # make the rotation matrix
        R = np.array(((cosY, 0, sinY), (0, 1, 0), (-sinY, 0, cosY)))  # y-axis
        R = axis_matrix(np.dot(R, np.array((1, 0, 0))), theta[0]) @ R  # x-axis
        return axis_matrix(np.dot(R, np.array((0, 0, 1))), theta[2]) @ R  # z-axis

    def __eq__(self, point2):
        if not isinstance(point2, Point3D):
            try:
                return np.array_equal(np.array(point2, dtype=np.float32) == self.point)
            except:
                return False
        else:
            return np.array_equal(self.point, point2.point)

    def project(self, camera={'pos': (0, 0, -1000), 'theta': (0, 0, 0)}, perspective=True):
        cz = camera.z
        camera = Camera3D(camera=camera)
        p = self - camera
        if not np.array_equal(camera.theta, np.zeros(3)):
            p.rotate(-camera.theta, camera.pos)

        if perspective:
            def proj_coord(c):
                return -camera.FoV * c / p.z
        else:
            def proj_coord(c):
                return 1000 *c / (cz + 1)

        x2d = proj_coord(p.x)
        y2d = proj_coord(p.y)
        return x2d + SCREEN_WIDTH * 0.5, y2d + SCREEN_HEIGHT * 0.5

    def dist_to(self, point2):
        point2 = Point3D(point2)
        p = self - point2
        return math.sqrt(p.point.dot(p.point))

    def __repr__(self):
        return str(self.point)

    def set_x(self, x):
        self.point[0] = self.x = x

    def set_y(self, y):
        self.point[1] = self.y = y

    def set_z(self, z):
        self.point[2] = self.z = z

    def set(self, point):
        self.point = np.array(point)
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

    def shift_x(self, dx):
        self.x += dx
        self.point[0] += dx

    def shift_y(self, dy):
        self.y += dy
        self.point[1] += dy

    def shift_z(self, dz):
        self.z += dz
        self.point[2] += dz

    def shift(self, delta):
        self.point = np.add(self.point, np.array(delta))
        self.x += delta[0]
        self.y += delta[1]
        self.z += delta[2]

    def get_shifted(self, delta):
        return Point3D(np.add(self.point, np.array(delta)))


class Camera3D(Point3D):
    def __init__(self, **kwargs):
        """
        Camera(pos : (float, float, float), theta : (float, float, float), FoV = 1000)
        Camera(x,y,z, thetaX,thetaY,thetaZ, FoV = 1000)
        Camera(camera)
        """

        if 'camera' in kwargs:
            camera = kwargs['camera']
            self.FoV = camera.FoV
            self.x = camera.x
            self.y = camera.y
            self.z = camera.z
            self.point = camera.point
            self.theta = camera.theta
        else:
            self.FoV = kwargs['FoV'] if 'FoV' in kwargs else 1000
            if 'pos' in kwargs:
                super().__init__(kwargs['pos'])
            else:
                super().__init__(kwargs['x'], kwargs['y'], kwargs['z'])
            if 'theta' in kwargs:
                self.theta = np.array(kwargs['theta'], dtype=np.float32)
            else:
                self.theta = np.array([kwargs['thetaX'],
                                       kwargs['thetaY'],
                                       kwargs['thetaZ']],
                                      dtype=np.float32)

    def rotate(self, theta, cor=None):
        """Camera3D.rotate(theta,cor=None)
        if cor is None, the center of rotation is
        the camera's position, and it doesn't move"""
        self.theta = np.add(self.theta, np.array(theta))
        if cor is not None:
            Point3D.rotate(self, theta, cor)

    def get_rotated(self, theta, cor=None):
        raise NotImplementedError('cannot preview camera rotation')

    def rotation_matrix(self, theta):
        raise NotImplementedError('method not supported for camera')

    def project(self, point):
        raise NotImplementedError('cannot project a camera')

    def __eq__(self, camera2):
        return np.array_equal(self.point, camera2.point) and np.array_equal(self.theta, camera2.theta)

    def set(self, pos, theta):
        Point3D.set(self, pos)
        self.theta = np.array(theta)

    def set_pos(self, pos):
        Point3D.set(self, pos)

    def set_theta(self, theta):
        self.theta = np.array(theta)

    def forward_step(self, delta):
        """makes camera take a step."""
        # rotate about y-axis
        sinY = math.sin(self.theta[1])
        cosY = math.cos(self.theta[1])

        R = np.array([[cosY, 0, sinY],
                      [0, 1, 0],
                      [-sinY, 0, cosY]])

        delta = np.dot(R, np.array(delta))
        self.shift(delta)

    def step(self, delta):
        """makes camera take a step."""
        # use unit vector of unaimed camera
        R = Point3D.rotation_matrix(self, self.theta)
        delta = np.dot(R, np.array(delta))
        self.shift(delta)

    def get_shifted(self, delta):
        return Camera3D(pos=(Point3D(delta).point + self.point), theta=self.theta, FoV=self.FoV)


class Solid3D:
    def __init__(self, vertices, faces, color, center=None, renderDist=None):
        self.vertices = np.array(list(map(Point3D, vertices)))
        self.faces = list(map(list, faces))
        self.color = np.array(color) if color is not None else None
        self.facesRendered = len(self.faces) // 2

        if center is None:
            self.center = self.median()
        else:
            self.center = Point3D(center)

        if renderDist is None:
            self._renderDist = np.max([self.center.dist_to(p) for p in self.vertices])
        else:
            self._renderDist = renderDist

    def min_z(self):
        return min(map(lambda pos: pos.z, self.vertices))

    def copy(self):
        return Solid3D(self.vertices, self.faces, self.color, self.center, self._renderDist)

    def get_rotated(self, theta, cor=None):
        """Solid3D.get_rotated(theta, cor=self.center) -> Solid3D
            theta: x, y, and z rotations
            cor: center of rotation
        copies the solid then rotates it"""
        if cor is None:
            cor = self.center
        else:
            cor = Point3D(cor)
        R = self.center.rotation_matrix(theta)
        return Solid3D([np.dot(R, (p - cor).point) + cor.point for p in self.vertices],
                       self.faces,
                       self.color,
                       np.dot(R, (self.center - cor).point) + cor.point,
                       self._renderDist)

    def rotate(self, theta, cor=None):
        """Solid3D.rotate(theta, cor=self.center) -> Solid3D
            theta: x, y, and z rotations
            cor: center of rotation
        rotates the solid and returns it"""
        if cor is None:
            cor = self.center
        else:
            cor = Point3D(cor)
        R = self.center.rotation_matrix(theta)
        self.vertices -= cor
        self.vertices = np.array([Point3D(np.dot(R, p.point)) for p in self.vertices]) + cor
        self.center = Point3D(np.dot(R, (self.center - cor).point)) + cor
        # return self.copy()

    def get_matrix_rotated(self, rotMat, cor=None):
        """Solid3D.get_matrix_rotated(rotMat, cor=self.center) -> Solid3D
            rotMat: rotation matrix
            cor: center of rotation
        copies the solid then rotates it"""
        if cor is None:
            cor = self.center
        else:
            cor = Point3D(cor)

        return Solid3D([np.dot(rotMat, (p - cor).point) + cor.point for p in self.vertices],
                       self.faces,
                       self.color,
                       np.dot(rotMat, (self.center - cor).point) + cor.point,
                       self._renderDist)

    def matrix_rotate(self, rotMat, cor=None):
        if cor is None:
            cor = self.center
        else:
            cor = Point3D(cor)

        self.vertices -= cor
        self.vertices = np.array([Point3D(np.dot(rotMat, p.point)) for p in self.vertices]) + cor
        self.center = Point3D(np.dot(rotMat, (self.center - cor).point)) + cor

    def get_shifted(self, offset):
        offset = Point3D(offset)
        return Solid3D(self.vertices + offset,
                       self.faces,
                       self.color,
                       self.center + offset,
                       self._renderDist)

    def shift(self, offset):
        offset = Point3D(offset)
        self.center += offset
        self.vertices += offset
        # return self.copy()

    def get_placed(self, point):
        point = Point3D(point)
        return self.get_shifted(point - self.center)

    def place(self, point):
        point = Point3D(point)
        # return self.shift(point-self.center)

    def project(self, camera=Camera3D(pos=(0, 0, -1000), theta=(0, 0, 0)), perspective=True):
        return np.array([p.project(camera, perspective) for p in self.vertices])

    def median(self):
        return np.sum(self.vertices) / np.shape(self.vertices)[0]

    def draw_shadow(self, surface, color=(50, 50, 50), camera=Camera3D(pos=(0, 0, -1000), theta=(0, 0, 0)),
                    offset=(0, 0, 0), theta=(0, 0, 0), perspective=True):
        """draws the solid's shadow"""
        offset = Point3D(offset)
        theta = np.add(camera.theta, np.array(theta))
        R1 = self.center.rotation_matrix(-camera.theta)  # camera rotation
        R2 = self.center.rotation_matrix(theta)  # object rotation
        offset.set(np.dot(R2, offset.point))

        # calculate 2D coordinates
        shadow = []
        for p in self.vertices:
            p.set_y(self.center.y)
            p.set(np.dot(R2, np.dot(R1, (p - camera).point) + (offset - self.center).point) + self.center.point)
            shadow.append(p.project(camera, perspective))
        shadow = np.array(shadow)

        # draw
        for face in self.faces:
            pygame.draw.polygon(surface, color, list(shadow[face]))

    def draw(self, surface, camera=Camera3D(pos=(0, 0, -1000), theta=(0, 0, 0)), **kwargs):
        """Solid3D.draw(
            surface,
            camera=Camera3D(pos=(0,0,-1000),theta=(0,0,0)),
            theta=(0,0,0),
            cor=Solid3D.center,
            pos=Solid3D.center,
            offset=(0,0,0),
            outline=0,
            shadow=False,
            shadowTheta=(0,0,0),
            shadowOffset=(0,0,0),
            shadowColor=(50,50,50)
        ) -> None
        rotates, places, shifts, draws"""
        camera = Camera3D(camera=camera)

        theta = kwargs['theta'] if 'theta' in kwargs else (0, 0, 0)
        cor = kwargs['cor'] if 'cor' in kwargs else self.center
        pos = kwargs['pos'] if 'pos' in kwargs else self.center
        offset = kwargs['offset'] if 'offset' in kwargs else Point3D(0, 0, 0)
        perspective = kwargs['perspective'] if 'perspective' in kwargs else True

        # copy solid
        newSolid = self.get_rotated(theta, cor)
        # rotate
        # newSolid.rotate(theta, cor)
        # place & shift
        newSolid.place(pos + offset)

        # apply camera
        cz = camera.z + 0
        newSolid.shift(-camera.point)
        newSolid.rotate(-camera.theta, camera.point)
        camera.set((0, 0, 0 if perspective else cz), (0, 0, 0))

        # check if the object is in front of the camera
        if newSolid.median().z > camera.z:
            # order faces by distance to camera
            faceCoords = [newSolid.vertices[face] for face in newSolid.faces]
            faceDepths = []

            for face in faceCoords:
                midpoint = np.sum(face) / len(face)
                faceDepths.append(camera.dist_to(midpoint) * (1 if midpoint.z >= camera.z else -1))

            orderedFaces = [x for _, x in sorted(zip(faceDepths, self.faces), reverse=True)]
            faceDepths = np.array(sorted(faceDepths)) - np.min(faceDepths)
            if len(faceDepths) != 1:
                faceDepths /= np.max(faceDepths)
            else:
                faceDepths += 1

            projection = np.array([p.project(camera, perspective) for p in newSolid.vertices])

            # draw shadow
            drawShadow = kwargs['shadow'] if 'shadow' in kwargs else False

            if drawShadow:
                shadowTheta = kwargs['shadowTheta'] if 'shadowTheta' in kwargs else (0, 0, 0)
                shadowOffset = kwargs['shadowOffset'] if 'shadowOffset' in kwargs else Point3D(0, 0, 0)
                shadowColor = kwargs['shadowColor'] if 'shadowColor' in kwargs else (50, 50, 50)
                newSolid.draw_shadow(surface, shadowColor, camera, shadowOffset, shadowTheta, perspective=perspective)

            # draw solid
            outline = kwargs['outline'] if 'outline' in kwargs else -1

            for i, face in enumerate(orderedFaces[self.facesRendered:]):
                if self.color is not None:
                    pygame.draw.polygon(surface,
                                        tuple(self.color * faceDepths[i+self.facesRendered]),
                                        list(projection[face]))
                if outline > 0:
                    pygame.draw.polygon(surface,
                                        tuple(np.array([50, 50, 50]) * faceDepths[i+self.facesRendered]),
                                        list(projection[face]),
                                        outline)

    def __eq__(self, solid2):
        return np.array_equal(self.vertices, solid2.vertices) and np.array_equal(self.faces, solid2.faces)
