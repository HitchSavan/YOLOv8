
from math import sqrt
import cv2

class Point:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def __add__(self, _pt):
        return Point(self.x + _pt.x, self.y + _pt.y)

    def __sub__(self, _pt):
        return Point(self.x - _pt.x, self.y - _pt.y)
    
    def __gt__(self, _pt):
        return True if self.x > _pt.x else (self.y > _pt.y if self.x == _pt.x else False)
    
    def __lt__(self, _pt):
        return self.__gt__(_pt, self)

class Vector:
    def __init__(self, _start, _end):
        self.start = _start
        self.end = _end
        self.coords = _end - _start

    def __init__(self, _start_x, _start_y, _end_x, _end_y):
        self.start = Point(_start_x, _start_y)
        self.end = Point(_end_x, _end_y)
        self.coords = self.end - self.start

    # codirectional vectors
    def __eq__(self, other):
        if ((self.coords.x / other.coords.x) == (self.coords.y / other.coords.y)):
            return 0 < (self.coords.x * other.coords.x + self.coords.y * other.coords.y)
        return False

    def distance(self, r):
        return sqrt(pow(self.coords.x - r.coords.x, 2) + pow(self.coords.y - r.coords.y, 2))

    def length(self):
        return sqrt(pow(self.coords.x, 2) + pow(self.coords.y, 2))

    # if positive --> <90 degrees
    def getScalarMult(self, r):
        return (self.coords.x * r.coords.x) + (self.coords.y * r.coords.y)

    # if cos > 0.7 --> <45 degrees
    # > 0,866 --> <30
    def getCos(self, r):
        return (self.getScalarMult(r) / (self.length() * r.length()))

    def update(self, newCoords):
        self.coords = newCoords
        self.end = self.start + newCoords

    def draw(self, image, colour, thickness=3):
        cv2.arrowedLine(image, (self.start.x, self.start.y), (self.end.x, self.end.y), colour, thickness)
