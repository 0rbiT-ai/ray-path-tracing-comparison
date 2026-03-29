from core.utils import normalize

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)