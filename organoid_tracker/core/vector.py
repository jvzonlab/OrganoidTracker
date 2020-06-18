from typing import Union

import math


class Vector2:
    ZERO: "Vector2"

    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def dot(self, other: "Vector2") -> float:
        """Returns the dot product."""
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        """Length of this vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def multiply(self, amount: float) -> "Vector2":
        """Scalar multiplication."""
        return Vector2(self.x * amount, self.y * amount)

    def __add__(self, other) -> "Vector2":
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> "Vector2":
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x - other.x, self.y - other.y)

    def distance(self, other: "Vector3") -> float:
        """Gets the distance to the other point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def __repr__(self) -> str:
        return "Vector2(" + str(self.x) + ", " + str(self.y) + ")"

    def __hash__(self) -> int:
        return hash(int(self.x)) ^ hash(int(self.y))

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) \
               and abs(self.x - other.x) < 0.01 and abs(self.x - other.x) < 0.01

    def normalized(self) -> "Vector2":
        """Returns a new vector with the same orientation, but with a length of 1."""
        length = self.length()
        return Vector2(self.x / length, self.y / length)

    def to_vector3(self, *, z: float) -> "Vector3":
        """Returns a Vector3 with the current x and y and the given z."""
        return Vector3(self.x, self.y, z)


class Vector3:

    ZERO: "Vector3"  # Initialized after this class is defined

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def cross(self, other: "Vector3") -> "Vector3":
        """Returns the cross product of this vector with another vector"""
        return Vector3((self.y * other.z - self.z * other.y),
                       (self.z * other.x - self.x * other.z),
                       (self.x * other.y - self.y * other.x))

    def dot(self, other: "Vector3") -> float:
        """Returns the dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        """Length of this vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def multiply(self, amount: float) -> "Vector3":
        """Scalar multiplication."""
        return Vector3(self.x * amount, self.y * amount, self.z * amount)

    def __add__(self, other) -> "Vector3":
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other) -> "Vector3":
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance(self, other: "Vector3") -> float:
        """Gets the distance to the other point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def __repr__(self) -> str:
        return "Vector3(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __hash__(self) -> int:
        return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z))

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) \
               and abs(self.x - other.x) < 0.01 and abs(self.x - other.x) < 0.01 and abs(self.z - other.z) < 0.01

    def normalized(self) -> "Vector3":
        """Returns a new vector with the same orientation, but with a length of 1."""
        length = self.length()
        return Vector3(self.x / length, self.y / length, self.z / length)


Vector3.ZERO = Vector3(0, 0, 0)
