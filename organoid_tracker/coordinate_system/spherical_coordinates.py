import math
from typing import Any, Optional

from organoid_tracker.core.vector import Vector3


class SphericalCoordinate:
    """Spherical coordinate using the ISO convention."""

    @staticmethod
    def from_cartesian(vector_um: Vector3) -> "SphericalCoordinate":
        """Gets the equivalent spherical coordinate."""
        x_squared_plus_y_squared = vector_um.x ** 2 + vector_um.y ** 2
        radius = math.sqrt(x_squared_plus_y_squared + vector_um.z ** 2)
        theta = math.atan2(vector_um.z, math.sqrt(x_squared_plus_y_squared))
        phi = math.atan2(vector_um.y, vector_um.x)

        # Switch to ISO convention
        if theta < 0:
            theta += math.pi
        if phi < 0:
            phi += 2 * math.pi

        return SphericalCoordinate(radius, math.degrees(theta), math.degrees(phi))

    radius_um: float  # Radius, r >= 0. Read-only.
    theta_degrees: float  # Inclination, 0 <= theta <= 180. Read-only.
    phi_degrees: float  # Azimuth, 0 <= phi <= 360. Read-only.

    def __init__(self, radius_um: float, theta_degrees: float, phi_degrees: float):
        if radius_um < 0 or theta_degrees < 0 or theta_degrees > 180 or phi_degrees < 0 or phi_degrees > 360:
            raise ValueError(f"Invalid spherical coordinate: ({radius_um}, {theta_degrees}, {phi_degrees})")

        self.radius_um = radius_um
        self.theta_degrees = theta_degrees
        self.phi_degrees = phi_degrees

    def to_cartesian(self, *, radius: Optional[float] = None) -> Vector3:
        """Gets the equivalent cartesian coordinate. You can override the radius with another value."""
        if radius is None:
            radius = self.radius_um

        phi = math.radians(self.phi_degrees)
        theta = math.radians(self.theta_degrees)

        x = radius * math.cos(phi) * math.sin(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(theta)
        return Vector3(x, y, z)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SphericalCoordinate):
            return False
        return other.radius_um == self.radius_um\
               and other.phi_degrees == self.phi_degrees\
               and other.theta_degrees == self.theta_degrees

    def __hash__(self) -> int:
        return hash(self.radius_um) ^ hash(self.theta_degrees) ^ hash(self.phi_degrees)

    def __repr__(self) -> str:
        return f"SphericalCoordinate({self.radius_um}, {self.theta_degrees}, {self.phi_degrees})"
