"""
This package contains code for a "biological" coordinate system for organoids. In this system, organoid are represented
as a sphere, and cells have polar coordinates on that sphere.

To go from cartesian coordinates to spherical coordinates:

>>> from organoid_tracker.core.vector import Vector3
>>> vector3 = Vector3(30, 50, 3)
>>> from organoid_tracker.coordinate_system.spherical_coordinates import SphericalCoordinate
>>> vector_spherical_coords = SphericalCoordinate.from_cartesian(vector3)
>>> print(vector_spherical_coords.radius_um, vector_spherical_coords.phi_degrees, vector_spherical_coords.theta_degrees)

The class :class:`~organoid_tracker.sphere_representer.SphereRepresentation` is used to draw the organoid as a sphere.
You can draw points and tracks on the sphere. The center point is defined by the beacons of the experiment.
"""
