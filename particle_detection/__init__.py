from numpy import ndarray


class Detector:

    def detect(self, image: ndarray, **kwargs) -> ndarray:
        """Detects all particles in the given image. Image is a 2d array of intensities. An array of
         [ [x,y], [x,y], ...], representing the particle positions, is returned.
         """
        raise ValueError("detect() method is not overridden")