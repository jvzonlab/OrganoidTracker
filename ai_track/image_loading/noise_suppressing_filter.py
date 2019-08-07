from numpy.core.multiarray import ndarray

from ai_track.core.image_loader import ImageFilter


class NoiseSuppressingFilter(ImageFilter):

    _noise_limit: int  # Scaled 0 to 255

    def __init__(self, noise_limit: float = 0.08):
        """noise_limit = 0.5 would remove all pixels of less than 50% of the max intensity of the image."""
        self._noise_limit = int(noise_limit * 255)

    def filter(self, image_8bit: ndarray):
        image_8bit[image_8bit < self._noise_limit] = 0

    def copy(self) -> ImageFilter:
        return NoiseSuppressingFilter(self._noise_limit / 255)

    def get_name(self) -> str:
        return "Suppress noise"
