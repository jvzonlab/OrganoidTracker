"""Image preloading, for the case where you need to sequentially load all images in an experiment."""

import threading
from abc import ABC
from typing import Optional, Dict

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageChannel, ImageLoader
from organoid_tracker.core.images import Image, Images
from organoid_tracker.core.position import Position

# We use this as a placeholder for time points that don't have an image, so that we can distinguish between
# "not loaded yet" and "loaded but no image for this time point".
_NONE_IMAGE = numpy.array([])

def _to_image_or_none(image_array: numpy.ndarray, offset: Position) -> Optional[Image]:
    """Converts the given image array to an Image object, or returns None if the image array is empty
    (which we use as a placeholder for time points that don't have an image)."""
    if image_array is _NONE_IMAGE:
        return None
    else:
        return Image(image_array, offset)


class ImagePreloader(ABC):
    """Abstract class for preloading images. Implementations of this class can load images in the background and keep them
    in memory, with a certain window of time points. This can speed up loading images when you need to load multiple
    time points in a row, for example when predicting positions over time."""

    def __enter__(self):
        """Starts the preloader. For example, if the preloader uses a background thread to load images, it can start
        that thread here. It is required to use this class as a context manager (using the "with" statement),
        so that resources can be properly released when done."""
        raise NotImplementedError()

    def get_image(self, time_point: TimePoint) -> Optional[Image]:
        """Returns the image for the given time point, or None if there is no image for this time point. Also preloads
        images for future time points in the background.

        The first time you call this method, you can call it with any time point. You can thereafter call it with a
        time point in the past (up to older_time_points_to_keep time points in the past), the same time point,
        or one time point in the future. In that case, the time window shifts one forward. The oldest image is unloaded,
        and a new image is loaded in the background.

        You cannot skip time points, so you cannot call it with a time point that is more than one time point in the
        future compared to the then-latest requested time point.

        Raises RuntimeError if you call this method before entering the context manager (before __enter__ was called).
        """
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def create_image_preloader(images: Images, channel: ImageChannel, *, preload_window_size: int = 1,
                           older_time_points_to_keep: int = 0, use_threading: bool = True) -> ImagePreloader:
    """Creates an image preloader. If use_threading is True, it will load images in the background using a separate thread.
    If use_threading is False, it will not load images in the background, but it will still keep track of the latest
    requested time point and enforce the allowed time point window. (So essentially, it just makes sure your code is
    ready for threading.)"""
    if use_threading:
        return _ThreadedImagePreloader(images, channel, preload_window_size=preload_window_size,
                                       older_time_points_to_keep=older_time_points_to_keep)
    else:
        return _NullImagePreloader(images, channel, preload_window_size=preload_window_size,
                                   older_time_points_to_keep=older_time_points_to_keep)


class _NullImagePreloader(ImagePreloader):
    """Does not preload any images, and does not keep any images in memory. However, it does keep track of whether
    you're staying within the allowed time point window. Useful if you don't want to use any background loading,
    but still checks that your loading pattern is correct."""

    _latest_time_point_requested: Optional[TimePoint]
    _preload_window_size: int  # How many time points in front of _latest_time_point_requested to preload in memory
    _older_time_points_to_keep: int  # How many time points before _latest_time_point_requested to keep in memory
    _images: Images
    _channel: ImageChannel
    _entered: bool = False

    def __init__(self, images: Images, channel: ImageChannel, *, preload_window_size: int, older_time_points_to_keep: int):
        self._images = images
        self._channel = channel
        self._preload_window_size = preload_window_size
        self._older_time_points_to_keep = older_time_points_to_keep

        self._latest_time_point_requested = None

    def __enter__(self):
        self._entered = True
        return self

    def get_image(self, time_point: TimePoint) -> Optional[Image]:
        if self._latest_time_point_requested is None:
            # Initially load the first time point and preload the next ones
            self._latest_time_point_requested = time_point
            return self._images.get_image(time_point, self._channel)

        elif time_point > self._latest_time_point_requested:
            # Loading from the future
            if time_point - 1 > self._latest_time_point_requested:
                raise ValueError(
                    f"Time points must be requested in order, but got {time_point} after {self._latest_time_point_requested}")
            self._latest_time_point_requested = time_point

            # We confirmed we're loading one time point into the future
            return self._images.get_image(time_point, self._channel)
        elif time_point < self._latest_time_point_requested:
            # Loading from the past
            delta_time_points = self._latest_time_point_requested.time_point_number() - time_point.time_point_number()
            if delta_time_points > self._older_time_points_to_keep:
                raise ValueError(f"Time point {time_point} is too old to be kept in memory (latest requested time point"
                                 f" is {self._latest_time_point_requested}, and older_time_points_to_keep={self._older_time_points_to_keep})")
            return self._images.get_image(time_point, self._channel)
        else:
            # Time point is the same as the latest requested time point
            return self._images.get_image(time_point, self._channel)


class _ThreadedImagePreloader(ImagePreloader):
    """Loads a set of images in the background and keeps them in memory. It only keeps certain time points in memory,
    with a window of size X. Once a new time point is requested, the oldest time point is discarded, and a new image
     (of one or more time points ahead) is already loaded in the background."""


    _latest_time_point_requested: Optional[TimePoint]
    _preload_window_size: int  # How many time points in front of _latest_time_point_requested to preload in memory
    _older_time_points_to_keep: int  # How many time points before _latest_time_point_requested to keep in memory

    _cached_images: Dict[TimePoint, numpy.ndarray]
    _images: Images
    _image_loader: ImageLoader  # This loader is uncached, unlike calling on _images.get_image(...). We don't need that cache, since we have our own here
    _channel: ImageChannel

    _image_loading_thread: Optional[threading.Thread] = None
    _images_ready_signal: threading.Event
    _images_requested_signal: threading.Event
    _stop_requested: bool

    def __init__(self, images: Images, channel: ImageChannel, *, preload_window_size: int, older_time_points_to_keep: int):
        self._images = images
        self._image_loader = images.image_loader()
        self._channel = channel
        self._preload_window_size = preload_window_size
        self._older_time_points_to_keep = older_time_points_to_keep

        self._latest_time_point_requested = None
        self._cached_images = dict()

        self._images_ready_signal = threading.Event()
        self._images_requested_signal = threading.Event()
        self._stop_requested = False

    def __enter__(self):
        self._image_loading_thread = threading.Thread(target=self._preload_images_in_background, daemon=True)
        self._image_loading_thread.start()
        return self

    def _preload_images_in_background(self):
        """Preloads images in the background, starting from the latest requested time point and going forward."""
        while True:
            # Wait until a new time point is requested
            self._images_requested_signal.wait()
            self._images_requested_signal.clear()
            if self._stop_requested:
                break

            latest_time_point_requested = self._latest_time_point_requested
            if latest_time_point_requested is None:
                raise RuntimeError("ImagePreLoader: thread started before any time point was requested")

            # Preload images for the next time points
            for i in range(0, self._preload_window_size + 1):
                time_point_to_load = latest_time_point_requested + i
                if time_point_to_load in self._cached_images:
                    continue  # Already loaded
                image = self._image_loader.get_3d_image_array(time_point_to_load, self._channel)
                if image is None:
                    image = _NONE_IMAGE
                self._cached_images[time_point_to_load] = image
                self._images_ready_signal.set()  # Signal that some new image is ready

                # Check for requests to stop, so that we don't unnecessarily load a few more images
                if self._stop_requested:
                    break

    def get_image(self, time_point: TimePoint) -> Optional[Image]:
        if self._image_loading_thread is None:
            raise RuntimeError("ImagePreLoader: get_image called before entering context manager (before __enter__ was called)")
        if self._latest_time_point_requested is None:
            # Initially load the first time point and preload the next ones
            self._latest_time_point_requested = time_point
            self._images_requested_signal.set()  # Signal the background thread to load the first image
            self._images_ready_signal.wait()  # Wait until the first image is loaded
            self._images_ready_signal.clear()

            return _to_image_or_none(self._cached_images[time_point], self._images.offsets.of_time_point(time_point))

        elif time_point > self._latest_time_point_requested:
            # Loading from the future
            if time_point - 1 > self._latest_time_point_requested:
                raise ValueError(f"Time points must be requested in order, but got {time_point} after {self._latest_time_point_requested}")

            # We confirmed we're loading one time point into the future
            self._latest_time_point_requested = time_point
            self._images_requested_signal.set()  # Signal the background thread to load the next image

            # Wait until the requested image is loaded
            while time_point not in self._cached_images:
                # Ideally, we should never enter this loop, since the background thread should load the images in advance.
                # A print statement can be added here for debugging if needed, to check if we're actually waiting here.
                self._images_ready_signal.wait()
                self._images_ready_signal.clear()

            return _to_image_or_none(self._cached_images[time_point], self._images.offsets.of_time_point(time_point))
        elif time_point < self._latest_time_point_requested:
            # Loading from the past
            delta_time_points = self._latest_time_point_requested.time_point_number() - time_point.time_point_number()
            if delta_time_points > self._older_time_points_to_keep:
                raise ValueError(f"Time point {time_point} is too old to be kept in memory (latest requested time point"
                                 f" is {self._latest_time_point_requested}, and older_time_points_to_keep={self._older_time_points_to_keep})")
            return _to_image_or_none(self._cached_images[time_point], self._images.offsets.of_time_point(time_point))
        else:
            # Time point is the same as the latest requested time point
            return _to_image_or_none(self._cached_images[time_point], self._images.offsets.of_time_point(time_point))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the preloader and stops the background thread."""
        self._stop_requested = True
        self._images_requested_signal.set()  # Wake up the thread so it can exit
        self._image_loading_thread.join()
        self._image_loading_thread = None
