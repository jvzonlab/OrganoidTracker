import threading
from queue import Queue
from random import Random
from typing import Any, Iterable, Iterator

from torch.utils.data import IterableDataset


class RepeatingDataset(IterableDataset):
    """Wraps an IterableDataset and repeats it an infinite number of times."""

    _internal_dataset: IterableDataset

    def __init__(self, dataset: IterableDataset):
        self._internal_dataset = dataset

    def __iter__(self) -> Iterable[Any]:
        while True:
            yield from iter(self._internal_dataset)

    def __len__(self) -> int:
        """We still return the length of the internal dataset, so that the user knows how long an epoch should be."""
        # Will throw a TypeError if the internal dataset doesn't have a __len__ method
        # noinspection PyTypeChecker
        return len(self._internal_dataset)


class PrefetchingDataset(IterableDataset):
    """Wraps an IterableDataset and prefetches samples into a buffer. The prefetching is done in a separate thread.

    The advantage of using multithreading instead of multiprocessing is that we can use the same memory space for the
    buffer, so we don't need to copy things and use more (V)RAM. The downside is that we can't use multiple CPU cores,
    but that's not really a problem here since the bottleneck in our case is the disk I/O.
    """

    _LAST_ELEMENT = "~~~LAST_ELEMENT~~~"

    _internal_dataset: IterableDataset
    _prefetch_buffer: Queue

    def __init__(self, dataset: IterableDataset, buffer_size: int = 5):
        self._internal_dataset = dataset
        self._prefetch_buffer = Queue(maxsize=buffer_size)

    def _run(self):
        """Starts prefetching samples from the internal dataset. This method should be called from a separate thread."""
        for sample in self._internal_dataset:
            qsize = self._prefetch_buffer.qsize()
            self._prefetch_buffer.put(sample, block=True)
        self._prefetch_buffer.put(self._LAST_ELEMENT)

    def __iter__(self) -> Iterable[Any]:
        threading.Thread(target=self._run, daemon=True).start()

        while True:
            element = self._prefetch_buffer.get(block=True)
            if element == self._LAST_ELEMENT:
                break

            yield element

    def __len__(self) -> int:
        # Will throw a TypeError if the internal dataset doesn't have a __len__ method
        # noinspection PyTypeChecker
        return len(self._internal_dataset)


class _LimitingIterator(Iterator):
    _i: int = 0
    _max_samples: int
    _internal_iterator: Iterator[Any]

    def __init__(self, internal_iterator: Iterator[Any], max_samples: int):
        self._internal_iterator = internal_iterator
        self._max_samples = max_samples

    def __next__(self) -> Any:
        self._i += 1
        if self._i > self._max_samples:
            raise StopIteration
        return next(self._internal_iterator)


class LimitingDataset(IterableDataset):
    """Wraps an IterableDataset and limits the number of samples that are yielded."""

    _internal_dataset: IterableDataset
    _max_samples: int

    def __init__(self, dataset: IterableDataset, max_samples: int):
        self._internal_dataset = dataset
        self._max_samples = max_samples

    def __iter__(self) -> Iterator[Any]:
        return _LimitingIterator(iter(self._internal_dataset), self._max_samples)


    def __len__(self) -> int:
        # Will throw a TypeError if the internal dataset doesn't have a __len__ method
        # noinspection PyTypeChecker
        return min(len(self._internal_dataset), self._max_samples)


class ShufflingDataset(IterableDataset):
    """Data loading works best when images are read in sequence, i.e. that we don't read random patches spread over
    all kinds of imaging files. Neural networks on the other hand need a random order of the data for most optimized
    learning.

    This class is a compromise between the two. It reads X number of samples in order, places them into a buffer, and
    then shuffles the buffer before yielding the samples. This way we can still read the data in sequence, but the
    neural network will see the data in a pseudo-random order.
    """

    _internal_dataset: IterableDataset
    _buffer_size: int
    _random: Random

    def __init__(self, dataset: IterableDataset, buffer_size: int = 2000, seed: int = 1):
        """Wraps an IterableDataset and shuffles the samples in a buffer before yielding them."""
        self._internal_dataset = dataset
        self._buffer_size = buffer_size
        self._random = Random(seed)

    def __iter__(self) -> Iterable[Any]:
        buffer = list()
        for incoming_sample in self._internal_dataset:
            if len(buffer) < self._buffer_size:
                # Keep on collecting samples until buffer is full
                buffer.append(incoming_sample)
            else:
                # Buffer is now full, we can start yielding random samples
                # Every time, we pick one random position, and yield the existing sample in that position,
                # and put the incoming sample in that place
                picked_index = self._random.randint(0, len(buffer) - 1)
                picked_sample = buffer[picked_index]
                buffer[picked_index] = incoming_sample
                yield picked_sample

        # No more incoming samples, yield the remaining samples in the buffer
        self._random.shuffle(buffer)
        yield from buffer

    def __len__(self) -> int:
        # Will throw a TypeError if the internal dataset doesn't have a __len__ method
        # noinspection PyTypeChecker
        return len(self._internal_dataset)
