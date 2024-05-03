from random import Random
from typing import List, Any, Iterable, Optional

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


class LimitingDataset(IterableDataset):
    """Wraps an IterableDataset and limits the number of samples that are yielded."""

    _internal_dataset: IterableDataset
    _max_samples: int

    def __init__(self, dataset: IterableDataset, max_samples: int):
        self._internal_dataset = dataset
        self._max_samples = max_samples

    def __iter__(self) -> Iterable[Any]:
        for i, sample in enumerate(self._internal_dataset):
            if i >= self._max_samples:
                break
            yield sample

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
        for sample in self._internal_dataset:
            # Keep on collecting samples until buffer is full
            buffer.append(sample)

            if len(buffer) >= self._buffer_size:
                # Time to shuffle buffer and yield all samples
                self._random.shuffle(buffer)
                yield from buffer
                buffer.clear()

        # Yield the remaining samples
        self._random.shuffle(buffer)
        yield from buffer

    def __len__(self) -> int:
        # Will throw a TypeError if the internal dataset doesn't have a __len__ method
        # noinspection PyTypeChecker
        return len(self._internal_dataset)
