from random import Random
from typing import List, Any, Iterable, Optional

from torch.utils.data import IterableDataset


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
    _seed: int

    def __init__(self, dataset: IterableDataset, buffer_size: int = 2000, seed: int = 1):
        """Wraps an IterableDataset and shuffles the samples in a buffer before yielding them."""
        self._internal_dataset = dataset
        self._buffer_size = buffer_size
        self._seed = seed

    def __iter__(self) -> Iterable[Any]:
        buffer = list()
        random = Random(self._seed)
        for sample in self._internal_dataset:
            # Keep on collecting samples until buffer is full
            buffer.append(sample)

            if len(buffer) >= self._buffer_size:
                # Time to shuffle buffer and yield all samples
                random.shuffle(buffer)
                yield from buffer
                buffer.clear()
