import random
from typing import Optional, Tuple, NamedTuple

import matplotlib.cm
from matplotlib.colors import Colormap, ListedColormap, LinearSegmentedColormap

# The colormaps, in order of appearance in the GUI
COLORMAP_LISTS = {
    "basic": ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"],
    "uniform": ["viridis", "plasma", "inferno", "magma", "cividis"],
    "misc": ["bone", "pink", "hot", "afmhot", "gist_heat", "copper"],
    "segmentation": ["segmentation"]
}

# Initialized in get_colormap
_CACHED_COLORMAPS = None


def _create_segmentation_colormap() -> Colormap:
    """Create a colormap for segmentation masks."""
    source_colormap: Colormap = matplotlib.cm.jet
    samples = [source_colormap(sample_pos / 1000) for sample_pos in range(1000)]
    random.Random("fixed seed to ensure same colors").shuffle(samples)
    samples[0] = (0, 0, 0, 0)  # Force background to black
    return ListedColormap(samples, name="segmentation")


def _create_black_to_color_colormap(color_name: str, max_color_rgb: Tuple[float, float, float]) -> Colormap:
    """Create a colormap that goes from black to a given RGB color."""
    return LinearSegmentedColormap.from_list(color_name, [(0.0, 0.0, 0.0), max_color_rgb])


def _create_colormap(name: str) -> Colormap:
    """Internal function to create a colormap by name. Will also create colormaps for names outside the allowed list."""
    if name == "segmentation":
        return _create_segmentation_colormap()
    if name == "red":
        return _create_black_to_color_colormap("red", (1.0, 0.0, 0.0))
    if name == "green":
        return _create_black_to_color_colormap("green", (0.0, 1.0, 0.0))
    if name == "blue":
        return _create_black_to_color_colormap("blue", (0.0, 0.0, 1.0))
    if name == "cyan":
        return _create_black_to_color_colormap("cyan", (0.0, 1.0, 1.0))
    if name == "magenta":
        return _create_black_to_color_colormap("magenta", (1.0, 0.0, 1.0))
    if name == "yellow":
        return _create_black_to_color_colormap("yellow", (1.0, 1.0, 0.0))
    return matplotlib.cm.get_cmap(name)


def get_colormap(name: Optional[str]) -> Colormap:
    """Load a colormap by name. If the name is not in the allowed list, the gray colormap will be returned.

    Note that these colormaps can be different from the built-in Matplotlib colormaps. For example, "green" returns
    a colormap that goes from black to pure green (#00FF00), while Matplotlib's "Greens" colormap goes from white to
    another shade of green.
    """
    global _CACHED_COLORMAPS
    if _CACHED_COLORMAPS is None:
        _CACHED_COLORMAPS = {}
        for sublist in COLORMAP_LISTS.values():
            for name_to_load in sublist:
                _CACHED_COLORMAPS[name_to_load] = _create_colormap(name_to_load)

    if name not in _CACHED_COLORMAPS:
        # If the name is not in the allowed list, simply return the gray colormap
        name = "gray"

    return _CACHED_COLORMAPS[name]


def get_segmentation_colormap() -> Colormap:
    """Get the colormap for segmentation masks."""
    return get_colormap("segmentation")