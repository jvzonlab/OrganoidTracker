import numpy
from matplotlib import colors, cm
from matplotlib.colors import ListedColormap

# Colors Sander Tans likes in his figures - these are all bright colors
# Source: flatuicolors.com - US palette
SANDER_APPROVED_COLORS = colors.to_rgba_array({"#55efc4", "#fdcb6e", "#636e72", "#74b9ff", "#e84393",
                                               "#ff7675", "#fd79a8", "#fab1a0", "#e84393"})

# A bit darker colors
BAR_COLOR_1 = "#0984e3"
BAR_COLOR_2 = "#d63031"


def _create_qualitative_colormap():
    color_list = cm.get_cmap('jet', 256)(numpy.linspace(0, 1, 256))
    numpy.random.shuffle(color_list)
    color_list[0] = numpy.array([0, 0, 0, 1])  # First color is black
    return ListedColormap(color_list)

QUALITATIVE_COLORMAP = _create_qualitative_colormap()
