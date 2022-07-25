"""
Tons of code for analysis on links, just have a look around.

One example: this is how to draw a lineage tree:

>>> from organoid_tracker.core.experiment import Experiment
>>> import matplotlib.pyplot as plt
>>> experiment = Experiment()  # placeholder
>>>
>>> from organoid_tracker.linking_analysis import lineage_drawing
>>> ax = plt.gca()
>>> width = lineage_drawing.LineageDrawing(experiment.links).draw_lineages_colored(ax)
>>> ax.set_xlim(0, width)

"""
