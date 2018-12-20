Autotrack
=========

Autotrack is a program designed for automating the detection and linking of cells. It assists in making manual corrections to the data. It also contains a Python library for working with the data sets, see the [API] page for details.

The intended workflow is as follows:

1. Obtain nucleus positions (for now this is done using an external program)
2. Obtain nucleus shapes
3. Link the cells of different time points together
4. Manually correct all warnings

The following scripts are available:

Visual:
* `autotrack.py` - starts the program without any data loaded. The other visual scripts preload some data, but apart from that there is nothing special about them: every visualizer can perform the tasks of any other visualizer, after clicking a bit through the menus.
* `autotrack_show_images.py` - shows images in an directory, but no other metadata.
* `autotrack_visualize_and_edit.py` - shows images, cell positions and cell links. Used to edit linking data.

Command-line:
* `autotrack_convert_from_manual.py` - converts Guizela's track format to a cell positions file, discarding linking data.
* `autotrack_create_links.py` - uses a cell positions file to link cells from different time points together.
* `autotrack_extract_mother_scores.py` - creates a CSV file, showing how the mother scores are built up.
* `autotrack_compare_positions.py` - compare two sets of position detection data
* `autotrack_compare_links.py` - compare two sets of linking data.

The command-line scripts have documentation included inside themselves. The visual scripts are all essentially the same, and are documented in [VISUALIZER].

[API]: API.md
[VISUALIZER]: VISUALIZER.md
