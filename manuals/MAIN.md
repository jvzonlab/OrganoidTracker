Autotrack
=========

Autotrack is a program designed for automating the detection and linking of cells. It assists in making manual
corrections to the data.

The intended workflow is as follows:

* Obtain cell positions
* Obtain cell shapes
* Link the cells of different time points together

The following scripts are available:

Visual:
* `autotrack.py` - starts the program without any data loaded. The other visual scripts preload some data, but apart
  from that there is nothing special about them: every visualizer can perform the tasks of any other visualizer, after
  clicking a bit through the menus.
* `autotrack_show_images.py` - shows images in an directory, but no other metadata.
* `autotrack_visualize_and_edit.py` - shows images, cell positions and cell links. Used to edit linking data.
* `autotrack_compare_links.py` - shows images, cell positions and two sets of linking data. Used to compare the two.

Command-line:
* `autotrack_convert_from_manual.py` - converts Guizela's track format to a cell positions file, discarding linking
  data.
* `autotrack_create_links.py` - uses a cell positions file to link cells from different time points together.
* `autotrack_extract_mother_scores.py` - creates a CSV file, showing how the mother scores are built up.

Other:
* `track_manually.py` - graphical program written by Guizela to manually track cells. See [TRACK_MANUALLY] for details.

The command-line scripts have documentation included inside themselves. The visual scripts are all essentially the same,
and are documented in [VISUALIZER].

[TRACK_MANUALLY]: TRACK_MANUALLY.md
[VISUALIZER]: VISUALIZER.md