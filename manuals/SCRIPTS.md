# The scripts
[← Back to main page](./INDEX.md)

The following scripts are available:

* `autotrack.py` - starts a graphical program, which is used to visualize all data.
* `autotrack_compare_positions.py` - compare two sets of position detection data
* `autotrack_compare_links.py` - compare two sets of linking data.
* `autotrack_create_links.py` - uses a cell positions file to link cells from different time points together.
* `autotrack_detect_gaussian_shapes.py` - uses the raw images and provided cell positions to 
* `autotrack_extract_mother_scores.py` - creates a CSV file, showing how the mother scores are built up.

The command-line scripts have documentation included inside themselves. The visual scripts are all essentially the same, and are documented in [VISUALIZER].

Running the scripts
-------------------

You can always run a script using the Python command from an Anaconda terminal:

    python <path/to/script_name>.py

Or, if you are fine with using your default Python 3 installation of your system, you can run the script as follows:

    <path/to/script_name>.py

On Windows, replace ".py" by ".bat" or simply leave out the extension altogether: (".bat" is one of the default file extensions). If you don't want to always type out the complete path to the script, then you need to add the Autotrack folder to your system's PATH.
