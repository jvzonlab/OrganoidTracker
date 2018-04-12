Tracking code
=============

Code for tracking the positions of cells. Tracking consists of two tasks: identification and linking. Identification
is the process where the positions of cells are determined in single images, while linking is the process where it is
determined wat cell at time point t corresponds to wat cell(s) in time point t + 1.


Setting up the environment
--------------------------
To be able to run the scripts, you first need to install the appropriate libraries. That is:

* Anaconda
* The `tifffile` package from Anaconda Forge
* The `opencv` package from Anaoconda Forge 

If you want to have an exact copy of my environment, run the following commands: (they require
Anaconda to be installed)

    conda env create -f environment.yml
    activate rutger-tracking

(On macOs, run `source activate` instead of `activate`.)


Running the scripts
-------------------

AT the moment, there are two types of scripts in this folder. The older ones need to be run like this:

    python <script_name>.py

They need to be run from this folder. You'll need to edit the scripts beforehand to correct the paths
inside them.

Newer scripts (they have a name starting with `autotrack_`) are easier to run. You can run them from any
folder, and they don't need to be edited beforehand. However, you need to add this folder (the folder where
this README file is stored) to the system PATH first. After you have done that, on Unix systems (Mac/Linux)
you can run the scripts like this:

    <script_name>.py

For Windows, I created `.bat` files. As a result, you can run the scripts simply by running:

    <script_name>.bat

(You can even leave out the `.bat` extension if you want.) The first time you run any script, it will ask
for some parameters, which are then saved to a configuration file.
   
For now, we have the following scripts:


autotrack_visualize_and_edit.py
-------------------------------
Generic data visualization tool that does not do any calculations first. Loads images from TIFF files, positions and
tracks from JSON files. In [the manuals folder](manuals/VISUALIZER.md) a manual is available. This program can be used
to edit positional and linking data.


autotrack_show_images.py
------------------------
Stripped down version of the visualizer that only shows the images, no positions or links.


track_manually.py
-----------------
Code by Guizela, modified by Rutger. This script is used to manually track the particles. In
[the manuals folder](manuals/TRACK_MANUALLY.md) a short manual is available.


autotrack_convert_from_manual.py
--------------------------------
Extracts the positions from the above manually obtained data. This throws away all trajectory information, which is on
purpose, as our linking algorithm should be able to reconstruct it. See the file itself for instructions.


link_nearest_neighbors.py
-------------------------
Simple nearest neighbor-linking.


autotrack_create_links.py
-------------------------
Nearest neighbor linking with some error corrections afterwards.

1. IF a cell is dead (has no future positions), nearby cells are checked. If a cell appears to be newborn, it will
   be connected to the dead cell instead.
2. If a cell appears to be a mother cell (has two linked positions in the next timepoint), a scoring system is used to
   give it a mother cell. If a nearby cell has a higher mother score, that cell becomes the mother instead.


autotrack_compare_links.py
--------------------------
Assists in comparing two linking results. The "scratchpad" links are shown as dotted lines, the "verified" links as
solid lines.


detect_cells_*.py
-----------------
Several detectors using (slightly) different algorithms to extract cell positions. The detector can also show insights
into the algorithm and display earlier results.
