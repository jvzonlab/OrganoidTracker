Tracking code
=============

Code for tracking the positions of cells. Tracking consists of two tasks: identification and linking. Identification
is the process where the positions of cells are determined in single images, while linking is the process where it is
determined wat cell at frame t corresponds to wat cell(s) in frame t + 1.


Setting up the environment
--------------------------
To be able to run the scripts, you first need to install the appropriate libraries. That is:

* Anaconda
* The `tifffile` package from Anaconda Forge

If you want to have an exact copy of my environment, run the following commands: (they require
Anaconda to be installed)

    conda env create -f environment.yml
    activate rutger-tracking

(On macOs, run `source activate` instead of `activate`.)


Running the scripts
-------------------

All Python scripts in the project root (this folder) are intended to be executed from the command line
using:

    python <script_name>.py
   
For now, we have the following scripts:


visualize.py
------------
Generic data visualization tool that does not do any calculations first. Loads images from TIFF files, positions and
tracks from JSON files. In [the manuals folder](manuals/VISUALIZER.md) a manual is available.


track_manually.py
-----------------
Code by Guizela, modified by Rutger. This script is used to manually track the particles. In
[the manuals folder](manuals/TRACK_MANUALLY.md) a short manual is available.


extract_manual_positions.py
---------------------------
Extracts the positions from the above manually obtained data. This throws away all trajectory information, which is on
purpose, as our linking algorithm should be able to reconstruct it. See the file itself for instructions.


link_nearest_neighbors.py
-------------------------
Simple nearest neighbor-linking.

compare_links.py
----------------
Assists in comparing linking results of 
