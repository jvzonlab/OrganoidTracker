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

track_manually.py
-----------------

Code by Guizela, modified by Rutger. This script is used to manually track the particles. Short manual:

### Keys:

1​: Previous time frame  
2​: Next time frame  
3​: Move 5 frames forwards  
4​: Move 5 frames backwards

z​: Show current time frame in green and next time point in red
x​: Show current time frame (only here you can create and delete tracks)
c​: Show next time frame in red

o​: Show brightfield

q​: Slice below  
w​: Slice above  
e​: 5 slices below  
r​: 5 slices above

-​: Go to first timepoint of selected track
=​: Go to last timepoint of selected track

n​: Create track where mouse is positioned  
Left click​: Move selected track to where mouse is positioned  
Right click​: Select track  
Del​: Delete point in selected track  
Space​: Goes to z of selected track 

Press ​escape ​to save tracks!!! 

Selected track is shown in red, others in gray. Arrows indicate z position of track, points indicate that z position of
track is within ± 2 z slices. Every time a cell divides, create a new track for each daughter cell. 
 
### From version 4: (Needs scikit image)  
a​: Increase contrast  
d​: Decrease contrast 

extract_manual_positions
------------------------
Extracts the positions from the above manually obtained data. This throws away all trajectory information, which is on
purpose, as our linking algorithm should be able to reconstruct it. See the file itself for instructions.

link_nearest_neighbors
----------------------
Simple nearest neighbor-linking.