# Automatic tracking
[← Back to main page](INDEX.md)

The intended workflow is as follows:

1. Obtain nucleus positions (for now this is done using an external program)
2. Obtain nucleus shapes from positions
3. Link the cells of different time points together
4. Manually correct all warnings

The steps are described below.

Before you start, create a directory that will hold all you analysis data. You can create this directory inside the directory that holds your images.

Step 1: Obtaining nucleus positions
-----------------------------------

Obtaining nucleus positions from raw tracking data is currently done using an external program, that is not yet integrated in Autotrack. See the documentation of that program on how to use it. Once you have the positions (those are *.npy files), import them using the Autotrack GUI. Use `File -> Load positions in Laetitia's format` to import the positions and export them again using `File -> Save tracking data...`. You can name the file `Automatic positions.aut`.

In the future, a machine learning script that outputs all data in the expected format will be added, so that you don't need to install extra programs and convert data.

Step 2: Obtaining nucleus shapes
--------------------------------

Obtaining nucleus shapes is done using the `autotrack_detect_gaussian_shapes.py` file. Run it from the data analysis directory. It will ask you to set all parameters. Make sure the resolution, the path to the image folder and the path to the positions file are correct, and then run the script again.

Note: this script takes a few minutes to run per time point, so please be patient.

Step 3: Obtaining links
-----------------------

Automatic linking is relatively fast: running the script should only take a few minutes.

Run the script `autotrack_create_links.py`. It will ask you to make sure that the paths to the images and to the nucleus positions and shapes are correct.

Step 4: Manually correct warnings
---------------------------------

This step is not automated. 😉 Open the Autotrack GUI and load the images and data (`File` menu). Then go to `Edit -> Manually change data...`. Now you can use `View -> Lineage errors and warnings` to view which lineages have warnings in them. Lineages that are marked green are OK, lineages that are marked in gray need to be manually revisted.

To do this, hover you mouse over a nucleus and press E. You will be shown all errors in the lineage tree. Use the left and right arrow keys to move to the next error. Press E again to exit that view, and edit the data.

Data editing is done mainly using the Insert and Delete keys, which are used to insert and delete links. Detected positions can also be inserted and deleted if necessary.
