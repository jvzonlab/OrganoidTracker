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

Open the OrganoidTracker GUI, load your images and select `Tools` -> `Detect cells in images...`. Select the already trained neural network that you want to use. ([See here for how to train a model](./TRAINING_THE_NETWORK.md). Find pre-trained models at [our Github page](https://github.com/jvzonlab/OrganoidTracker).) You'll end up with a folder containing configuration files. Run the `organoid_tracker_predict_positions` script in that directory to put the neural network into action. (On Windows, double-click the BAT file. On Linux/macOs, run the SH file instead.) You'll end up with a file that contains the positions of the cells. Load it into the program to see how the neural network performed.

If you want to debug what the program is doing, open the `organoid_tracker.ini` file and give the `predictions_output_folder` setting a value, for example `test`. If you run the script again, it will create a folder with that name (`test` in this example) and place images there indicating how likely it is to find a cell there.

Step 2: Obtaining nucleus shapes
--------------------------------

Open the OrganoidTracker GUI again and load the images and the positions from the previous step. Select `Tools` -> `Detect shapes using Gaussian fit...` and follow the steps. Run the `organoid_tracker_detect_gaussian_shapes` script. Note: this script takes a few minutes to run per time point, so please be patient.

The script works by fitting 3D Gaussian functions to a blurred version of the original image, starting from the positions from step 1 and a default covariance matrix. To restrict the fitting algorithm, separate clusters of cells are fitted separately. In this way, the fitting algorithm will ignore debris elsewhere in the image. Clusters are found using a quick segmentation: every pixel that has an intensity lower than the local average is background, the rest is foreground (see figure 1). To improve the segmentation, some erosion can be applied, which makes the white areas in the figure 1 smaller. If your cells are already well separated, you can set it to 0, and if your cells form one big structure (like in figure 1) then a value of 3 will be necessary.

You can view the resulting Gaussians by loading the output file in the OrganoidTracker GUI and pressing `R`. This will draw the Gaussian functions.

![Thresholding](images/thresholding.png)  
Figure 1: Thresholding, used to separate foreground and background. Gaussian fits are carried out within clumps.

Step 3: Obtaining links
-----------------------

Open the OrganoidTracker GUI again and load the images and the output file from the previous step. Use `Tools` -> `Create links between time points...` and run the resulting `organoid_tracker_create_links` script.

First, the script will create some very basic initial links. Then it will calculate the likeliness (a score) of each cell being a dividing cell. Then it will create the actual links.

If you're not satisfied with the results, try changing the input parameters in the `organoid_tracker.ini` file. There are comments in there to explain each setting. You can make cell divisions more/less likely, allow more or less cellular movement and make cell deaths more likely. If you don't see (m)any links anymore, then you have made link creation too expensive, and you should make events like cellular movement cheaper.

### Correcting for changed image offsets
When taking longer time lapse movies, the organoid can slowly slide out of the view. For this reason, the microscope user can move the imaged area so that the organoid stays in the view. In the time lapse movie, this makes the organoid appear to "teleport": from the edges of the image it jumps back to the center.

To create links over this "jump", the program would need to create a lot of large-distance links. The program will likely refuse to do this. To correct for this, *before running the linking process*, open the OrganoidTracker GUI and load the images and positions (`File` menu). Then edit the data (`Edit -> Manually change data...`) and edit the image offsets (`Edit -> Edit image offsets...`) of the correct time point. Instructions are on the bottom of the window.

Step 4: Manually correct warnings
---------------------------------

This step is not automated. 😉 Open the OrganoidTracker GUI and load the images and data (`File` menu). Then go to `Edit -> Manually change data...`. A cross will appear over all locations where the program has detected some inconsistency in the tracking data:

![Example of a warning](images/warning.png)

To view why OrganoidTracker warned you about this position, hover you mouse over a nucleus and press E. This will show you what the warning is. If the warning was a false positive, just press Delete to delete the warning. Otherwise, exit the warning screen (press E or Escape). This takes you back to the cell track editor, where you can fix the error.

Data editing is done mainly using the Insert and Delete keys, which are used to insert and delete links. Detected positions can also be inserted and deleted if necessary. Because correcting data works exactly the same as manual tracking, please see the [manual tracking](MANUAL_TRACKING.md) tutorial for more information.

If you press Left and Right in the warning screen, you will be taken to other warnings in the same lineage. If you press Up or Down, you will be taken to the warnings of other lineage trees.

If you press the L key from the cell track editor, a screen is opened that shows whether there are still warnings remaining in the lineage of that cell. If you hover your mouse over such a lineage and press E, the program will take you to the (first) warning in that lineage tree.

If you don't want to correct everything, in the `Edit` menu there's an option to delete all lineage trees that still have errors. Make sure you have backed-up your data before you do this. You can also delete all tracking data in certain parts of the images at certain times, see [batch editing](BATCH_OPERATIONS.md) for details.

You can change a few settings of the error checker, to make it stricter or less strict. In the error checking screen (the screen you opened with `E`) there are three options available in the `Edit` menu: the minimum time in between two cell divisions of the same cell, and the maximum distance a cell may move per minute.
