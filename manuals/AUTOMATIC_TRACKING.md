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

Open the AI_track GUI, load your images and select `Process` -> `Detect cells in images...`. Select the (already trained) neural network you want to use. You'll end up with a folder containing configuration files. Run the `ai_track_predict_positions` script in that directory to put the neural network into action. (On Windows, double-click the BAT file. On Linux/macOs, run the SH file instead.) You'll end up with a file that contains the positions of the cells. Load it into the program to see how the neural network performed.

If you want to debug what the program is doing, open the `ai_track.ini` file and give the `predictions_output_folder` setting a value, for example `test`. If you run the script again, it will create a folder with that name (`test` in this example) and place images there indicating how likely it is to find a cell there.

Step 2: Obtaining nucleus shapes
--------------------------------

Open the AI_track GUI again and load the images and the positions from the previous step. Select `Process` -> `Detect shapes using Gaussian fit...` and follow the steps. Run the `ai_track_detect_gaussian_shapes` script. Note: this script takes a few minutes to run per time point, so please be patient.

The script works by fitting 3D Gaussian functions to a blurred version of the original image, starting from the positions from step 1 and a default covariance matrix. To restrict the fitting algorithm, separate clusters of cells are fitted separately. In this way, the fitting algorithm will ignore debris elsewhere in the image. Clusters are found using a quick segmentation: every pixel that has an intensity lower than the local average is background, the rest is foreground (see figure 1). To improve the segmentation, some erosion can be applied, which makes the white areas in the figure 1 smaller. If your cells are already well separated, you can set it to 0, and if your cells form one big structure (like in figure 1) then a value of 3 will be necessary.

You can view the resulting Gaussians by loading the output file in the AI_track GUI and pressing `R`. This will draw the Gaussian functions.

![Thresholding](images/thresholding.png)  
*Figure 1: Thresholding, used to separate foreground and background. Gaussian fits are carried out within clumps.*

Step 3: Obtaining links
-----------------------

Open the AI_track GUI again and load the images and the output file from the previous step. Use `Process` -> `Create links between time points...` and run the resulting `ai_track_create_links` script.

First, the script will create some very basic initial links. Then it will calculate the likeliness (a score) of each cell being a dividing cell. Then it will create the actual links.

If you're not satisfied with the results, try changing the input parameters in the `ai_track.ini` file. There are comments in there to explain each setting. You can make cell divisions more/less likely, allow more or less cellular movement and make cell deaths more likely. If you don't see (m)any links anymore, then you have made link creation too expensive, and you should make events like cellular movement cheaper.

Step 4: Manually correct warnings
---------------------------------

This step is not automated. 😉 Open the AI_track GUI and load the images and data (`File` menu). Then go to `Edit -> Manually change data...`. Now you can use `View -> Lineage errors and warnings` to view which lineages have warnings in them. Lineages that are marked green are OK, lineages that are marked in gray need to be manually revisited.

To do this, hover you mouse over a nucleus and press E. You will be shown all errors in the lineage tree. Use the left and right arrow keys to move to the next error. Press E again to exit that view, and edit the data.

Data editing is done mainly using the Insert and Delete keys, which are used to insert and delete links. Detected positions can also be inserted and deleted if necessary.
