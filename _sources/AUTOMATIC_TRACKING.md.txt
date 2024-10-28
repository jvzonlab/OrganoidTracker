# Automatic tracking
[← Back to main page](index.md)

The intended workflow is as follows:

1. Obtain nucleus positions (for now this is done using an external program)
2. Obtain division probabilities.
3. Obtain linking probabilities.
4. Link the cell detections at different time points together
5. Calculate error probabilities
6. Manually correct warnings
7. OR Filter and do automated analysis

The steps are described below.

Before you start, create a directory that will hold all your analysis data. You can create this directory inside the directory that holds your images.

Step 1: Obtaining nucleus positions
-----------------------------------
To get the best results it is key that the input data somewhat resembles the data being trained on. This can often simply be achieved with a little pre-processing ([See here for tips](./PREPROCESSING_TIPS.md)). 

Open the OrganoidTracker GUI, load your images and select `Tools` -> `Detect cells in images...`. Select the already trained neural network that you want to use. ([See here for how to train a model](./TRAINING_THE_NETWORK.md) or find links to pre-trained models at [our Github page](https://github.com/jvzonlab/OrganoidTracker).) You'll end up with a folder containing configuration files. Run the `organoid_tracker_predict_positions` script in that directory to put the neural network into action. (On Windows, double-click the BAT file. On Linux/macOs, run the SH file instead.) You'll end up with a file that contains the positions of the cells. Load it into the program to see how the neural network performed.

If you want to debug what the program is doing, open the `organoid_tracker.ini` file and give the `predictions_output_folder` setting a value, for example `test`. If you run the script again, it will create a folder with that name (`test` in this example) and place images there indicating how likely it is to find a cell there.

### Correcting for changed image offsets
When taking longer time lapse movies, the organoid can slowly slide out of the view. For this reason, the microscope user can move the imaged area so that the organoid stays in the view. In the time lapse movie, this makes the organoid appear to "teleport": from the edges of the image it jumps back to the center.

To create links over this "jump", the program would need to create a lot of large-distance links. The program will likely refuse to do this. To correct for this, *before running the linking process*, open the OrganoidTracker GUI and load the images and positions (`File` menu). Then edit the data (`Edit -> Manually change data...`) and edit the image offsets (`Edit -> Edit image offsets...`) of the correct time point. Instructions are on the bottom of the window.

Step 2: Obtaining division scores
---------------------------------

Open the OrganoidTracker GUI again and load the images and the positions from the previous step. Select `Tools` -> `Detect dividing cells...` and follow the steps. Run the `organoid_tracker_predict_divisions` script.After the script is finished, which should be after a few minutes, you can load the resulting data in OrganoidTracker.

If you double-click on a cell in the main window, you can view its data. Among that data, you should now see the division probability. Verify that this probability is what you would expect: high for cells that are about to divide, low otherwise.

Step 3: Obtaining link scores
-----------------------------

This step works the same as the above script, except that you now run `Tools` -> `Detect link likelihoods...`. This script will predict the scores for all links that OrganoidTracker considers to be possible. (That are links to the nearest position, plus links to positions at most two times as far as the nearest position.)

Step 4: Obtaining links
-----------------------

Open the OrganoidTracker GUI again and load the images and the output file from the previous step. Use `Tools` -> `Create links between time points...` and run the resulting `organoid_tracker_create_links` script.

In order for the tracking to work we need to give the algorithm a few more probabilities to work with, these can be adjusted in the `organoid_tracker.ini` file. The first is the probability of a cell disappearing. This is simply the false negative rate of the cell detection plus the death rate of cells in the system. The appearance probability is again given by false negative rate alone. Cells can of course also appear from outside the field of view, to account for this you can set the maximum distance from which a cell could move out of view in a single timepoint. The (dis)appearance probabilities are then adjusted automatically near the edges of the imaging volume. It is good to note that the exact values of these probabilities should not substantially change the results in our experience, so a reasonable estimate will do.

Now you can also set some parameters to clean up the data after tracking. It is advisable to at least remove very short tracks for clarity.

This step outputs both a file containing the tracks and a file in which all links are retained. The latter is important when calculating error rates.    

Step 5: Calculate error rates
---------------------------------

Now that we have the tracks we can compute error rates through marginalization. Use `Tools` -> `Compute marginalized error rates...` and run the resulting `organoid_tracker_marginalize` script.

In the `organoid_tracker.ini` file you might need to change the so-called 'temperature'. This accounts for the amount of shared information between the individual neural network predictions. If your data is similar to the data the neural networks are trained on you can use the temperature associated with them (1.5 for our intestinal organoid models). If you have trained your own models you have to calibrate the marginalization procedure to get a temperature ([See here for how to calibrate](./CALIBRATE_MARGINALIZATION.md)). It is good to note that this temperature is generally able to absorb any miscalibration of the neural network outputs as well. 

This step also gives you dataset where all low-confidence links are filtered out. The threshold can be set in the `organoid_tracker.ini` file.

Step 6: Manually correct warnings
---------------------------------

This step is not automated. 😉 Open the OrganoidTracker GUI and load the images and data (`File` menu). Then go to `Edit -> Manually change data...`. A cross will appear over all locations where the program has detected some inconsistency in the tracking data:

![Example of a warning](images/warning.png)

To view why OrganoidTracker warned you about this position, hover you mouse over a nucleus and press E. This will show you what the warning is. If the warning was a false positive, just press Delete to delete the warning. Otherwise, exit the warning screen (press E or Escape). This takes you back to the cell track editor, where you can fix the error.

Data editing is done mainly using the Insert and Delete keys, which are used to insert and delete links. Detected positions can also be inserted and deleted if necessary. Because correcting data works exactly the same as manual tracking, please see the [manual tracking](MANUAL_TRACKING.md) tutorial for more information.

If you press Left and Right in the warning screen, you will be taken to other warnings in the same lineage. If you press Up or Down, you will be taken to the warnings of other lineage trees.

If you press the L key from the cell track editor, a screen is opened that shows whether there are still warnings remaining in the lineage of that cell. If you hover your mouse over such a lineage and press E, the program will take you to the (first) warning in that lineage tree.

If you don't want to correct everything, there are several options:
* Limiting the number of time steps that you are looking at. In the warning screen, if you open the Edit menu, there's an option to set the minimum and maximum time point for correction.
* Limiting the area you are looking at. This is most easily done by pressing L in the cell track editor, and then observing any lineages that still have warnings. Correct any lineages that need to be corrected (press E while you hover your mouse over them). Once you are satisfied, you can then delete all lineages that still have errors in them. This is done from the cell track editor: press `Edit > Batch deletion > Delete lineages with errors`. See [batch editing](BATCH_OPERATIONS.md) for more ways to delete a lot of lineages at once.

You can change a few settings of the error checker, to make it stricter or less strict. In the error checking screen (the screen you opened with `E`) there are three options available in the `Edit` menu: the minimimum marginalized probability a link should have, the minimum time in between two cell divisions of the same cell, and the maximum distance a cell may move per minute (only useful before marginalization).

Often correcting mistakes below a certain threshold also fixes a few mistakes that were not flagged. The increase in tracking quality might therefore be more than naively expected. To check the resulting quality you can rerun the marginalization on your corrected data while flagging in the `organoid_tracker.ini` file that you have checked part or all of the errors.

## What to do if my results aren't good?
That's a difficult problem! You have a few options:

* Improve image quality. The images should be good enough that it is straightforward to track the nuclei by hand in the large majority of cases.
* Change some settings. The generated `organoid_tracker.ini` files contain an explanation for each of the settings.
* [Retrain the neural networks for your data.](TRAINING_THE_NETWORK.md)
* [Replace parts of the tracker with custom code](CUSTOM_TRACKING_SCRIPTS.md)

Step 7: Automated analysis
---------------------------------

In step 5 a set of filtered high-confidence tracks is produced. You can choose to analyze these further. Generally for many application, manual corrections are not needed. 
* If you want to quantify a fluorescence reporter you can take long tracks (`Edit` -> `Manually change data` -> `Batch deletion` -> `Delete short lineages`) and work from there.
* If you want to quantify tissue flows flawless trajectories are often overkill. Setting a lower probability threshold should still give you good enough data. 
* Analyzing cell cycle dynamics should be done using survival analysis ([See here for how to do this](./SURVIVAL_ANALYSIS.md)) anyway and this can deal with cells that are lost to follow up. So it is not a problem to use the uncorrected filtered data.