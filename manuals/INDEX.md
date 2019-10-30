# AI_track manual

Welcome to the AI_track manual! This document will tell you how to use the "visualizer", the program that displays all the images and the annotations.

Tutorials
---------
* [Tutorial for manual tracking and error correction](MANUAL_TRACKING.md)
* [Tutorial for semi-automated tracking](AUTOMATIC_TRACKING.md)
* [Tutorial for writing a plugin for AI_track](PLUGIN_TUTORIAL.md)
* [Tutorial for training a neural network](TRAINING_THE_NETWORK.md)


For reference
-------------
* [Installation instructions](INSTALLATION.md)
* [Supported image formats](IMAGE_FORMATS.md) / [Supported tracking formats](TRACKING_FORMATS.md)
* [Scripts reference](SCRIPTS.md)
* [Programming API](API.md)
* [Batch editing](BATCH_EDITING.md)

Getting started
---------------
The program always displays your images in the center. Using the button on the menu bar, or using `File` -> `Load images...` you can load some images. You can load tracking data on top of that, or alternatively you can manually track the cells. The graphical program *cannot* automatically track cells, for this you need to use the other scripts. However, the program can generate configuration files for you, so that you don't need to spend too much time on the command line. 🙂

To load the tracking data, use the button on the toolbar, or use `File` -> `Load tracking data...`. If your tracking data contains links between the time points, then `Graph` -> `Interactive lineage tree...` will show a lineage tree of your tracking data.

![Toolbar](images/toolbar.png)  
The toolbar of the program.

Now would be a good moment to verify that you can actually save the tracking data; do so using `File` -> `Save tracking data...` or using the button on the toolbar. 

Highlights
----------

It's best to start with [manual tracking](MANUAL_TRACKING.md), to make sure that you understand the program. After tracking a few cells, you can start with [automated tracking](AUTOMATIC_TRACKING.md).

For data analysis, you can write your own Python scripts that make use of the [AI_track API](API.md). This is useful, as then you don't need to write your own functions to find dividing cells, dead cells, etc. You can even extend the graphical user interface by [writing plugins](PLUGIN_TUTORIAL.md). For that, you just need to place a Python file in the correct directory. You can use plugins to add additional cell types and menu options.

Want to measure the migration of a cell? Yuo can of course track the x, y and z position of a cell, but you can also [draw a (curved) axis yourself](DATA_AXES.md) and measure the position of a cell along that axis.

