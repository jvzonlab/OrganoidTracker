# Preprocessing tips
[← Back to main page](index.md)

This page contains some preprocessing tips.  

Scaling your image
-----------------------------------
The neural networks in OrganoidTracker are all convolutional neural networks. This means first of all that they are not scale invariant. The best results are achieved when the nuclei have the same volume (in pixels) as the nuclei in the training data. For the models trained on intestinal organoid data available [on zenodo](https://zenodo.org/records/13946119), this means that nuclei should be around 25x25x4 (xyz) pixels. This match should be rather exact, within a 80% to 120% range. Scaling the data is easily achieved in for instance imageJ. Downscaling in Z can often introduce weird effects if not done by an integer factor, so try to avoid that.  

Removing your background
-----------------------------------
The cell detection network has to decide between what is foreground and what is background. So it helps a lot if the background is similar to the training data. In our training data the offset was set such that the background has largely a value of zero. This also seems to help during training. In order to give your data background values of zero, using a background subtraction algorithm followed by subtracting a baseline value is recommended. Blurring the images a bit to remove noise before this process can also help.

Tune contrast
-----------------------------------
In our training data the fluorescence intensities of the nuclei are rather similar (factor four difference between the bright ones and the dim ones). If the contrast is higher the data will be difficult to interpret and dim cells classified as background. This is easily solved by taking the square root of the intensity values (do not do this in uint8, but in uint16 or in 32bit). If that is not enough you can take another square root.
