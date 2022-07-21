# Supported image formats
[← Back to main page](index.md)

*See also: [Supported tracking formats](TRACKING_FORMATS.md)*

The program was made with movies of grayscale 3D images in mind. Multiple channels (for example captured at different wavelengths) are supported.

* There is some basic support for 2D images. They will be loaded as 3D images with only a single xy plane.
* If you try to load a colored image (like a PNG file), then the image will be converted to grayscale.
* The program is made for tracking cells. Loading a single image is therefore not useful. The program will, after displaying a warning, load the image as a "movie" with only one time point.

## LIF and ND2 files
LIF files are used by Leica microscopes while ND2 files are used by Nikon microscopes. Both file types contain images of all available time points in a single file. To load such a file, simply select `File` -> `Load images...` and select the file. If your file contains multiple time series, you will be prompted to pick one.

## A folder containing one 3D image for every time point
Many biologists save images like `image_0.tif`, `image_1.tif`, `image_2.tif` etc., with each image representing a single channel of a single time point. If you load an image **from time point 0 or 1**, the program will automatically try to search for other images with a similar naming pattern. The program is able to recognize the following naming patterns:

* Images **ending** with a number, like `image0.png`, `image1.png`, `image2.png`. The number will be interpreted as a time point number. (Brackets) around the number are allowed.
* Images **containing** the letter `t` followed by a number, like `nd56t1.tif`, `nd56t2.tif`. The number will be interpreted as a time point number. A captial `T` instead of a `t` is also allowed.
* Images **containing** the letter `c` followed by a number, like `nd56t1c1.tif`, `nd56t1c2.tif`. The number will now be interpreted as a channel. A capital `C` instead of a `c` is also allowed.

Leading zeroes (like in `image_0001.png`) are supported.

## A multi-page TIFF file
If instead only a single TIFF file is found, OrganoidTracker assumes that an the entire time-series is saved to that file. OrganoidTracker uses the `tifffile` Python package to read the metadata of the file, to found our how the file is constructed. If OrganoidTracker doesn't understand the file format, an error message is displayed.

## A single image
If the program cannot find any pattern in the file name, and the file is not a multi-page TIFF file, it will display a warning and load only that image. You'll end up with a "movie" of only one time point.
