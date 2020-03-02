# Supported tracking formats
[← Back to main page](INDEX.md)

*See also: [Supported image formats](IMAGE_FORMATS.md)*

The program itself uses `.aut` files. This is the only file format that saves all data, including positions, position shape, errors and warnings, data axes, etc. You can also import/export from other formats, but beware that data will be lost in the conversion process.

* The Cell Tracking Challenge track format is described at [https://celltrackingchallenge.net/](https://celltrackingchallenge.net). You can export your data using the `File -> Export links` menu option, and you can import your data simply by loading the `man_track.txt` file. OrganoidTracker only exports the positions and links. Cell merges (like in the skin) are not supported, as this file format only allows one single parent for each cell track.
* The Paraview CSV format just consists of a bunch of CSV files. Use `File -> Export positions` to export the positions with or without extra data like cell density and track id. You'll be asked to select a folder to export everything to. After exporting you'll find a help file in that folder with instructions on how to import your files in Paraview. If you choose to export the positions with metadata, you'll also get a JSON file with a colormap, so that you can color the cells by lineage in Paraview.

If you're writing a Python script for data analysis, I recommend looking at the [API](API.md) to see how you can interact with the data files. If you're using another programming language, you can load the `.aut` file as a JSON file.
