# Supported tracking formats
[← Back to main page](index.md)

*See also: [Supported image formats](IMAGE_FORMATS.md)*

The program itself uses `.aut` files. This is the only file format that saves all data, including positions, position shape, errors and warnings, data axes, etc. You can also import/export from other formats, but beware that data will be lost in the conversion process.

* The TrackMate format saves positions, links and the resolution. Other data is lost. Cell merges are included. Importing works by opening a TrackMate XML file, exporting can be done in the `File -> Export links` menu.
* The Cell Tracking Challenge track format is described at [https://celltrackingchallenge.net/](https://celltrackingchallenge.net). You can export your data using the `File -> Export links` menu option, and you can import your data simply by loading the `man_track.txt` file. OrganoidTracker only exports the positions and links. Cell merges (like in the skin) are not supported, as this file format only allows one single parent for each cell track.
* The Paraview CSV format just consists of a bunch of CSV files. Use `File -> Export positions` to export the positions with or without extra data like cell density and track id. You'll be asked to select a folder to export everything to. After exporting you'll find a help file in that folder with instructions on how to import your files in Paraview. If you choose to export the positions with metadata, you'll also get a JSON file with a colormap, so that you can color the cells by lineage in Paraview.
* For other formats you will need to write an importer yourself. Instructions can be found at the [custom tracking formats](CUSTOM_TRACKING_FORMATS.md) page.

If you're writing a Python script for data analysis, I recommend looking at the [API](API.md) to see how you can interact with the data files. If you're using another programming language, you can load the `.aut` file as a JSON file.


## Some details of AUT tracking format
AUT files are simply JSON files, and the format is mostly self-explanatory. However, for storing the position and links
two varations exist.

There is the older "v1" format, in use from 2018-2024. You can recognize these files by the `"version": "v1"` entry in
the root of the file. It stored positions and links, plus their metadata, in a kind of cumbersome and duplicative way:

```javascript
// Old format
"positions": {
 "1": [[56.4, 54.1, 4], ...],
 ...
},
"links": { // Links i nthe d3js graph format
    "directed": false,
    "multigraph": false,
    "graph": {},
    "nodes": [
      // All positions again, so that we follow the d3js format
      // Difference with above is that we now also list position metadata
      // Note how "_time_point_number" and the metadata keys are repeated over and over,
      // further bloating file size
      {"id": {"x": 136.92842303555807, "y": 186.91887053353733, "z": 12.0,
        "_time_point_number": 1, "blah": ...}},
      {"id": {"x": 157.02603447837174, "y": 184.34746761541615, "z": 12.0,
        "_time_point_number": 1, "blah": ...}}
      ...
    ],
    "links": [
       // All individual links
       // The save format doesn't know about tracks, so those are reconstructed when loading tracking data
       {
         "source": {"x": 132.0952578173535, "y": 249.18141069746628, "z": 12.0, "_time_point_number": 1},
         "target": {"x": 125.84057341732412, "y": 253.73027207930585, "z": 12.0, "_time_point_number": 2},
         "blah": ...  // Link metadata
       },
       ...
    ]
}
```

There is also the newer "v2" format, in use from 2025 onwards. You can recognize these files by the `"version": "v2"`
entry in the root of the file. It directly stores tracks, and also stores metadata in a more efficient way, without
repeating the metadata keys over and over:

```javascript
// New format
"positions": [
  { // Holds all positions and metadata for a single time point
    "time_point": 0,
    "coords_xyz_px": [[56.4, 54.1, 4], ...],
    "position_meta": { // Position meta now stored among the positions (instead of in the links graph)
       "blah": [...]  // One entry per position, null if not defined
    }
  },
  ...
]
"tracks": [
  { // Holds all positions and metadata for a single track

    // Coords of mother cell, in time point 17. Usually one entry, multiple in case of cell merges
    "coords_xyz_px_before": [[52.4, 51.1, 3]],  
    "time_point_start": 18,  // Time point of first position of track, 18 in this case
    // Coords of all positions, first for time point 18, then time point 19, etc.
    "coords_xyz_px": [[56.4, 54.1, 4], [58.4, 55.1, 4], ...],  
    "link_meta": { // Link metadata
       "blah": [...],
       ...
    },
    "link_meta_before": { // Link metadata for the links connecting this track to the previous track(s)
       "blah": [...],
       ...
    },
    
    // Metadata for the entire lineage (so also daughter tracks), only defined for tracks without a previous track
    "lineage_meta": { 
        "blah": ...  
    }
  },
  ...
],
```
