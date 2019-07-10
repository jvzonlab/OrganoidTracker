Tracking code
=============

Code for tracking the positions of cells. Tracking consists of two tasks: detection and linking. Detection is the process where the positions of cells are determined in single images, while linking is the process where it is determined wat cell at time point t corresponds to wat cell(s) in time point t + 1.

Installation
------------
AI_track must be installed using Anaconda. See the [installation] page for details. If you are updating from an older version of AI_track, its dependencies might have changed, and in that case you also need to visit the [installation] page.


Tracking tutorial
-----------------
After you have installed the software, please see the [tracking tutorial] to get started.

Visualizer
----------
The `ai_track.py` starts a graphical program that allows you to visualize and edit your data. See the documentation on the [visualizer] page.

Scripts
-------
Detection and linking must be done from the command line. See the [scripts] page for details.

API
---
You can also use AI_track as a library to write your own scripts. All public functions in AI_track have docstrings to explain what they are doing. As a starting point for using the API, see the [API] page.

[API]: manuals/API.md
[installation]: manuals/INSTALLATION.md
[scripts]: manuals/SCRIPTS.md
[tracking tutorial]: manuals/AUTOMATIC_TRACKING.md
[visualizer]: manuals/INDEX.md
