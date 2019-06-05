# Installation instructions
[← Back to main page](INDEX.md)

To be able to run the scripts, you first need to have Anaconda or Miniconda installed. Then, open an Anaconda Prompt and use `cd` to navigate to this directory. Run the following two commands:

    conda env create -f environment.yml
    activate autotrack

(On macOs or Linux, run `source activate` instead of `activate`.)

The first command creates an Anaconda environment named "autotrack" with all dependencies installed. The second command activates this environment.

To test if the software is working, run `python autotrack.py`. A window should pop up, from which you can load images and tracking data.
