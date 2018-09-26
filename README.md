Tracking code
=============

Code for tracking the positions of cells. Tracking consists of two tasks: identification and linking. Identification
is the process where the positions of cells are determined in single images, while linking is the process where it is
determined wat cell at time point t corresponds to wat cell(s) in time point t + 1.


Setting up the environment
--------------------------
To be able to run the scripts, you first need to have Anaconda or Miniconda installed. Then, open an Anaconda Prompt and use `cd` to navigate to this directory. If you don't need Anaconda to run other Python scripts, just run the following command:

    conda env update -n root -f environment.yml

This installs all the required Python packages, and uninstalls all others. This makes it impossible to run most other Python scripts. So instead, it is probably better to create a separate environment for Autotrack. Instead of running the above command, run the following two commands:

    conda env create -f environment.yml
    activate autotrack

(On macOs or Linux, run `source activate` instead of `activate`.)


Running the scripts
-------------------

You can always run a script using the Python command from an Anaconda terminal:

    python <path/to/script_name>.py

Or, if you are fine with using your default Python 3 installation of your system, you can run the script as follows:

    <path/to/script_name>.py

On Windows, replace ".py" by ".bat" or simply leave out the extension altogether: (".bat" is one of the default file
extensions). If you don't want to always type out the complete path to the script, then you need to add the Autotrack folder to your
system's PATH.

The documentation of the scripts is found on the first few lines. See also the [manuals] folder for some additional
documentation.

[manuals]: manuals/MAIN.md
