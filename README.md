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

AT the moment, there are two types of scripts in this folder. The older ones need to be run like this:

    python <script_name>.py

They need to be run from this folder. You'll need to edit the scripts beforehand to correct the paths
inside them.

Newer scripts (they have a name starting with `autotrack`) are easier to run. You can run them from any
folder, and they don't need to be edited beforehand. However, you need to add this folder (the folder where
this README file is stored) to the system PATH first. After you have done that, on Unix systems (Mac/Linux)
you can run the scripts like this:

    <script_name>.py

For Windows, I created `.bat` files. As a result, you can run the scripts simply by running:

    <script_name>.bat

(You can even leave out the `.bat` extension if you want.) The first time you run any script, it will ask
for some parameters, which are then saved to a configuration file.
   
See the [manuals] folder for the actual documentation of the scripts.

[manuals]: manuals/MAIN.md
