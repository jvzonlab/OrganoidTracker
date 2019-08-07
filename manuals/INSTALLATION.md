Installation instructions
=========================
[← Back to main page](./INDEX.md)

To be able to run the scripts, you first need to have Anaconda or Miniconda installed. Then, open an Anaconda Prompt and use `cd` to navigate to this directory. Run the following two commands:

    conda env create -f environment.yml
    activate ai_track

More specifically about the current working enviroment, run the two commands:
    conda create --name ai_track --file spec-file-win64.txt
    activate ai_track

(On macOs or Linux, run `source activate` instead of `activate`.)

The first command creates an Anaconda environment named "ai_track" with all dependencies installed. The second command activates this environment.

If you have updated AI_track, and you want to update the dependencies, execute this command instead:

    conda env update -n ai_track -f environment.yml

If you need to remove the previous version of AI_track, execute this command:
    conda env remove -n ai_track

To test if the software is working, run `python ai_track.py`. A window should pop up, from which you can load images and tracking data.
