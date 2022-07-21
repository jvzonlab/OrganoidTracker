# Jupyter notebook
[← Back to main page](index.md)

Instead of [writing a plugin](PLUGIN_TUTORIAL.md), it's also possible to work with Jupyter Notebooks. Jupyter Notebooks offer a way to mix code and data visualization. While there are [good arguments against doing this](https://datapastry.com/blog/why-i-dont-use-jupyter-notebooks-and-you-shouldnt-either/), for sure it can offer a rapid way to analyze data.

First, we need to install Jupyter Notebook. First, make sure that you have installed OrganoidTracker. Next, open up the Anaconda Prompt, navigate to the directory of OrganoidTracker and type `activate organoid_tracker`. If you haven't installed OrganoidTracker, or forgot how to do these steps, please refer back to [the installation instructions](INSTALLATION.md).

Next, still in the Anaconda Prompt, type `conda install -c conda-forge notebook` to install Jupyter Notebook. Once it's installed, run `jupyter notebook`. This will open a web browser that shows a list of files. You can start by trying the file `Example Jupyter notebook.ipynb`. You should see a screen like this:

![Screenshot of Jupyter notebook](images/jupyter.png)

The next time you want to start Jupyter Notebook, follow the same steps, but skip the step were you do `conda install -c conda-forge notebook`, as you have already installed Jupyter Notebook.

