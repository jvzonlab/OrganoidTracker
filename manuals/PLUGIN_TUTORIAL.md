# Plugin tutorial
[← Back to main page](INDEX.md)

## Introduction
Plugins (like the rest of OrganoidTracker) are programmed in Python, a "friendly" programming language used from beginners to experts. Every Python file that you place in the `organoid_tracker_plugins` that starts with `plugin_` is automatically loaded when the program starts. Plugins can be reloaded while the program is running, making very quick development possible.

In this tutorial, we're going to create a plugin that how many detected cells there are in each time point. No prior programming knowledge is necessary to read the code, but for writing your own code some knowledge of Python is necessary. If you haven't used Python before, it's best to follow a course or have a supervisor explaining things to you. Alternatively, there is the website [automatetheboringstuff.com](https://automatetheboringstuff.com/) with a good Python tutorial.

## First steps
Let's get started. Create a text file in your favorite plain text editor (Notepad will work, but I personally recommend Pycharm from Jetbrains) with the following contents: *(copy-paste it, and note that every missing space or colon (':') can make your plugin crash)*

```python
from organoid_tracker.gui import dialog

def get_menu_items(window):
    return {
        "My menu item//Count cells in time points...": lambda: dialog.popup_message("My plugin", "Not yet implemented")
    }
```

We'll walk through the code shortly. For now, save the file as `plugin_count_cells_in_time_points.py` in the `organoid_tracker_plugins` folder (there should already be quite a few other `plugin_` files there). Open the OrganoidTracker program. Notice that in the menu bar now contains a "My menu item" item. If you open that menu and click on the only option, a dialog window pops up.

On the first line of the script, we import the `dialog` module, so that we can use that module later on to show a dialog message. On the third line (starting with `def`), we define a function called `get_menu_items` that takes one parameter, `window`. That parameter refers to the currently open window, and you can use it to access the currently loaded project(s). We will show how do that that later, but for now we're actually not using the window yet, so you won't see the word `window` elsewhere in the script. 

The function returns a so-called dictionary. If you haven't seen these structures before in Python, you should look them up, but for now we can show how you would use them to create a small "real" dictionary:

```python
my_dictionary = {
    "apple": "Type of fruit",
    "biology": "Study of living systems",
    "boat": "Vehicle for water transporation",
    "cell": "Some kind of enclosed space."
}
```

For our menu, it looks a bit different. The "lookup words" are now the names of the menu option, and the "descriptions" are now the actions that should be taken when that menu option is clicked. 

Here, the value is `lambda: dialog.popup_message("My plugin", "Not yet implemented")`, which stores a piece of code (using `lambda: `) to show a popup message later, with the window title set to `"My plugin"` and the contents set to `"Not yet implemented"`.

That's basically the entire plugin for now. Let's create a second function to do the actual calculations.

## Counting the cells
Replace the text in the plugin file you have previously created with the following:

```python
from organoid_tracker.gui import dialog

def get_menu_items(window):
    return {
        "My menu item//Count cells in time points...": lambda: _count_cells(window)
    }

def _count_cells(window):
    experiment = window.get_experiment()
    cell_counts = list()
    for time_point in experiment.time_points():
        cells_of_this_time_point = experiment.positions.of_time_point(time_point)
        cell_count_for_this_time_point = len(cells_of_this_time_point)
        
        cell_counts.append(cell_count_for_this_time_point)
    dialog.popup_message("My plugin", "The cells counts are " + str(cell_counts))
```

Save the file and reload all plugins (see the `File` menu of OrganoidTracker), or restart the program. If you have some tracking data loaded, the code will now display the number of cells at each time point, for example `[5, 5, 5, 6, 7, 7, 7]`. The part under `for time_point in experiment.time_points():` is run for reach time point, and every time that part runs it appends one value (the amount of cells in that time point) to the list `cell_counts`.

## Displaying a graph
It would be better to display this list as a graph. Fortunately, this is easily done using the widely used Python library Matplotlib. OrganoidTracker has a function to show a Matplotlib figure as a popup window. At the bottom of your plugin file, add the following function:

```python
def _display_my_figure(figure, cell_counts):
    axis = figure.gca()  # gca is short for "get current axis"
    
    axis.plot(cell_counts)  # Plots the points
    axis.set_xlabel("Time point")
    axis.set_ylabel("Cell count")
    
    figure.suptitle("Cell count over time")  # Supplies a title
```

Finally, replace the line `dialog.popup_message("My plugin", "The cells counts are " + str(cell_counts))` with
```python
dialog.popup_figure(window.get_gui_experiment(), lambda figure: _display_my_figure(figure, cell_counts))
```
to actually display that figure. The function `popup_figure` first sets up a window and then uses the code `lambda figure: _display_my_figure(figure, cell_counts))` to draw something to the figure that was just created.

After saving the file and reloading the plugins, you should now end up with a graph.

Congratulations, you now know the basics of plugin writing!OrganoidTracker contains a large number of functions that help you writing a plugin. See the [API](API.md) page for a short introduction. The main code of OrganoidTracker is also contains a lot of comments, which should help you understanding what the code is doing. A "smart" code editor that automatically displays documentation (such as Pycharm or Visual Studio Code) also helps a lot. Try and see if you can understand how the plugin `plugin_count_dividng_cells.py` works.

## Plugins in a folder instead of a file
For larger plugins, placing all code in a single file becomes unwieldy. Instead, you can also put your code in multiple files placed in a single directory. The directory must have a name that starts with `plugin_`, for example `plugin_this_is_an_example`. In that folder, you must place an `__init__.py` file. This file should contain the hooks described above, such as the `get_menu_items` function. You can import other files using relative imports: use for example `from . import another_file`. This will import the file `another_file.py` that is placed next to the `__init__.py` file.
