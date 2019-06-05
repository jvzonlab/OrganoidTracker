# Autotrack manual

* [Installation instructions](INSTALLATION.md)
* [Scripts reference](SCRIPTS.md)
* [Programming API](API.md)
* [Tutorial for semi-automated tracking](AUTOMATIC_TRACKING.md)

Welcome to the Autotrack manual! This document will tell you how to use the "visualizer", the program that displays all the images and the annotations.

Navigating around
-----------------
The program consists of a menu bar, a title area, a figure area and a status/help area. The contents of all of these depends on what you're doing. If you enter the cell detection view, then methods relevant to cell detection will be shown. You can always go back to the main view using `View -> Exit this view`.

The program is highly keyboard-dependent, although there are also menu options available for everything. I prefer using the keyboard shortcuts myself, but for people new to this program that is of course no option.

For most keyboard shortcuts, a menu option is available. For shortcuts where that wasn't possible (for example because you need to place your mouse above a cell), an explanation text was added to the status/help area.

Many shortcuts consist of a single letter: if you press `E`. the errors and warnings screen is opened. For some other shortcuts, you need to type a bit more: `/deaths` + `ENTER` for example opens up a list of all cell deaths.

The best way to get used to the program is to just try things. As long as you do not overwrite existing data on disk, nothing can go wrong.

The links and positions editor
----------
If you press `C` from the main screen, you can make changes to the data. In the editor, you can select up to two cells at once by double-clicking them. Using the Insert, Shift and Delete keys, you can insert, shift or delete cells or links. Press `C` again to exit the view.

If you press the Insert key while having two cells selected, a link will be inserted between them. If you press the Insert key while having no cells selected, a new cell position will be added at your mouse position. If you press the Insert key while having only one cell selected, a link will be inserted from the currently selected cell to the position where your mouse is. If there is no cell position at your mouse, a new one will be created.

If you press the Delete key while having two cells selected, the link between those cells will be deleted. If you press Delete while having only a single cell selected, that cell will be delted.

If you press Shift while having a single cell selected, that cell will be moved to your mouse psotion. The shape and links of the cell will be preserved.

Undo and Redo functions are available from the Edit menu. You can also use the Control+Z and Control+Y keyboard shortcuts, respectively.

The data axes editor
--------------------
Say you want to know how far along a particle is along an axis. This custom axis does not align with the x, y or z axis. For example, you are looking at cell migration from the intestinal crypt to the intestinal villus. Then you want to draw an axis from the crypt to the villus, and see how far along the cells are over time.

For this, you first need to draw the data axis. This is a manual process. Open the data editor (in the `Edit` menu, or press `C` in the main screen) and then the axis editor (again in the `Edit` manu, or alternatively press `A`).

You need to draw axis from the lowest point to the highest point. Hover you mouse at the start of the axis (the zero point) and press Insert. A marker will be added. Then move your mouse to another point and press Insert again to insert a line to this point from the previuos marker. If your axis is not a straight line, you can add more points and a spline will be drawn using those points.

You can add a second (or third, fourth, etc.) axis by deselecting the first axis (double-click) and then pressing Insert without having an axis selected. Every particle will be assigned to the axis that was nearest in the first time point.

For the next time point, you can either draw the axis again, or (if the particles haven't moved too much) you can simply press C to copy the selected axis from another time point over to this time point. You can repeat this until the whole experiment is analyzed.

If you select an axis and then press Delete, the whole axis will be deleted.

Plugin support
--------------

Any Python file you place in the `autotrack_plugins` folder that has a name starting with `plugin_` (so for example `plugin_extra_images.py`) will automatically be loaded. A very minimal plugin looks like this:

```python
from autotrack.gui import dialog

def get_menu_items(window):
    return {
        "Tools//Messages-Show Hello World... [Ctrl+W]":
            lambda: dialog.popup_message("My Title", "Hello World!"),
        "Tools//Messages-Show other message...":
            lambda: dialog.popup_message("My Title", "Nice weather, isn't it?")
    }
```

The `get_menu_items` function is automatically called. It is used here to add some custom menu options. The options are shown in the "Tools" menu, in the "Messages" category. (The name of the category is never shown, but menu options in the same category will always appear next to each other.) Ctrl+W is the shortcut for the menu option.

You can call any method in the Python standard library, the Autotrack API (see API.md) and its dependencies. There is no sandbox implemented. You can access the currently loaded experiment using `window.get_experiment()`. o show a matplotlib figure, use `dialog.popup_figure(..)`.
