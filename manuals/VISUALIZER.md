Autotrack manual
================

Installation
------------
First, install Anaconda or Miniconda. Then, open an Anaconda Prompt and run the following commands:

    conda env create -f environment.yml
    activate autotrack

(On macOs or Linux, run `source activate autotrack` instead of `activate autotrack`.)

Then, run `python autotrack.py` to start the program.

Navigating around
-----------------
The program consists of a menu bar, a title area, a figure area and a status/help area. The contents of all of these depends on what you're doing. If you enter the cell detection view, then methods relevant to cell detection will be shown. You can always go back to the main view using `View -> Exit this view`.

The program is highly keyboard-dependent, although there are also menu options available for everything. I prefer using the keyboard shortcuts myself, but for people new to this program that is of course no option.

For most keyboard shortcuts, a menu option is available. For shortcuts where that wasn't possible (for example because you need to place your mouse above a cell), an explanation text was added to the status/help area.

Many shortcuts consist of a single letter: if you press `E`. the errors and warnings screen is opened. For some other shortcuts, you need to type a bit more: `/deaths` + `ENTER` for example opens up a list of all cell deaths.

The best way to get used to the program is to just try things. As long as you do not overwrite existing data on disk, nothing can go wrong.

The editor
----------
If you press `C` from the main or errors screen, you can make changes to the data. In the editor, you can select up to two cells at once by double-clicking them. Using the Insert, Shift and Delete keys, you can insert, shift or delete cells or links. Press `C` again to exit the view.

If you press the Insert key while having two cells selected, a link will be inserted between them. If you press the Insert key while having no cells selected, a new cell position will be added at your mouse position. If you press the Insert key while having only one cell selected, a new cell will be inserted with a link to the selected cell.

If you press the Delete key while having two cells selected, the link between those cells will be deleted. If you press Delete while having only a single cell selected, that cell will be delted.

If you press Shift while having a single cell selected, that cell will be moved to your mouse psotion. The shape and links of the cell will be preserved.

Undo and Redo functions are available from the Edit menu. You can also use the Control+Z and Control+Y keyboard shortcuts, respectively.

Plugin support
--------------

Any Python file you place in the `autotrack_plugins` folder that has a name starting with `plugin_` (so for example `plugin_extra_images.py`) will automatically be loaded. A very minimal plugin looks like this:

    from autotrack.gui import dialog

    def get_menu_items(window):
        return {
            "Tools//Messages-Show Hello World...":
                lambda: dialog.popup_message("My Title", "Hello World!"),
            "Tools//Messages-Show other message...":
                lambda: dialog.popup_message("My Title", "Nice weather, isn't it?")
        }

The `get_menu_items` function is automatically called. It is used here to add some custom menu options. The options are shown in the "Tools" menu, in the "Messages" category. (The name of the category is never shown, but menu options in the same category will always appear next to each other.)
