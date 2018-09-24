Autotrack manual
================

Installation
------------
First, install Anaconda or Miniconda. Then, open an Anaconda Prompt and run the following commands:

    conda env create -f environment.yml
    activate autotrack

(On macOs or Linux, run `source activate` instead of `activate`.)

Then, run `python autotrack.py` to start the program.

Navigating around
-----------------
The program consists of a menu bar, a title area, a figure area and a status/help area. The contents of all of these
depends on what you're doing. If you enter the cell detection view, then methods relevant to cell detection will be
shown. You can always go back to the main view using `View -> Exit this view`.

The program is highly keyboard-dependent, although there are also menu options available for everything. I prefer using
the keyboard shortcuts myself, but for people new to this program that is of course no option.

For most keyboard shortcuts, a menu option is available. For shortcuts where that wasn't possible (for example because
you need to place your mouse above a cell), an explanation text was added to the status/help area.

Many shortcuts consist of a single letter: if you press `E`. the errors and warnings screen is opened. For some other
shortcuts, you need to type a bit more: `/deaths` + `ENTER` for example opens up a list of all cell deaths.

The best way to get used to the program is to just try things. As long as you do not overwrite existing data, nothing
can go wrong.

Plugin support
--------------

Any Python file you place in the `autotrack_plugins` folder that has a name starting with `plugin_` (so for example
`plugin_extra_images.py`) will automatically be loaded. A very minimal plugin looks like this:

    from autotrack.gui import dialog

    def get_menu_items(window):
        return {
            "Tools/Messages-Show Hello World...":
                lambda: dialog.popup_message("My Title", "Hello World!"),
            "Tools/Messages-Show other message...":
                lambda: dialog.popup_message("My Title", "Nice weather, isn't it?")
        }

The `get_menu_items` function is automatically called. It is used here to add some custom menu options. The options are
shown in the "Tools" menu, in the "Messages" category. (The name of the category is never shown, but menu options in the
same category will always appear next to each other.)
