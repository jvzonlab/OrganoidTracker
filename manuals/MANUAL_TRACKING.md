# Manual tracking and error correction
[← Back to main page](INDEX.md)

You cannot edit tracking data from the main screen. Instead, you need to use `Edit` -> `Manually change data...` to open the data editor (or just press `C`). In this editor, you can select up to two cells at once by double-clicking them. Using the Insert, Shift and Delete keys, you can insert, shift or delete cells or links. Press `C` again to exit the view.

### Working with (cell) positions
To insert positions, make sure you have no cell selected, and then press the Insert key. To delete that cell position again, make sure that you have only selected that cell, and press the Delete key.

If you press Shift while having a single cell selected, that cell will be moved to your mouse psotion. The shape and links of the cell will be preserved.

Undo and Redo functions are available from the Edit menu. You can also use the Control+Z and Control+Y keyboard shortcuts, respectively.

### Working with links
To insert links between two positions, select two positions at two consecutive time points and press the Insert key. Select two positions and press Delete to delete the link between the two again.

A nice shortcut exists where if you have selected one single cell, then place your mouse cursor at the next or previous time point, and then press insert. The program will then both insert a position at your mouse cursor and create a link from the selected cell position to the newly added cell positions. This can be used to quickly track cells.

### Futher details
Links cannot skip time points. If you insert a link between two cells that are not in consecutive time points, then additions cell positions will be created in between the selected time points.

Links are used to tell the program that two cell positions at different time points actually represent the same biological cell. Therefore, you cannot create a link from one cell position in a time point to another cell position in the same time point. If you try anyways, a so-called *connection* will be made. You can write plugins to do something with those connections. You can also automatically create connections for all cells within a certain distance of each other, see `Edit` -> `Connect positions by distance...`