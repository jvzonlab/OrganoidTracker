The visualizer
==============

The visualizer is used in many scripts to visualize data. It can also be used to make modifications to the data.

The image viewer
----------------
The arrow keys allow you to switch to other photos. The left and right keys move backward and forward in time,
respectively. The up and down keys move up and down in the z-direction, respectively.

If you need to jump to a time point far away, it is best to use a command. Type `/f30` and then `ENTER`  to go to frame
number 30.

Press `T` to view the track of a single particle. You will end up in a separate view, where you can see the particle
trajectory 10 points forward or backwards in time. Double-click on any particle to focus on that particle instead.
Press `T` again to go back to the normal view.

Press `M` to jump to the mother view. Here you get an overview of all cell divisions found during the experiment. The
left and right arrow keys can now be used to jump to the next or previous cell division. Press `M` again to exit the
mother view and return to the normal view.

Press `E` to go the errors and warnings view. All errors and warnings reported during the analysis are shown here. Use
the left and right arrow keys to jump to the next or previous error. After you have fixed an error (for example by
changing the linking data, see next section), or after you have made sure that the error was in error, press `DELETE` to
delete an error. Press `E` again to leave this view, and go back to the normal view.

The link editor
---------------
Links are the connections between cells at different time frames.

Press `L` to jump to an editor for the particle links. Here you need to select two cells. Cells can be selected by
double-clicking them. Then press `INSERT` to insert a link between cells, or `DELETE` to delete a link between two
cells.

Dotted lines show the current links, while straight lines show the situation from before you made changes.
Once you are happy with your changes, type `/commit` + `ENTER`. This will update the straight lines to match the
positions of the dotted lines. Type `/export <file name.json>` to export your changes to a file.

If you are not happy with your changes, type `/revert`. This will remove any uncommitted changes, and bring you back to
the state indicated by the straight lines. If you have already committed your changes, the only option is to close the
application without saving.

Press `L` again to exit the link editor and go back to the image viewer. Note that all committed changes will
immediately show up in all other views, like the cell division viewer and the track viewer.
