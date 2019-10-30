# Batch operations
[← Back to main page](INDEX.md)

This page describes the batch editing options available from the GUI. The positions and links editor, described on the [manual tracking page](MANUAL_TRACKING.md), is mainly intended to make one change at a time. However, there are a few options for large-scale data editing.

All the instructions below assume that you are in the manual data editing screen (`Edit` -> `Manually change data...`).

## Deleting all positions within a rectangle
Select `Edit` -> `Delete all positions within a rectangle`. Double-click somewhere to define one corner, double-click somewhere else to define the other corner. A rectangle will be drawn. Then use `Edit` -> `Delete all positions inside the rectangle` to delete those positions. Note that only positions at the layer where the rectangle is will be deleted.

The "rectangle" can span multiple z-layers, making it a three-dimensional volume. The rectangle can even span multiple time points, which allows for large-scale deletion of points. For example, you can drawn a rectangle from `x=50`, `y=40`, `z=4`, `t=2` all the way to `x=400`, `y=300`, `z=20`, `t=200`.

In the `Edit` menu, you can also choose to delete all positions *outside* the rectangle, instead of inside. This essentially keeps on the positions inside the rectanble. Note that this operation only deletes positions in the time points where you have drawn the rectangle; other time points are unaffected.

## Deleting all positions of a time point
One way would be to drawn a rectangle over all points (see above). Howeve,r there's an easier option: simply select `Edit` -> `Delete all data of time point`. Note that this also deletes other annotations, such as [splines](DATA_AXES.md).

## Deleting all positiosn of multiple time points
The best way to do this is to draw a single volume that goes over all time points and z layers. Select one corner of the image at the first time point, and the opposite corner at the last time point.
