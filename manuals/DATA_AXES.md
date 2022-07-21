# The data axes editor
[← Back to main page](index.md)

Say you want to know how far along a particle is along an axis. This custom axis does not align with the x, y or z axis. For example, you are looking at cell migration from the intestinal crypt to the intestinal villus. Then you want to draw an axis from the crypt to the villus, and see how far along the cells are over time.

For this, you first need to draw the data axis. This is a manual process. Open the data editor (in the `Edit` menu, or press `C` in the main screen) and then the axis editor (again in the `Edit` manu, or alternatively press `A`).

You need to draw axis from the lowest point to the highest point. Hover you mouse at the start of the axis (the zero point) and press Insert. A marker will be added. Then move your mouse to another point and press Insert again to insert a line to this point from the previuos marker. If your axis is not a straight line, you can add more points and a spline will be drawn using those points.

You can add a second (or third, fourth, etc.) axis by deselecting the first axis (double-click) and then pressing Insert without having an axis selected. Every particle will be assigned to the axis that was nearest in the first time point.

For the next time point, you can either draw the axis again, or (if the particles haven't moved too much) you can simply press C to copy the selected axis from another time point over to this time point. You can repeat this until the whole experiment is analyzed.

If you select an axis and then press Delete, the whole axis will be deleted.
