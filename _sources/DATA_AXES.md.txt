# The data axes editor
[← Back to main page](index.md)

In biology, cells don't always move perfectly along the x, y or z axis. Still, you might want to track how far along a cell is along some custom axis. For example, you might be looking at cell migration from the intestinal crypt to the intestinal villus. In that case, you'll need to draw the axis yourself from the crypt bottom to the villus tip. Then, you can track how far along this axis the cells are at each time point. This is what the data axes editor is for.

In OrganoidTracker, data axis are implemented as splines. You need to draw them by hand. Open the data editor (in the `Edit` menu, or press `C` in the main screen) and then the axis editor (again in the `Edit` manu, or alternatively press `A`).

You need to draw axis from the starting point of your axis to the final point. Hover you mouse at the start of the axis (the zero point) and press Insert. A marker will be added. Then move your mouse to another point and press Insert again to insert a line to this point from the previous marker. You can continue adding points like this, and the program will draw a spline through all points.

For the next time point, you can either draw the axis again, or (if the cells haven't moved too much) you can simply press Insert to copy the selected axis from another time point over to this time point. You can repeat this until the whole experiment is analyzed.

If you select an axis and then press Delete, the whole axis will be deleted.

## Types of data axes
By default, there are just splines. However, by placing a tiny Python file in the plugins folder, you can create new types of data axes.

```python
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.spline import Spline

def get_markers() -> list[Marker]:
    return [
        Marker([Spline], "CRYPT", "crypt-villus axis", (255, 0, 0), is_axis=True),
    ]
```

Save this file as `plugin_SOME_NAME_HERE.py` to the plugins folder (see `File` -> `Install new plugin` in the main menu). Then, when you reload all plugins and open the data axes editor, you'll see that under `Edit` -> `Set type of spline`.

If `is_axis=True`, then the spline uses arrowheads to indicate the direction of the axis. Otherwise, it will just use dots.

The type of data axis only affect how it's displayed. It's up to you to distinguish between them in your downstream data analysis. You can use the name of the axis (in this case, "CRYPT") to do so.

## Spline checkpoints
Along the spline, you can highlight specific points. In the crypt-villus axis example, you might want to indicate the "neck", the transition area between the crypt and the villus. To do this, first select the spline by clicking on it. Then, click a second time on the spline at the point you want to highlight. That point is now selected. To insert a checkpoint there, press Insert or Enter.

Analogous to naming the spline types, you can also name the checkpoint types. The full plugin code would then become:

```python
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.spline import Spline, SplineCheckpoint

def get_markers() -> list[Marker]:
    return [
        Marker([Spline], "CRYPT", "crypt-villus axis", (255, 0, 0), is_axis=True),
        Marker([SplineCheckpoint], "CRYPT_NECK", "crypt neck", (255, 0, 0)),
    ]
```

When inserting a new checkpoint, you will be prompted which type it should be. When inserting another checkpoint, it will automatically be of the same type as the previously inserted checkpoint. You can change the type of a checkpoint by selecting it and then choosing a different type in the `Edit` menu.
