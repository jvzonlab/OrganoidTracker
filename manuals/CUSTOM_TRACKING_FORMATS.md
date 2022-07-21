# Custom tracking formats
[← Back to main page](index.md)

Sometimes you need to import a file format for which OrganoidTracker has no support. In that case, you'll need to write a script yourself that converts the file format.

In this example, we are going to import ImageJ tracks. This example uses the [OrganoidTracker API](API.md) to load the file.

```python
import pandas

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import io

file_name = r"C:\path\to\file\of\imagej\input.csv"
output_file_name = r"C:\path\to\output.aut"

experiment = Experiment()  # Creates an empty experiment
data_frame = pandas.read_csv(file_name)  # Reads the CSV file (requires Pandas)

# Iterate through the rows
position_previous = None
for x, y, z, t in zip(data_frame.X, data_frame.Y, data_frame.Slice, data_frame.Frame):
    # Create a position, add it to the experiment
    position = Position(x, y, z - 1, time_point_number=t - 1)
    experiment.positions.add(position)

    # Connect it to the previous position
    if position_previous is not None:
        if position.time_point_number() - position_previous.time_point_number() > 1:
            break
        experiment.links.add_link(position, position_previous)

    # Update the previous position
    position_previous = position

# Save the result
io.save_data_to_json(experiment, output_file_name)
```

You can also create a plugin out of this example code, see the [plugin development tutorial](PLUGIN_TUTORIAL.md) for details on how to develop a plugin. That page contains information on how to add a custom menu option to import your file format.
