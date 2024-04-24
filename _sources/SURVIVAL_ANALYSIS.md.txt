# Calibrate Marginalization
[← Back to main page](index.md)

This allows you to replicate the analysis from our most recent paper [link]. It allows you to extract lineage dynamics from uncorrected but filtered data in a statistically rigorous manner using survival analysis.


Step 1: Detect missed divisions
-----------------------------------
In survival analysis it is key that the chance of the event under study (in our case cell divisions) is independent from the chance of being lost to follow-up. During cell tracking cells are often lost just before division which violates this condition. To break the relationship between cell division and being lost, we need to look a the end of every track fragment and evaluate if it was in the process of dividing. 

We do this with a neural network that detects if the cell is in the division process, defined as three frames before and after the moment of chromosome separation. This network can be trained the same as the normal division detection network by setting `full_window` to `True` in the `organoidtracker.ini`file. 

The first step is thus to run the full window division network on the filtered data.


Step 2: Run analysis in GUI
-----------------------------------
You can now make the survival curves using use `Tools` -> `Cell cycle` -> `Survival curves ...`. If are in the `all_experiments` tab it will automatically make curves for all organoids.

Step 3: Export data
-----------------------------------
For more in depth analysis you can export the cell cycle lengths using `File` -> `Export cell cycle info`. If you want to customize the output you can adapt `plugin_cell_cycle_exporter.py`.