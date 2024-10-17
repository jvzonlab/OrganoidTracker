@rem Automatically generated script for running organoid_tracker
@rem Unfortunately organoid_tracker cannot deal with parameters in command line. 
@rem All parameters can be found in configuration_files/DatasetName/SequenceID/organoid_tracker.ini 
@rem See .yml file for all dependencies and prerequisites
@echo off
cd configuration_files\c_elegans\02
python "..\..\..\organoidtracker\organoid_tracker_predict_positions.py"
python "..\..\..\organoidtracker\organoid_tracker_predict_divisions.py"
python "..\..\..\organoidtracker\organoid_tracker_predict_links.py"
python "..\..\..\organoidtracker\organoid_tracker_create_links.py"
python "..\..\..\organoidtracker\organoid_tracker_ctc_exporter.py"
pause