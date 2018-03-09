import os
import sys

# Make files in manual_tracking discoverable
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_folder = os.path.join(script_dir, "manual_tracking")
sys.path.append(modules_folder)

# Data to analyse
data_name = 'multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08'
pref = 'nd799xy08' # Prefix for image files

# Selected cell
curr_tr = 0

# Initial and final time points
t0 = 1
tf = 5000


t_zeros = 3
data_dir = os.path.join(script_dir, '../Images/' + data_name + '/')
save_dir = os.path.join(script_dir, '../Results/' + data_name + '/Manual tracks/')
exec(open(os.path.join(script_dir, 'manual_tracking/tracks_dia.py')).read())
plt.show()