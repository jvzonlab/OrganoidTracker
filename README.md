OrganoidTracker old model conversion
====================================
 
Branch specifically for the purpose of converting old Tensorflow models to the new Keras 3 format.

To convert:

1. Set up an OrganoidTracker Tensorflow 2.7 environment following the installation instructions of
   OrganoidTracker2-tensorflow. You probably already have such an environment if you're reading this.
2. Copy `dump_tensorflow_model.py` to that OrganoidTraker folder (where `organoid_tracker.py` is) and run it to save 
   the Tensorflow model weights in Numpy format. You need to run it for each of the three models
   (positions, links and divisions).
3. Set up a Keras 3 environment following the installation instructions of the Pytorch-keras3 version of OrganoidTracker.
4. Run `convert_tensorflow_model_to_keras3.py` to load the Numpy weights and save them in Keras 3 format. You might need
   to adjust the model architecture in the codebase if it doesn't match the Tensorflow model exactly. (If that is the
   case, you'll get errors about missing weight files.)

That should be it! Feel free to contact the authors of OrganoidTracker if you run into issues.
