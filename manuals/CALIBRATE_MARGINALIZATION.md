# Calibrate Marginalization
[← Back to main page](index.md)

If your data differs substantially from the training data of the neural networks you might have reason to doubt the accuracy of the error rate predictions. In this case you can calibrate the error rates against a ground truth. The ground truth here would consist of manually corrected data.

Alternatively if you have trained new neural networks you have to calibrate them for the first time. In this case you can reuse your validation data (or even training data in our experience) as the ground truth. 

Calibration then proceeds as follows:
1. Run your division detection model on all your ground truth datasets.
2. Run your link detection model on the outputs.
3. Go to the `all experiments` tab and use `Tools` -> `Calibrate marginalization...` to get the marginalization script.
4. Run the script. The result will be saved in a `.json`.