import os

import numpy

from organoid_tracker.position_detection_cnn.loss_functions import loss, position_precision, position_recall, overcount

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow
from tensorflow.keras.models import Model

_MODEL_FOLDER = r"E:\Scratch\Tensorflow conversion\model_positions"
_EXPORT_FOLDER = _MODEL_FOLDER + "_exported"

def main():
    os.makedirs(_EXPORT_FOLDER, exist_ok=True)
    model = tensorflow.keras.models.load_model(_MODEL_FOLDER, custom_objects={"loss": loss,
                                                                      "position_precision": position_precision,
                                                                      "position_recall": position_recall,
                                                                      "overcount": overcount})

    # Export the model weights to a numpy file
    for i, layer in enumerate(model.layers):
        layer_name = layer.name.replace("/", "_")
        layer_folder = os.path.join(_EXPORT_FOLDER, f"{i + 1}. {layer_name}")
        os.makedirs(layer_folder, exist_ok=True)
        for j, weights in enumerate(layer.get_weights()):
            shape_str = "x".join(str(dim) for dim in weights.shape)
            numpy.save(os.path.join(layer_folder, f"{j + 1}.weights_{shape_str}.npy"), weights)

    # Save a fancy model summary
    with open(os.path.join(_EXPORT_FOLDER, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


if __name__ == "__main__":
    main()