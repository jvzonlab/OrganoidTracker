import _keras_environment

_keras_environment.activate()

import os

from typing import Optional


from keras import Model
import numpy

_MODEL_TYPE = "links"  # "positions", "links" or "divisions"
_INPUT_MODEL = r"E:\Scratch\Tensorflow conversion\model_" + _MODEL_TYPE + "_exported"  # From dump_keras3_model.py
_EXPORT_FILE = r"E:\Scratch\Tensorflow conversion\model_" + _MODEL_TYPE + "\model.keras"  # Where to save the converted model


def _find_matching_folder(layer_name: str) -> Optional[str]:
    for subfolder in os.listdir(_INPUT_MODEL):
        if layer_name in subfolder:
            return os.path.join(_INPUT_MODEL, subfolder)
    return None


def _create_empty_model(model_type: str) -> Model:
    batch_size = 1

    if model_type == "positions":
        input_shape = (32, None, None, 2)
        from organoid_tracker.neural_network.position_detection_cnn.convolutional_neural_network import build_model
        return build_model(input_shape, batch_size)
    elif model_type == "links":
        from organoid_tracker.neural_network.link_detection_cnn.convolutional_neural_network import build_model
        input_shape = (16, 64, 64, 2)
        return build_model(input_shape, batch_size)
    elif model_type == "divisions":
        from organoid_tracker.neural_network.division_detection_cnn.convolutional_neural_network import build_model
        input_shape = (12, 64, 64, 3)
        return build_model(input_shape, batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def main():
    model: Model = _create_empty_model(_MODEL_TYPE)
    print(f"Converting {_MODEL_TYPE.upper()} model...")
    errors = False

    # Export the model weights to a numpy file
    for i, layer in enumerate(model.layers):
        layer_name = layer.name.replace(">", "_")
        layer_folder = _find_matching_folder(layer_name)
        if layer_folder is None:
            if len(layer.get_weights()) > 0:
                # Only print a warning if we were actually expecting weights
                print(f"ERROR: Could not find matching folder for layer {i + 1}. {layer_name}")
                errors = True
            else:
                print(f"Skipping layer {i + 1}. {layer_name} (no weights)")
            continue

        new_weights = list()
        layer_errors = False
        for j, weights in enumerate(layer.get_weights()):
            shape_str = "x".join(str(dim) for dim in weights.shape)
            weights_file = os.path.join(layer_folder, f"{j + 1}.weights_{shape_str}.npy")
            if not os.path.isfile(weights_file):
                print(f"ERROR: Could not find weights file for layer {i + 1}. {layer_name}, missing file {os.path.basename(layer_folder)}/{os.path.basename(weights_file)}")
                layer_errors = True
                errors = True
                continue
            weights_loaded = numpy.load(weights_file)
            new_weights.append(weights_loaded)
        if not layer_errors:
            layer.set_weights(new_weights)
            print(f"Converted layer {i + 1}. {layer_name} from {os.path.basename(layer_folder)}")

    if errors:
        print("Errors occurred during conversion. Model not saved.")
        return
    model.save(_EXPORT_FILE)
    print(f"Saved converted model to {_EXPORT_FILE}")


def _remove_non_ascii(s: str) -> str:
    return "".join(c for c in s if ord(c) <= 126)


if __name__ == "__main__":
    main()