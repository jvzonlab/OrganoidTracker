"""Import this file to make Keras use PyTorch as backend, and to activate ANSI escape codes in Windows."""

import os

def activate():
    # To make Keras use PyTorch as backend
    os.environ["KERAS_BACKEND"] = "torch"

    # This activates ANSI escape codes in Windows
    # (this allows colors to be rendered, so that Keras progress bars are displayed correctly)
    if os.name == 'nt':
        from ctypes import windll

        kernel = windll.kernel32
        kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)

    # For loading models for a bit older Keras 3 versions
    import keras
    try:
        from keras.src import api_export
        api_export.REGISTERED_NAMES_TO_OBJS["keras.models.functional.Functional"] = keras.src.models.functional.Functional
        api_export.REGISTERED_NAMES_TO_OBJS["keras.ops.numpy.Concatenate"] = keras.src.ops.numpy.Concatenate
        api_export.REGISTERED_NAMES_TO_OBJS["keras.ops.numpy.Flip"] = keras.src.ops.numpy.Flip
        api_export.REGISTERED_NAMES_TO_OBJS["keras.ops.numpy.GetItem"] = keras.src.ops.numpy.GetItem
        api_export.REGISTERED_NAMES_TO_OBJS["keras.ops.numpy.Stack"] = keras.src.ops.numpy.Stack
        api_export.REGISTERED_NAMES_TO_OBJS["keras.ops.numpy.Absolute"] = keras.src.ops.numpy.Absolute
        api_export.REGISTERED_NAMES_TO_OBJS["keras.ops.nn.Conv"] = keras.src.ops.nn.Conv
        api_export.REGISTERED_NAMES_TO_OBJS["keras.backend.torch.optimizers.torch_adam.Adam"] = keras.src.optimizers.Adam
    except ModuleNotFoundError:
        pass  # Not necessary for this version of Keras