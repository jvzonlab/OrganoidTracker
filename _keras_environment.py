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
