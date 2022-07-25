"""

Internal OrganoidTracker code to load plugins. Example usage:

>>> from organoid_tracker.plugin import plugin_loader
>>> plugins = plugin_loader.load_plugins("path/to/folder")
>>> print(plugins[0].get_commands())  # Prints the commands of the first plugin in the folder

"""
