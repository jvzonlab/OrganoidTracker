"""

Internal OrganoidTracker code to load plugins. Example usage:

>>> from organoid_tracker.plugin import plugin_loader
>>> plugin = plugin_loader.load_plugin("path/to/folder/plugin_file.py")
>>> print(plugin.get_commands())  # Prints the commands of the first plugin in the folder

"""
