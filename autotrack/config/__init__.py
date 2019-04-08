from configparser import RawConfigParser


class ConfigFile:
    """Simple wrapper around ConfigParser"""

    FILE_NAME = "autotrack.ini"

    _config: RawConfigParser
    _section_name: str
    made_changes: bool = False  # Changes to True when the configuration file was changed.

    def __init__(self, section_name: str):
        """Creates a new configuration. Loads from disk using the standard file name. section_name is the section of
        the config file that we're currently looking at. (Different scripts should store their settings in
        different sections.)
        """
        self._section_name = section_name

        self._config = RawConfigParser()
        self._config.read(self.FILE_NAME, encoding="UTF-8")

        # Create some sections
        if "DEFAULTS" not in self._config:
            self._config["DEFAULTS"] = {}
            self.made_changes = True
        if section_name not in self._config:
            self._config[section_name] = {}
            self.made_changes = True

    def get_or_default(self, key: str, default_value: str, store_in_defaults: bool = False) -> str:
        """Gets a string from the config file for the given key. If no such string exists, the default value is
        stored in the config first. If store_in_defaults is True, then the default value is stored in the DEFAULTS
        section, otherwise the default value is stored in the section given to __init__.
        """
        if key in self._config[self._section_name]:
            return self._config[self._section_name][key]
        if key in self._config["DEFAULTS"]:
            return self._config["DEFAULTS"][key]

        if store_in_defaults:
            self._config["DEFAULTS"][key] = default_value
        else:
            self._config[self._section_name][key] = default_value
        self.made_changes = True
        return default_value

    def get_or_prompt(self, key: str, question: str, store_in_defaults: bool = False):
        """Gets a string from the config file for the given key. If no such string exists, the user is asked for a
        default value, which is then stored in the configuration file. If store_in_defaults is True, then the default
        value is stored in the DEFAULTS section, otherwise the default value is stored in the section given to __init__.
        """
        if key in self._config[self._section_name]:
            return self._config[self._section_name][key]
        if key in self._config["DEFAULTS"]:
            return self._config["DEFAULTS"][key]

        try:
            default_value = input(question + " ")
        except EOFError:
            # Happens when the user presses CTRL + C. Allow to try again
            exit(200)
            return  # To keep Python linters from complaining that default_value may not be initialized on the next line
        return self.get_or_default(key, default_value, store_in_defaults=store_in_defaults)

    def save_and_exit_if_changed(self):
        """Saves the configuration if it was changed. Returns True if the file was saved, False otherwise."""
        if not self.made_changes:
            return
        with open(self.FILE_NAME, 'w') as config_writing:
            self._config.write(config_writing)
        print("Configuration file was updated automatically. Please review the settings in the"
              " [" + self._section_name + "] section to check if they are correct, then rerun this command.")
        exit(301)
