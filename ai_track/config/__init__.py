from configparser import RawConfigParser
import os.path
from typing import Tuple, Callable, Any

from ai_track.core import typing


def config_type_str(input: str) -> str:
    """Default type of settings in a configuration file."""
    return input


def config_type_int(input: str) -> int:
    """Parses values as integers."""
    return int(input)


def config_type_image_shape(input: str) -> Tuple[int, int, int]:
    """Parses a string like "512, 512, 32" as the tuple (32, 512, 512). Note the inversion of order."""
    split = input.split(",")
    if len(split) != 3:
        raise ValueError(f"Expected 3 coords, got {len(split)}")
    return int(split[2]), int(split[1]), int(split[0])


def config_type_bool(input: str) -> bool:
    input = input.lower()
    if input in ["true", "yes", "y", "t", "1"]:
        return True
    if input in ["false", "no", "n", "f", "0"]:
        return False
    raise ValueError("Expected \"True\" or \"False\", got \"" +input+"\"")


class ConfigFile:
    """Simple wrapper around ConfigParser"""

    FILE_NAME = "ai_track.ini"

    _folder_name: str
    _config: RawConfigParser
    _section_name: str
    made_changes: bool = False  # Changes to True when the configuration file was changed.

    def __init__(self, section_name: str, *, folder_name: str = "./"):
        """Creates a new configuration. Loads from disk using the standard file name. section_name is the section of
        the config file that we're currently looking at. (Different scripts should store their settings in
        different sections.)
        """
        self._section_name = section_name
        self._folder_name = folder_name

        self._config = RawConfigParser(allow_no_value=True)
        self._config.optionxform = str  # Prevent parser from converting everything to lowercase
        self._config.read(os.path.join(folder_name, self.FILE_NAME), encoding="UTF-8")

        # Create some sections
        if "DEFAULTS" not in self._config:
            self._config["DEFAULTS"] = {}
            self.made_changes = True
        if section_name not in self._config:
            self._config[section_name] = {}
            self.made_changes = True

    def get_or_default(self, key: str, default_value: str, *, comment: str = "", store_in_defaults: bool = False,
                       type: Callable[[str], Any] = config_type_str) -> Any:
        """Gets a string from the config file for the given key. If no such string exists, the default value is
        stored in the config first. If store_in_defaults is True, then the default value is stored in the DEFAULTS
        section, otherwise the default value is stored in the section given to __init__.
        """
        if key in self._config[self._section_name]:
            try:
                return type(self._config[self._section_name][key])
            except ValueError as e:
                print(f"Invalid value for setting {key}: {self._config[self._section_name][key]}")
        if key in self._config["DEFAULTS"]:
            try:
                return type(self._config["DEFAULTS"][key])
            except ValueError:
                print(f"Invalid value for setting {key} in default section of config: {self._config['DEFAULTS'][key]}")

        if store_in_defaults:
            if comment:
                self._config["DEFAULTS"]["; " + comment] = None
            self._config["DEFAULTS"][key] = default_value
        else:
            if comment:
                self._config[self._section_name]["; " + comment] = None
            self._config[self._section_name][key] = default_value
        self.made_changes = True
        return type(default_value)

    def get_or_prompt(self, key: str, question: str, store_in_defaults: bool = False, *,type: Callable[[str], Any] = config_type_str
                      ) -> Any:
        """Gets a string from the config file for the given key. If no such string exists, the user is asked for a
        default value, which is then stored in the configuration file. If store_in_defaults is True, then the default
        value is stored in the DEFAULTS section, otherwise the default value is stored in the section given to __init__.
        """
        if key in self._config[self._section_name]:
            try:
                return type(self._config[self._section_name][key])
            except ValueError as e:
                print(f"Invalid value for setting {key}: {self._config[self._section_name][key]}")
        if key in self._config["DEFAULTS"]:
            try:
                return type(self._config["DEFAULTS"][key])
            except ValueError:
                print(f"Invalid value for setting {key} in default section of config: {self._config['DEFAULTS'][key]}")

        try:
            default_value = input(question + " ")
        except EOFError:
            # Happens when the user presses CTRL + C.
            exit(200)
            return  # To keep Python linters from complaining that default_value may not be initialized on the next line
        return self.get_or_default(key, default_value, store_in_defaults=store_in_defaults, comment=question)

    def save_if_changed(self) -> bool:
        """Saves the configuration file if there are any changes made. Returns True if the config file was saved,
        False if no save was necessary."""
        if not self.made_changes:
            return False
        os.makedirs(self._folder_name, exist_ok=True)
        with open(os.path.join(self._folder_name, self.FILE_NAME), 'w', encoding="UTF-8") as config_writing:
            self._config.write(config_writing)
        self.made_changes = False
        return True

    def save_and_exit_if_changed(self):
        """If the configuration file was changed, the changes are stored an then the program is exited. This allows the
        user to see what changes were made, and then restart the program."""
        if not self.save_if_changed():
            return
        print("Configuration file was updated automatically. Please review the settings in the"
              " [" + self._section_name + "] section to check if they are correct, then rerun this command.")
        exit(301)
