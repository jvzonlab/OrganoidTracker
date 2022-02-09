from configparser import RawConfigParser
import os.path
from typing import Tuple, Callable, Any, Dict

from organoid_tracker.core import typing


def config_type_str(input: str) -> str:
    """Default type of settings in a configuration file."""
    if len(input) >= 2:
        if input.startswith("'") and input.endswith("'"):
            return input[1:-1]
        if input.startswith('"') and input.endswith('"'):
            return input[1:-1]
    return input


def config_type_json_file(input: str) -> str:
    """A string that will automatically have ".json" appended to it if it hasn't already (except for empty strings)."""
    input = config_type_str(input)  # First parse as string
    if len(input) == 0:
        return input
    if not input.lower().endswith(".json"):
        return input + ".json"
    return input


def config_type_csv_file(input: str) -> str:
    """A string that will automatically have ".csv" appended to it if it hasn't already (except for empty strings)."""
    input = config_type_str(input)  # First parse as string
    if len(input) == 0:
        return input
    if not input.lower().endswith(".csv"):
        return input + ".csv"
    return input


def config_type_int(input: str) -> int:
    """Parses values as integers."""
    return int(input)


def config_type_float(input: str) -> float:
    """Parses values as floats."""
    return float(input)


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


class _ValueWithComment:
    value: str
    comment: str

    def __init__(self, value: str, comment: str):
        self.value = value
        self.comment = comment

    def __repr__(self) -> str:
        return "_ValueWithComment(" + repr(self.value) + ", " + repr(self.comment) + ")"

class ConfigFile:
    """Simple wrapper around ConfigParser"""

    FILE_NAME = "organoid_tracker.ini"

    _folder_name: str
    _config: Dict[str, Dict[str, _ValueWithComment]]
    _section_name: str
    made_value_changes: bool = False  # Changes to True when the configuration file was changed.

    def __init__(self, section_name: str, *, folder_name: str = "./"):
        """Creates a new configuration. Loads from disk using the standard file name. section_name is the section of
        the config file that we're currently looking at. (Different scripts should store their settings in
        different sections.)
        """
        self._section_name = section_name
        self._folder_name = folder_name

        config_parser = RawConfigParser()
        config_parser.optionxform = str  # Prevent parser from converting everything to lowercase
        config_parser.read(os.path.join(folder_name, self.FILE_NAME), encoding="UTF-8")
        self._config = dict()
        for section in config_parser.sections():
            self._config[section] = dict((name, _ValueWithComment(value, "")) for name, value in config_parser[section].items())

        # Check if our section name exists
        if "DEFAULTS" not in self._config:
            self._config["DEFAULTS"] = {}
        if section_name not in self._config:
            self._config[section_name] = {}
            self.made_value_changes = True

    def get_or_default(self, key: str, default_value: str, *, comment: str = "", store_in_defaults: bool = False,
                       type: Callable[[str], Any] = config_type_str) -> Any:
        """Gets a string from the config file for the given key. If no such string exists, the default value is
        stored in the config first. If store_in_defaults is True, then the default value is stored in the DEFAULTS
        section, otherwise the default value is stored in the section given to __init__.
        """
        if key in self._config[self._section_name]:
            setting = self._config[self._section_name][key]
            setting.comment = comment
            try:
                return type(setting.value)
            except ValueError as e:
                print(f"Invalid value for setting {key}: {self._config[self._section_name][key]}")
        if key in self._config["DEFAULTS"]:
            setting = self._config["DEFAULTS"][key]
            setting.comment = comment
            try:
                return type(setting.value)
            except ValueError:
                print(f"Invalid value for setting {key} in default section of config: {self._config['DEFAULTS'][key]}")

        # Setting not found, create
        if store_in_defaults:
            self._config["DEFAULTS"][key] = _ValueWithComment(default_value, comment)
        else:
            self._config[self._section_name][key] = _ValueWithComment(default_value, comment)
        self.made_value_changes = True
        return type(default_value)

    def get_or_prompt(self, key: str, question: str, store_in_defaults: bool = False, *,type: Callable[[str], Any] = config_type_str
                      ) -> Any:
        """Gets a string from the config file for the given key. If no such string exists, the user is asked for a
        default value, which is then stored in the configuration file. If store_in_defaults is True, then the default
        value is stored in the DEFAULTS section, otherwise the default value is stored in the section given to __init__.
        """
        if key in self._config[self._section_name] or key in self._config["DEFAULTS"]:
            # Setting exists, get it
            return self.get_or_default(key, "ERROR, VALUE SHOULD EXIST", store_in_defaults=store_in_defaults,
                                       comment=question)

        # Setting does not exist, get answer and store that
        try:
            default_value = input(question + " ")
        except EOFError:
            # Happens when the user presses CTRL + C.
            exit(200)
            return  # To keep Python linters from complaining that default_value may not be initialized on the next line
        return self.get_or_default(key, default_value, store_in_defaults=store_in_defaults, comment=question)

    def save(self) -> bool:
        """Saves the configuration file if there are any changes made. Returns True if the config file was saved,
        False if no save was necessary."""
        # Transform to RawConfigParser
        config_parser = RawConfigParser(allow_no_value=True)
        config_parser.optionxform = str  # Prevent parser from converting everything to lowercase
        for section_name, section in self._config.items():
            config_parser[section_name] = {}
            for setting_name, setting in section.items():
                if setting.comment:
                    config_parser[section_name]["; " + setting.comment] = None
                config_parser[section_name][setting_name] = setting.value

        # Write
        os.makedirs(self._folder_name, exist_ok=True)
        with open(os.path.join(self._folder_name, self.FILE_NAME), 'w', encoding="UTF-8") as config_writing:
            config_parser.write(config_writing)
        self.made_value_changes = False
        return True

    def save_and_exit_if_changed(self):
        """If the configuration file was changed, the changes are stored an then the program is exited. This allows the
        user to see what changes were made, and then restart the program."""
        if self.made_value_changes:
            self.save()
            print("Configuration file was updated automatically. Please review the settings in the"
                  " [" + self._section_name + "] section to check if they are correct, then rerun this command.")
            exit(301)
            return
        self.save()
