import os
import shlex
import sys

from organoid_tracker.config import ConfigFile


def find_conda_batch_file() -> str:
    """Finds the path to 'conda.bat' for activating conda environments on Windows."""

    # First attempt: find 'conda.bat' in the PATH directories
    path_variable = os.environ.get('PATH', '')
    for directory in path_variable.split(os.pathsep):
        if directory.endswith('condabin'):
            conda_bat_path = os.path.join(directory, 'conda.bat')
            if os.path.exists(conda_bat_path):
                return conda_bat_path

    # Second attempt: from Python executable path
    conda_env_folder = os.sep + "envs" + os.sep
    conda_installation_folder = sys.base_exec_prefix
    if conda_env_folder in conda_installation_folder:
        conda_installation_folder = conda_installation_folder[0:conda_installation_folder.index(conda_env_folder)]
    conda_bat_path = os.path.join(conda_installation_folder, 'condabin', 'conda.bat')
    if os.path.exists(conda_bat_path):
        return conda_bat_path

    # Fallback, only works if conda is in PATH
    return "conda"


def find_conda_environment_name() -> str:
    """Finds the name of the current conda environment."""

    # First attempt: check CONDA_DEFAULT_ENV environment variable
    conda_default_env = os.getenv('CONDA_DEFAULT_ENV')
    if conda_default_env:
        return conda_default_env

    # Second attempt: from Python executable path
    conda_env_folder = os.sep + "envs" + os.sep
    conda_installation_folder = sys.base_exec_prefix
    if conda_env_folder in conda_installation_folder:
        env_name = conda_installation_folder.split(conda_env_folder)[-1].split(os.sep)[0]
        return env_name

    return "base"


def create_run_script(output_folder: str, command_name: str):
    """If you have a plugin that uses get_commands to register commands, then you can call this method to create a
    BAT or SH file that calls that command. This will create those files, along with a config file pointing to the
    current plugin directory."""
    os.makedirs(output_folder, exist_ok=True)
    script_file = os.path.abspath("organoid_tracker.py")

    # For Windows
    bat_file = os.path.join(output_folder, command_name + ".bat")
    with open(bat_file, "w") as writer:
        writer.write(f"""@rem Automatically generated script for running {command_name}
@echo off
@CALL "{find_conda_batch_file()}" activate {find_conda_environment_name()}
"{sys.executable}" "{script_file}" "{command_name}"
pause""")

    # For Linux
    sh_file = os.path.join(output_folder, command_name + ".sh")
    with open(sh_file, "w") as writer:
        writer.write(f"""#!/bin/bash
# Automatically generated script for running {command_name}
{shlex.quote(sys.executable)} {shlex.quote(script_file)} {shlex.quote(command_name)}
""")
    os.chmod(sh_file, 0o777)

    # For both: register the plugin directory
    extra_plugin_directory = ConfigFile("scripts").get_or_default("extra_plugin_directory", default_value="")
    config = ConfigFile("scripts", folder_name=output_folder)
    config.get_or_default("extra_plugin_directory", extra_plugin_directory)
    config.save()
