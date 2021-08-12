import os
import shlex
import sys

from organoid_tracker.config import ConfigFile


def create_run_script(output_folder: str, command_name: str):
    """If you have a plugin that uses get_commands to register commands, then you can call this method to create a
    BAT or SH file that calls that command. This will create those files, along with a config file pointing to the
    current plugin directory."""
    os.makedirs(output_folder, exist_ok=True)
    script_file = os.path.abspath("organoid_tracker.py")

    # For Windows
    conda_env_folder = os.sep + "envs" + os.sep
    conda_installation_folder = sys.base_exec_prefix
    if conda_env_folder in conda_installation_folder:
        conda_installation_folder = conda_installation_folder[0:conda_installation_folder.index(conda_env_folder)]
    bat_file = os.path.join(output_folder, command_name + ".bat")
    with open(bat_file, "w") as writer:
        writer.write(f"""@rem Automatically generated script for running {command_name}
@echo off
@CALL "{conda_installation_folder}\\condabin\\conda.bat" activate {os.getenv('CONDA_DEFAULT_ENV')}
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
