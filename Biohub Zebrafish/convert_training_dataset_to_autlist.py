import os

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import zarr_image_loader
from organoid_tracker.imaging import geff_io, list_io

_INPUT_PATH = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\raw_data\Microscopy\Dataset Biohub zebrafish embryo\train"
_OUTPUT_PATH = r"P:\Rodriguez_Colman\vidi_rodriguez_colman\rkok\data_analysis\2026\2026-07 RK0275 Rutger Training on Biohub cell tracking dataset\Official data"

def main():
    print(f"Creating autlist file for training data in folder '{_INPUT_PATH}'")

    print("Discovering datasets...")
    experiments = list()
    for subfolder_name in os.listdir(_INPUT_PATH):
        if not subfolder_name.endswith(".zarr"):
            continue

        print(subfolder_name, end="  ")
        zarr_subfolder = os.path.join(_INPUT_PATH, subfolder_name)
        experiment = Experiment()
        zarr_image_loader.load_from_zarr_file(experiment, zarr_subfolder)
        geff_io.load_data_file(zarr_subfolder.replace(".zarr", ".geff"), experiment=experiment)
        experiment.name.set_name(experiment.name.get_name(), is_automatic=False)  # Flag name as final
        experiments.append(experiment)
    print("\nSaving...")
    os.makedirs(_OUTPUT_PATH, exist_ok=True)
    list_io.save_experiment_list_file(experiments, os.path.join(_OUTPUT_PATH, "Training tracks.autlist"),
                                      tracking_files_folder=os.path.join(_OUTPUT_PATH, "Training track files"))


if __name__ == "__main__":
    main()