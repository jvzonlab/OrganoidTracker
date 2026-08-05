from organoid_tracker.imaging import list_io
from organoid_tracker.linking import cell_division_finder

AUTLIST_FILE = r"D:\Nextcloud\rkok\Biohub competition\Official data\Training tracks Rutger.autlist"


def main():
    total_divisions = 0
    for experiment in list_io.load_experiment_list_file(AUTLIST_FILE, load_images=False):
        divisions = cell_division_finder.find_mothers(experiment.links, exclude_multipolar=False)
        if len(divisions) > 0:
            total_divisions += len(divisions)
            print(f" Found {len(divisions)} divisions in {experiment.name}")


    print(f"\n\nTotal divisions found: {total_divisions}")


if __name__ == "__main__":
    main()
