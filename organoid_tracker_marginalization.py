"""Predictions particle positions using an already-trained convolutional neural network."""

from organoid_tracker.core.position import Position
from organoid_tracker.config import ConfigFile
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader

from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import cell_error_finder
from organoid_tracker.local_marginalization.local_marginalization_functions import local_marginalization, \
    minimal_marginalization

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("marginalisation")
_experiment_file = config.get_or_default("solution_links_file",
                                            "cleanAutomatic links.aut",
                                            comment="What are the detected positions for those images?")
_all_links_file = config.get_or_default("all_links_file",
                                            "all_linksAutomatic links.aut",
                                            comment="What are the detected positions for those images?")

_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

_steps = int(config.get_or_default("size subset (steps away from link of interest)", str(3)))
_temperature = float(config.get_or_default("temperature (to account for shared information)", str(1.5)))
_filter_cut_off = float(config.get_or_default("maximum error rate", str(0.01)))

# Load experiments
experiment = io.load_data_file(_experiment_file, _min_time_point, _max_time_point)

experiment_all_links = io.load_data_file(_all_links_file, _min_time_point, _max_time_point)

_output_file = config.get_or_default("marginalized_output_file", "Marginalized positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_filtered_output_file = config.get_or_default("filtered_output_file", "Filtered positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")

config.save_and_exit_if_changed()
# END OF PARAMETERS

number_of_links = len(list(experiment.links.find_all_links()))
index = 1

# Marginalize
for (position1, position2), link_penalty in experiment.link_data.find_all_links_with_data("link_penalty"):
    print(f"Marginalizing {index}/{number_of_links}")

    print(position1)
    # if links are extremely (un)likely we do not have to perform marginalization over a large subgraph.
    if abs(link_penalty) < 4.:
        marginalized_probability = local_marginalization(position1, position2, experiment_all_links, complete_graph=True, steps=_steps, scale=1/_temperature)
    else:
        marginalized_probability = minimal_marginalization(position1, position2, experiment_all_links,  scale=1/_temperature)

    experiment.link_data.set_link_data(position1, position2, data_name="marginal_probability",
                                       value=marginalized_probability)

    index = index +1

# Assign errors based on marginalized scores
print("Checking results for common errors...")
warning_count, no_links_count = cell_error_finder.find_errors_in_experiment(experiment, marginalization=True)

print("Saving files...")
io.save_data_to_json(experiment, _output_file)

# Filter data
experiment_filtered = experiment.copy_selected(images = True, positions = True, position_data = True,
                      links = True, link_data = True, global_data = True)

mothers = cell_division_finder.find_mothers(experiment.links)
count = 0

for (position1, position2), marginal_probability in experiment.link_data.find_all_links_with_data("marginal_probability"):

    if (marginal_probability is not None):
        # remove uncertain links
        if marginal_probability < (1-_filter_cut_off):
            count = count + 1
            experiment_filtered.links.remove_link(position1, position2)

            # if the removed link is from a mother also remove the other outgoing links to avoid confusion about of it is dividing or not
            if position1 in mothers:
                for next_position in experiment.links.find_futures(position1):
                    experiment_filtered.links.remove_link(position1, next_position)

    # if the division probability is high, but no division is assigned, we add one to avoid misinterpretation
    division_probability = experiment.position_data.get_position_data(position1, 'division_probability')
    if division_probability is None:
        print('no available division probability')
    else:
        if (division_probability > 0.99) and (position2 not in mothers) and (position1 not in mothers):
            add_position = Position(x=position2.x +1,
                                        y=position2.y +1,
                                        z=position2.z,
                                        time_point=position2.time_point())

            experiment_filtered.positions.add(add_position)
            experiment_filtered.links.add_link(position1, add_position)

print('removed uncertain links:')
print(count)
print("Saving files...")
io.save_data_to_json(experiment_filtered, _filtered_output_file)

