"""Predictions particle positions using an already-trained convolutional neural network."""

from organoid_tracker.core.position import Position
from organoid_tracker.config import ConfigFile, config_type_bool
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

_reviewed = config.get_or_default("is the data (partially) reviewed?", str(False), type=config_type_bool)
_fully_reviewed = config.get_or_default("are all (dis)appearances reviewed?", str(False), type=config_type_bool)

# Load experiments
experiment = io.load_data_file(_experiment_file, _min_time_point, _max_time_point)

experiment_all_links = io.load_data_file(_all_links_file, _min_time_point, _max_time_point)

_output_file = config.get_or_default("marginalized_output_file", "Marginalized positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_filtered_output_file = config.get_or_default("filtered_output_file", "Filtered positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")

config.save_and_exit_if_changed()
# END OF PARAMETERS

if _reviewed:
    # remove single positions
    remove = []
    for position in experiment.positions:
        if not experiment.links.contains_position(position):
            remove.append(position)

    experiment.remove_positions(remove)

    # remove positions not present in curated data
    remove = []
    for position in experiment_all_links.positions:
        if not experiment.links.contains_position(position):
            remove.append(position)
    experiment_all_links.remove_positions(remove)

    # check which links have been corrected
    for position1, position2 in experiment.links.find_all_links():

        # newly created link? (bit ugly)
        if ((experiment.link_data.get_link_data(position1, position2, 'marginal_probability') is None)
                or experiment.link_data.get_link_data(position1, position2, 'link_penalty') is None):

            # set high probability to newly created links
            experiment_all_links.link_data.set_link_data(position1, position2, data_name="link_probability",
                                           value=1-10**-10)
            experiment_all_links.link_data.set_link_data(position1, position2, data_name="link_penalty",
                                               value=-10.)

            # set low probability to connecting links that are not part of the tracking
            for other_pos in experiment_all_links.links.find_pasts(position2):
                if not experiment.links.contains_link(other_pos, position2):
                    experiment_all_links.link_data.set_link_data(other_pos, position2, data_name="link_probability",
                                                                 value=10 ** -10)
                    experiment_all_links.link_data.set_link_data(other_pos, position2, data_name="link_penalty",
                                                                 value=10.)

            for other_pos in experiment_all_links.links.find_futures(position1):
                if not experiment.links.contains_link(position1, other_pos):
                    experiment_all_links.link_data.set_link_data(position1, other_pos, data_name="link_probability",
                                                                 value=10 ** -10)
                    experiment_all_links.link_data.set_link_data(position1, other_pos, data_name="link_penalty",
                                                                 value=10.)
        else:
            # Is there an error corrected
            error = experiment.position_data.get_position_data(position1, 'error') == 14
            suppressed_error = experiment.position_data.get_position_data(position2, 'suppressed_error') == 14

            # Is the error removed or supressed?
            if experiment.link_data.get_link_data(position1, position2, 'marginal_probability')<0.99 and not (error and not suppressed_error): #0.996 for example because the probs are not tempeerature scaled well
                experiment_all_links.link_data.set_link_data(position1, position2, data_name="link_probability",
                                                   value=1 - 10 ** -10)
                experiment_all_links.link_data.set_link_data(position1, position2, data_name="link_penalty",
                                                   value=-10.)


    # check which loose ends have been corrected
    loose_ends = list(experiment.links.find_disappeared_positions(
                time_point_number_to_ignore=experiment.last_time_point_number()))
    for position in loose_ends:
        error = experiment.position_data.get_position_data(position, 'error') == 1
        suppressed_error = experiment.position_data.get_position_data(position, 'suppressed_error') == 1

        # Is the error removed or supressed?
        if not (error and not suppressed_error):
            # cells have to disappear
            experiment_all_links.position_data.set_position_data(position, 'disappearance_probability', value=1 - 10 ** -10)
            experiment_all_links.position_data.set_position_data(position, 'disappearance_penalty', value=-10)

            # link is also corrected
            prev_pos = experiment.links.find_single_past(position)
            experiment_all_links.link_data.set_link_data(prev_pos, position, data_name="link_probability",
                                           value=1-10**-10)
            experiment_all_links.link_data.set_link_data(prev_pos, position, data_name="link_penalty",
                                               value=-10)

    # check which loose starts have been corrected
    loose_starts = list(experiment.links.find_appeared_positions(
        time_point_number_to_ignore=experiment.first_time_point_number()))
    for position in loose_starts:
        error = experiment.position_data.get_position_data(position, 'error') == 5
        suppressed_error = experiment.position_data.get_position_data(position, 'suppressed_error') == 5

        # Is the error removed or supressed?
        if not (error and not suppressed_error):
            # cells have to appear
            experiment_all_links.position_data.set_position_data(position, 'appearance_probability',
                                                                 value=1 - 10 ** -10)
            experiment_all_links.position_data.set_position_data(position, 'appearance_penalty', value=-10.)

            # link is also corrected
            next_pos = experiment.links.find_single_future(position)
            experiment_all_links.link_data.set_link_data(position, next_pos, data_name="link_probability",
                                           value=1-10**-10)
            experiment_all_links.link_data.set_link_data(position, next_pos, data_name="link_penalty",
                                               value=-10.)

if _fully_reviewed:
    # change appearance probabilities reflecting fully checked nature. If all cell endings are taken care of then cells cannot appear anymore (except at the image volume edges)
    for position in experiment.positions:
        appearance_penalty = experiment.position_data.get_position_data(position, 'appearance_penalty')
        if appearance_penalty is not None:
            # do not correct the penalties if they are are deemed likely to appear for other reasons.
            if appearance_penalty > 1.5:
                if (position.time_point_number()-experiment.first_time_point_number())>1:
                    experiment_all_links.position_data.set_position_data(position, 'appearance_probability',
                                                                         value=10 ** -10)
                    experiment_all_links.position_data.set_position_data(position, 'appearance_penalty', value=10.)

if _reviewed or _fully_reviewed:
    print("Saving files...")
    io.save_data_to_json(experiment_all_links, "updated_all_links.aut")

number_of_links = len(list(experiment.links.find_all_links()))
index = 1

# Marginalize
for (position1, position2) in experiment.links.find_all_links():
    print(f"Marginalizing {index}/{number_of_links}")

    link_penalty = experiment.link_data.get_link_data(position1, position2, 'link_penalty')

    if link_penalty is None:
        print('missing penalty')
        link_penalty=0

    if (not experiment_all_links.links.contains_link(position1, position2)):# or (link_penalty is None):
        print('link not present in graph, assumed to be corrected by user')
        print(position1)
        print(position2)
        experiment.link_data.set_link_data(position1, position2, data_name="marginal_probability",
                                           value=1)
    else:
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

