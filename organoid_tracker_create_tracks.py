#!/usr/bin/env python3

"""Creates links between known nucleus positions at different time points. Division and link probabilities are
necessary for this."""
import os

from organoid_tracker.config import ConfigFile, config_type_int, config_type_float
from organoid_tracker.imaging import io, list_io
from organoid_tracker.linking import dpct_linker
from organoid_tracker.linking.dpct_linker import calculate_appearance_penalty
from organoid_tracker.linking_analysis import cell_error_finder
from organoid_tracker.linking_analysis.links_postprocessor import postprocess, finetune_solution, connect_loose_ends, \
    bridge_gaps, pinpoint_divisions, remove_tracks_too_deep, bridge_gaps2, _remove_tracks_too_short, \
    _remove_single_positions, _remove_spurs_division

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("create_tracks")
_dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)
_method = config.get_or_default("track_finding_method", "FlowBased", comment="Can be FlowBased or Magnusson.")
_margin_um = int(config.get_or_default("margin_um", str(8)))
_link_weight = config.get_or_default("weight_links", str(1), comment="Penalty for link distance. Make this value"
                                                                     " higher if you're getting too many long-distance links. Lower this value if"
                                                                     " you're not ge tting enough links.",
                                     type=config_type_int)
_detection_weight = config.get_or_default("weight_detections", str(1), comment="Penalty for ignoring a detection."
                                                                               " Make this value higher if too many cells do not get any links.",
                                          type=config_type_int)
_division_weight = config.get_or_default("weight_division", str(1), comment="Score for creating a division. The"
                                                                            " higher, the more cell divisions will be created (although the volume of the"
                                                                            " cells still needs to be OK before any division is considered at all..",
                                         type=config_type_int)
_appearance_weight = config.get_or_default("weight_appearance", str(1), comment="Penalty for starting a track out of"
                                                                                " nowhere.", type=config_type_int)
_disappearance_weight = config.get_or_default("weight_dissappearance", str(1), comment="Penalty for ending a track.",
                                              type=config_type_int)
min_appearance_probability = config.get_or_default("min_appearance_probability", str(0.01),
                                                   comment="Estimate of a track appearing (no division).",
                                                   type=config_type_float)
min_disappearance_probability = config.get_or_default("min_disappearance_probability", str(0.01),
                                                      comment="Estimate of a track disappearing (no division).",
                                                      type=config_type_float)
_max_z = config.get_or_default("maximum z depth for which we want tracks", str(25),
                               comment="if tracks start and end above these heights (pixels) remove them",
                               type=config_type_int)
_minimum_track_length = config.get_or_default("minimum track length we want to keep", str(6),
                                              comment="tracks below this lenght are removed",
                                              type=config_type_int)
_margin_xy = config.get_or_default("remove positions below this distance from edge", str(0),
                                   comment="if positions are this close to the edge (pixels) remove them",
                                   type=config_type_int)
_links_output_folder = config.get_or_default("output_folder", "Output tracks")
config.save()
# END OF PARAMETERS

# Convert output folder to absolute path (as list_io changes the working directory)
_links_output_folder = os.path.abspath(_links_output_folder)

for experiment_index, experiment in enumerate(list_io.load_experiment_list_file(_dataset_file)):
    output_folder_experiment = os.path.join(_links_output_folder, f"{experiment_index + 1}. {experiment.name.get_save_name()}")
    if os.path.exists(os.path.join(output_folder_experiment, 'Clean.' + io.FILE_EXTENSION)):
        print(f"Experiment {experiment_index + 1} already processed, skipping.")
        continue

    possible_links = experiment.links

    # Check if images were loaded
    if not experiment.images.image_loader().has_images():
        print(f"No images were found for experiment \"{experiment.name}\". Please check the configuration file and make"
              f" sure that you have stored images at the specified location.")
        exit(1)

    print("calculate appearance and disappearance probabilities...")
    experiment = calculate_appearance_penalty(experiment, min_appearance_probability=min_appearance_probability,
                                              name="appearance_penalty", buffer_distance=_margin_um, only_top=True)
    experiment = calculate_appearance_penalty(experiment, min_appearance_probability=min_disappearance_probability,
                                              name="disappearance_penalty", buffer_distance=_margin_um, only_top=True)

    print("Deciding on what links to use...")
    link_result, naive_links = dpct_linker.run(experiment.positions, possible_links,
                                               link_weight=_link_weight,
                                               detection_weight=_detection_weight, division_weight=_division_weight,
                                               appearance_weight=_appearance_weight,
                                               dissappearance_weight=_disappearance_weight, method=_method)

    # The resulting tracks
    experiment_result = experiment.copy_selected(images=True, positions=True, name=True, links=False, global_data=True,
                                                 warning_limits=True)
    experiment_result.links = link_result
    experiment_result.links.merge_link_meta_data(possible_links)

    # This includes the unpruned set of links, important for later marginalization
    experiment_all = experiment.copy_selected(images=True, positions=True, name=True, links=False, global_data=True)
    experiment_all.links = naive_links
    experiment_all.links.merge_link_meta_data(possible_links)

    # Results before post-processing:
    #io.save_data_to_json(experiment_result, '00' + _links_output_file)

    # finetune solution
    experiment_result = finetune_solution(experiment_all, experiment_result)
    experiment_result = finetune_solution(experiment_all, experiment_result)
    experiment_result = finetune_solution(experiment_all, experiment_result)
    experiment_result = finetune_solution(experiment_all, experiment_result)

    # After finetinetuning, no change to the graph structure
    #io.save_data_to_json(experiment_result, '01' + _links_output_file)

    # Connect loose ends, use all available links (experiment) not the pruned set (experiment_all)
    experiment_result, experiment = connect_loose_ends(experiment, experiment_result)

    # remove the positions that were removed during connect_loose_ends from the file containing all links
    # this is done here because the experiment_all file is already pruned in dpct_linker()
    to_remove = []
    for position in experiment_all.positions:
        if not experiment_result.positions.contains_position(position):
            to_remove.append(position)
    experiment_all.remove_positions(to_remove)

    # add the links that were used during connect_loose_ends
    for link in experiment_result.links.find_all_links():
        if not experiment_all.links.contains_link(link[0], link[1]):
            experiment_all.links.add_link(link[0], link[1])

    for link in experiment_result.links.find_all_links():
        if experiment_all.links.contains_link(link[0], link[1]):
            experiment_all.link_data.set_link_data(link[0], link[1], 'link_penalty',
                                                   experiment_result.link_data.get_link_data(link[0], link[1],
                                                                                             'link_penalty'))
            experiment_all.link_data.set_link_data(link[0], link[1], 'link_probability',
                                                   experiment_result.link_data.get_link_data(link[0], link[1],
                                                                                             'link_probability'))

    #io.save_data_to_json(experiment_result, '03' + _links_output_file)

    # connect tracks broken up by a missing cell detection
    experiment_result, experiment_all = bridge_gaps(experiment_all, experiment_result)

    # connect tracks broken up by a link that is not part of the proposed set
    experiment_result, experiment_all = bridge_gaps2(experiment_all, experiment_result)

    # align the division in tracks so they match the division predictions from the neural network
    experiment_result, experiment_all = pinpoint_divisions(experiment_all, experiment_result)
    experiment_result, experiment_all = _remove_spurs_division(experiment_all, experiment_result)
    # io.save_data_to_json(experiment_result, '1' + _links_output_file)

    print("Checking results for common errors...")
    warning_count, no_links_count = cell_error_finder.find_errors_in_experiment(experiment_result)
    print("Writing results to file...")
    os.makedirs(output_folder_experiment, exist_ok=True)
    io.save_data_to_json(experiment_result, os.path.join(output_folder_experiment, 'Links.' + io.FILE_EXTENSION))
    io.save_data_to_json(experiment_all, os.path.join(output_folder_experiment, 'All possible links.' + io.FILE_EXTENSION))

    print(f"Done! Found {warning_count} potential errors in the data. In addition, {no_links_count} positions didn't get"
          f" links.")

    print("Applying final touches too clean up...")
    # Remove deep tracks and tracks on the edge
    experiment_result = remove_tracks_too_deep(experiment_result, max_z=_max_z)
    postprocess(experiment_result, margin_xy=_margin_xy)

    # remove too short tracks
    experiment_result, experiment_all = _remove_tracks_too_short(experiment_result, experiment_all,
                                                                 min_t=_minimum_track_length)
    experiment_result, experiment_all = _remove_single_positions(experiment_result, experiment_all)

    warning_count, no_links_count = cell_error_finder.find_errors_in_experiment(experiment_result)
    print(f"Done! Found {warning_count} potential errors in the data. In addition, {no_links_count} positions didn't get"
          f" links.")

    io.save_data_to_json(experiment_all, os.path.join(output_folder_experiment, 'All possible links clean.' + io.FILE_EXTENSION))
    io.save_data_to_json(experiment_result, os.path.join(output_folder_experiment, 'Clean.' + io.FILE_EXTENSION))
