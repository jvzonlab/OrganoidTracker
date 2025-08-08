"""Called using `python organoid_tracker.py analyze_link_scores`. Outputs a CSV file with the following columns:
Predicted, Correct, Predicted chance, XYZT
This is useful to check how correct the scores of the linking network are.
"""
import csv
from typing import Dict, Callable, List, NamedTuple, Optional, Tuple

import numpy
from numpy import ndarray

from organoid_tracker.config import ConfigFile, config_type_csv_file
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.imaging import io
from organoid_tracker.local_marginalization.local_marginalization_functions import local_marginalization, minimal_marginalization
#from organoid_tracker_plugins.plugin_tracking_output_link_changes import minimal_marginalization

_INT_MIN = numpy.iinfo(numpy.int32).min
_INT_MAX = numpy.iinfo(numpy.int32).max


def get_commands() -> Dict[str, Callable[[List[str]], int]]:
    return {
        "evaluate_marginalization_rebuttal": main
    }


class _PositionMatcher:

    _from_to: Dict[Position, Position]

    def __init__(self, resolution: ImageResolution, positions_from: List[Position], positions_to: List[Position]):
        costs_yfrom_xto = numpy.empty((len(positions_from), len(positions_to)), dtype=numpy.float32)
        for y, position_from in enumerate(positions_from):
            for x, position_to in enumerate(positions_to):
                costs_yfrom_xto[y, x] = position_from.distance_squared(position_to, resolution)

        results, total_cost = _vogel_solve(costs_yfrom_xto)
        self._from_to = dict()
        for y, x in zip(*numpy.nonzero(results)):
            self._from_to[positions_from[y]] = positions_to[x]
        #print(self._from_to)

    def get_mapping(self, position: Position) -> Optional[Position]:
        return self._from_to.get(position)


def _vogel_solve(costs: ndarray) -> Tuple[ndarray, float]:
    """Solves the transport problem using Vogel's approximation method. Every cell in the costs array is the cost of
    going from y to x. Returns an array of the same shape, that is True at the optimal distribution of costs."""
    particle_count_start = costs.shape[0]
    particle_count_end = costs.shape[1]

    row_done = numpy.zeros(particle_count_start, dtype=bool)
    col_done = numpy.zeros(particle_count_end, dtype=bool)
    results = numpy.zeros((particle_count_start, particle_count_end), dtype=bool)

    def next_cell() -> Tuple[int, int, float, float]:
        res1 = max_penalty(particle_count_start, particle_count_end, True)
        res2 = max_penalty(particle_count_end, particle_count_start, False)
        if res1[3] == res2[3]:
            return res1 if (res1[2] < res2[2]) else res2
        return res2 if (res1[3] > res2[3]) else res1

    def diff(j: int, len: int, is_row: bool) -> Tuple[float, float, int]:
        min_cost = _INT_MAX
        min_cost_2 = min_cost
        min_pos = -1
        for i in range(len):
            done = col_done[i] if is_row else row_done[i]
            if done:
                continue
            cost = costs[j][i] if is_row else costs[i][j]
            if cost < min_cost:
                min_cost_2 = min_cost
                min_cost = cost
                min_pos = i
            elif cost < min_cost_2:
                min_cost_2 = cost
        return min_cost_2 - min_cost, min_cost, min_pos

    def max_penalty(len1: int, len2: int, is_row: bool) -> Tuple[int, int, float, float]:
        md = _INT_MIN
        pc = -1
        pm = -1
        mc = -1
        for i in range(len1):
            done = row_done[i] if is_row else col_done[i]
            if done:
                continue
            res = diff(i, len2, is_row)
            if res[0] > md:
                md = res[0]  # max diff
                pm = i  # pos of max diff
                mc = res[1]  # min cost
                pc = res[2]  # pos of min cost
        return (pm, pc, mc, md) if is_row else (pc, pm, mc, md)

    input_particles_left = particle_count_start
    total_cost = 0
    while input_particles_left > 0:
        cell = next_cell()
        row_index = cell[0]
        column_index = cell[1]

        if row_index == -1 or column_index == -1:
            break  # No more particles left to link to

        col_done[column_index] = True
        row_done[row_index] = True
        results[row_index, column_index] = True
        input_particles_left -= 1
        total_cost += costs[row_index, column_index]
    return results, total_cost



class _ParsedConfig(NamedTuple):
    """Holds the configuration values for the script."""
    baseline_experiment: Experiment
    link_scores_experiment: Experiment
    scratch_experiment: Experiment
    output_csv: str


def _parse_config() -> _ParsedConfig:
    print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
    config = ConfigFile("analyze_link_scores")
    _min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
    _max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
    _link_scores_file = config.get_or_prompt("link_scores_file", "In what file are the link scores stored?")
    _automatic_links_file = config.get_or_prompt("automatic_file", "In what file are the resulting links stored?")
    _baseline_links_file = config.get_or_prompt("ground_truth_file", "In what file are the ground truth-links stored?")
    _output_file = config.get_or_default("link_scores_csv", "link_scores.csv", type=config_type_csv_file)
    config.save_and_exit_if_changed()
    # END OF PARAMETERS
    print("Starting...")
    scratch_experiment = io.load_data_file(_automatic_links_file, _min_time_point, _max_time_point)
    link_scores_experiment = io.load_data_file(_link_scores_file, _min_time_point, _max_time_point)
    baseline_experiment = io.load_data_file(_baseline_links_file, _min_time_point, _max_time_point)
    return _ParsedConfig(baseline_experiment=baseline_experiment, scratch_experiment=scratch_experiment,
                         link_scores_experiment=link_scores_experiment, output_csv=_output_file)


def _is_link_correct(config: _ParsedConfig, position1: Position, position2: Position,
                     mapping1: _PositionMatcher, mapping2: _PositionMatcher,
                     back_mapping1: _PositionMatcher, back_mapping2: _PositionMatcher) -> Optional[bool]:
    """Checks whether the given link is correct, according to the ground truth. Returns None if unknown, which happens
    if there is no ground truth at that location."""
    ground_position1 = mapping1.get_mapping(position1)
    ground_position2 = mapping2.get_mapping(position2)
    if ground_position1 is None or ground_position2 is None:
        return None
    return config.baseline_experiment.links.contains_link(ground_position1, ground_position2)


def main(args: List[str]) -> int:
    config = _parse_config()

    print("Comparing...")

    with open(config.output_csv, "w", newline='') as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(("Predicted", "Correct", "Predicted chance", "Marginal chance", "Marginal chance minimal",
                             "X", "Y", "Z", "X2", "Y2", "Z2", "T"))

        resolution = config.baseline_experiment.images.resolution()
        mapping = None  # Mapping for a single time point for the link scores to ground truth positions
        back_mapping = None  # Mapping for a single time point for the link scores to ground truth positions
        mapping_previous = None  # Same, but for the previous time point
        back_mapping_previous = None  # Same, but for the previous time point
        for time_point in config.link_scores_experiment.positions.time_points():
            mapping_previous = mapping
            back_mapping_previous = back_mapping
            mapping = _PositionMatcher(resolution,
                list(config.link_scores_experiment.positions.of_time_point(time_point)),
                list(config.baseline_experiment.positions.of_time_point(time_point)))

            back_mapping = _PositionMatcher(resolution,
                list(config.baseline_experiment.positions.of_time_point(time_point)),
                list(config.link_scores_experiment.positions.of_time_point(time_point)))

            for position1 in config.link_scores_experiment.positions.of_time_point(time_point):
                for position2 in config.link_scores_experiment.links.find_pasts(position1):
                    link_probability = config.link_scores_experiment.links.get_link_data(position1, position2, "link_probability")
                    if link_probability is None:
                        print(position1)
                        print(position2)
                        continue  # No probability calculated

                    is_correct = _is_link_correct(config, position1, position2, mapping, mapping_previous, back_mapping, back_mapping_previous)
                    if is_correct is None:
                        is_correct = None
                        print('no ground truth av')
                        # continue  # No ground truth available for this link

                    is_predicted = config.scratch_experiment.links.contains_link(position1, position2)
                    avg_position = (position1 + position2) * 0.5

                    if abs(numpy.log10((link_probability + 10 ** -10) / (1 - link_probability - 10 ** -10))) < 4:

                        #print(avg_position.time_point_number())
                        #print(avg_position.x)
                        #print(avg_position.y)
                        #print(avg_position.z)


                        marginal_chance = local_marginalization(position1, position2, config.link_scores_experiment, complete_graph=True, steps=3, scale=0.66)
                        marginal_chance_extra = local_marginalization(position1, position2, config.link_scores_experiment,
                                                                complete_graph=True, steps=3, scale=0.66)
                        marginal_chance_minimal = minimal_marginalization(position1, position2,
                                                                          config.link_scores_experiment, scale=0.77)

                        #print('priori')
                        #print(link_probability)
                        #print('posteriori')
                        #print(marginal_chance_minimal)
                        #print(marginal_chance)


                    else:
                        marginal_chance = minimal_marginalization(position1, position2, config.link_scores_experiment, scale=0.66)
                        marginal_chance_extra = minimal_marginalization(position1, position2, config.link_scores_experiment, scale=0.65)
                        marginal_chance_minimal = minimal_marginalization(position1, position2,
                                                                          config.link_scores_experiment, scale=0.77)

                    config.link_scores_experiment.links.set_link_data(position1, position2,
                                                                      data_name="marginal_probability",
                                                                      value=float(marginal_chance))
                    config.link_scores_experiment.links.set_link_data(position1, position2,
                                                                      data_name="marginal_probability_minimal",
                                                                      value=float(marginal_chance_minimal))
                    config.link_scores_experiment.links.set_link_data(position1, position2,
                                                                      data_name="marginal_probability_extra",
                                                                      value=float(marginal_chance))
                    config.link_scores_experiment.links.set_link_data(position1, position2,
                                                                      data_name="true_link",
                                                                      value=bool(is_correct))

                    csv_writer.writerow((is_predicted, is_correct, link_probability, marginal_chance,
                                         marginal_chance_minimal, marginal_chance_extra,  position1.x, position1.y,
                                 position1.z, position2.x, position2.y,
                                 position2.z, avg_position.time_point_number()))

        print("Done! Saved to " + config.output_csv + ".")

        io.save_data_to_json(config.link_scores_experiment, 'links_with_marginals.aut')

        for position1 in config.scratch_experiment.positions:
            for position2 in config.scratch_experiment.links.find_pasts(position1):
                marginal_probability = config.link_scores_experiment.links.get_link_data(position1, position2,
                                                                                         "marginal_probability")
                if marginal_probability is None:
                    print(position1)
                    print(position2)

                if marginal_probability < 0.99:
                    config.link_scores_experiment.links.remove_link(position1, position2)


        io.save_data_to_json(config.link_scores_experiment, 'links_filtered.aut')

        return 0


if __name__ == "__main__":
    main([])