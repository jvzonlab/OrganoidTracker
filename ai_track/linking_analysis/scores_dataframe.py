"""Creates a Dataframe (a scientific table) from scored families, explaining how a score is built up."""
from typing import List, Set

import numpy
import pandas

from ai_track.core.experiment import Experiment
from ai_track.core.position import Position
from ai_track.core.score import Family, Score
from ai_track.linking.scoring_system import MotherScoringSystem


def create(experiment: Experiment, putative_families: List[Family], scoring_system: MotherScoringSystem,
           actual_mothers: Set[Position]):
    tryout_score = _score(experiment, putative_families[0], scoring_system)
    keys = ["mother"] + tryout_score.keys() + ["total_score", "is_actual_mother"]

    scores_of_mother = dict()
    i = 0
    for family in putative_families:
        if i % 50 == 0:
            print("Working on putative family " + str(i) + "/" + str(len(putative_families)))

        score = _score(experiment, family, scoring_system)
        previous_score = scores_of_mother.get(family.mother)
        if previous_score is None or previous_score.total() < score.total():
            scores_of_mother[family.mother] = score
        i += 1

    i = 0
    data = numpy.zeros([len(scores_of_mother), len(keys)], dtype=numpy.object)
    for mother, score in scores_of_mother.items():
        if not isinstance(mother, Position):
            print(mother)
        data[i, 0] = str(mother)
        for j in range(1, len(keys) - 2):
            key = keys[j]
            data[i, j] = score.get(key)
        data[i, len(keys) - 2] = score.total()
        data[i, len(keys) - 1] = 1 if mother in actual_mothers else 0
        i += 1

    return pandas.DataFrame(data=data, columns=_table_names(keys))


def _table_names(keys: List[str]) -> List[str]:
    names = []
    for key in keys:
        if key.startswith("mother_"):
            key = key.replace("mother_", "m_")
        if key.startswith("daughters_"):
            key = key.replace("daughters_", "d_")
        names.append(key)
    return names


def _score(experiment: Experiment, family: Family, scoring_system: MotherScoringSystem) -> Score:
    return scoring_system.calculate(experiment.images, experiment.positions, family)
