"""Creates a Dataframe (a scientific table) from scored families, explaining how a score is built up."""
from typing import List, Set

import numpy
import pandas

from autotrack.core.experiment import Experiment
from autotrack.core.score import Family, Score
from autotrack.linking.scoring_system import MotherScoringSystem


def create(experiment: Experiment, putative_families: List[Family], scoring_system: MotherScoringSystem,
           actual_families: Set[Family]):
    tryout_score = _score(experiment, putative_families[0], scoring_system)
    keys = ["mother"] + tryout_score.keys() + ["total_score", "is_actual_mother"]
    data = numpy.zeros([len(putative_families), len(keys)], dtype=numpy.object)

    for i in range(len(putative_families)):
        if i % 50 == 0:
            print("Working on putative family " + str(i) + "/" + str(len(putative_families)))

        family = putative_families[i]
        score = _score(experiment, family, scoring_system)
        data[i, 0] = str(family)
        for j in range(1, len(keys) - 2):
            key = keys[j]
            data[i, j] = score.get(key)
        data[i, len(keys) - 2] = score.total()
        data[i, len(keys) - 1] = 1 if family in actual_families else 0

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
    daughters = list(family.daughters)
    return scoring_system.calculate(experiment, family.mother, daughters[0], daughters[1])
