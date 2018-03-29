from typing import List, Set

import numpy
import pandas

from imaging import Experiment, Particle
from linking_analysis.mother_finder import Family
from linking.score_system import MotherScoringSystem, Score


def create(experiment: Experiment, putative_families: List[Family], scoring_system: MotherScoringSystem,
           actual_families: Set[Family]):
    tryout_score = _score(experiment, putative_families[0], scoring_system)
    keys = tryout_score.keys()
    data = numpy.zeros([len(putative_families), len(keys) + 1], dtype=numpy.float32)

    for i in range(len(putative_families)):
        if i % 50 == 0:
            print("Working on putative family " + str(i) + "/" + str(len(putative_families)))

        family = putative_families[i]
        score = _score(experiment, family, scoring_system)
        for j in range(len(keys)):
            key = keys[j]
            data[i, j] = score.get(key)
        data[i, len(keys)] = 1 if family in actual_families else 0

    return pandas.DataFrame(data=data, columns=keys + ["is_actual_family"])

def _score(experiment: Experiment, family: Family, scoring_system: MotherScoringSystem) -> Score:
    daughters = list(family.daughters)
    return scoring_system.calculate(experiment, family.mother, daughters[0], daughters[1])
