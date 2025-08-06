import itertools

import numpy as np

def minimal_marginalization(position1, position2, experiment, scale=1):
    """Performs a marginalization on a susbet containing all the links with the same target node as the link of interest."""

    if position1.time_point_number() > position2.time_point_number():
        position1, position2 = position2, position1

    # list of positions in subset
    possible_previous = experiment.links.find_pasts(position2)

    # sum of all link likelihoods (the partition function)
    marginal = 10 ** ((-experiment.positions.get_position_data(position2, data_name='appearance_penalty'))*scale)

    for previous_pos in possible_previous:
        marginal = marginal + 10 ** ((-experiment.link_data.get_link_data(previous_pos, position2, data_name='link_penalty'))*scale)

    # actual marginalization
    return (10 ** ((-experiment.link_data.get_link_data(position1, position2, data_name='link_penalty'))*scale)) / marginal


def find_local_set(previous_pos, current_pos, links, steps=2, first_backward=True):
    """Defines local subset of nodes and links based on the distance to the target node of the link of interest.
    Link is defined by the current cell position and ist previous one.
    Distance from the source node is used if the first step is not backward in time."""

    previous_pos = {previous_pos}
    current_pos = {current_pos}

    links_out = []
    links_in = []

    i = 0
    while i <= steps:
        if first_backward:
            for pos in current_pos:
                pasts = links.find_pasts(pos)
                previous_pos = previous_pos.union(pasts)

                links_in = links_in + [pos] * len(pasts)
                links_out = links_out + list(pasts)
            i = i + 1

        first_backward = True

        if i >= steps:
            break

        for pos in previous_pos:
            futures = links.find_futures(pos)
            current_pos = current_pos.union(futures)

            links_out = links_out + [pos] * len(futures)
            links_in = links_in + list(futures)
        i = i + 1

    # unique links
    local_links = set(list(zip(links_out, links_in)))

    return current_pos, previous_pos, local_links


def check_completeness(complete_pos, links):
    """For every node we check if all the links leaving (out) or coming into (in) it are present in the subset (links)"""

    complete_in = {}
    complete_out = {}

    for pos in complete_pos:
        pasts = links.find_pasts(pos)
        complete_in[pos] = pasts.issubset(complete_pos)

        futures = links.find_futures(pos)
        complete_out[pos] = futures.issubset(complete_pos)

    return complete_in, complete_out



def get_flows_and_energies(previous_pos, current_pos, local_links, experiment, integrate_ignore_penalty = False, complete_graph=True, ignore_penalty = 2):
    """Get the allowed integer flows and associated energies for every element in the subset"""

    # for every position generate unique key
    complete_set = current_pos.union(previous_pos)
    key_dict = dict(zip(complete_set, range(len(complete_set))))

    # Characterize all possible links, (dis)appearances, and divisions
    links_out = []
    links_in = []
    energies = []
    flow = []
    types = []

    # add link energies and flows
    for link in local_links:
        energies.append(experiment.link_data.get_link_data(link[0], link[1], data_name='link_penalty'))
        links_out.append(key_dict[link[0]])
        links_in.append(key_dict[link[1]])
        flow.append(1)
        types.append('link')

    # add energy associated with not using a position in the early timepoint
    for pos in previous_pos:
        if complete_graph:
            futures = experiment.links.find_futures(pos)
            outside_futures = futures.difference(complete_set)

            outside_energies = []
            for future in outside_futures:
                outside_energies.append(experiment.link_data.get_link_data(pos, future, data_name='link_penalty'))

            outside_energies.append(experiment.positions.get_position_data(pos, data_name='disappearance_penalty'))

            if integrate_ignore_penalty: #New element
                outside_energies.append(ignore_penalty)

            energies.append(combine_events(outside_energies))
            #print(combine_events(outside_energies))

        elif integrate_ignore_penalty:
            combined = combine_events([ignore_penalty, experiment.positions.get_position_data(pos, data_name='disappearance_penalty')])
            energies.append(combined)
        else:
            energies.append(experiment.positions.get_position_data(pos, data_name='disappearance_penalty'))

        # these elements form self-pointers in the graph
        links_out.append(key_dict[pos])
        links_in.append(key_dict[pos])
        flow.append(1)
        types.append('previous_position')

        # Add divisions
        division_penalty = experiment.positions.get_position_data(pos, data_name='division_penalty')
        if division_penalty < (ignore_penalty +1):
            energies.append(division_penalty)
            # these elements form self-pointers in the graph
            links_out.append(key_dict[pos])
            links_in.append(key_dict[pos])
            # divisions have negative flow, to allow for an extra link on that node
            flow.append(-1)
            types.append('division')

        #if use_ignore_penalty:
            #energies.append(ignore_penalty)
            #links_out.append(key_dict[pos])
            #links_in.append(None)
            #flow.append(1)

    # add energy associated with not using a position in the later timepoint
    for pos in current_pos:
        if complete_graph:
            futures = experiment.links.find_pasts(pos)
            outside_futures = futures.difference(complete_set)

            outside_energies = []
            for future in outside_futures:
                outside_energies.append(experiment.link_data.get_link_data(pos, future, data_name='link_penalty'))

            outside_energies.append(experiment.positions.get_position_data(pos, data_name='disappearance_penalty'))

            if integrate_ignore_penalty: #New element
                outside_energies.append(ignore_penalty)

            energies.append(combine_events(outside_energies))

        elif integrate_ignore_penalty:
            combined = combine_events([experiment.positions.get_position_data(pos, data_name='appearance_penalty'), ignore_penalty])
            energies.append(combined)
        else:
            energies.append(experiment.positions.get_position_data(pos, data_name='appearance_penalty'))

        links_out.append(key_dict[pos])
        links_in.append(key_dict[pos])
        flow.append(1)
        types.append('current_position')

        #if use_ignore_penalty:
            #energies.append(ignore_penalty)
            #links_out.append(None)
            #links_in.append(key_dict[pos])
            #flow.append(1)

    return np.array(links_out), np.array(links_in), np.array(flow), np.array(energies), key_dict, np.array(types)


def build_constraint_matrix(previous_pos, current_pos, links_out, links_in, flow,  key_dict, complete_in, complete_out, complete_graph=False):
    """Constructs a matrix A, such that A*x=0, if x is a set of compatible flows. It also outputs per contraint if the flows have to match up exactly.
    This does not have to be the case if not all links associated with a node are part of the subset."""

    # build constraint matrix
    constraint_matrix = [np.zeros(len(links_out))]
    exact_match = [False]

    # constraints associated with the fact that only one or two links can emerge from a single node
    for pos in previous_pos:
        link_out = key_dict[pos]
        if complete_graph and (np.sum(links_out[links_in == link_out] == link_out) == 1):
            exact_match.append(True)
        else:
            exact_match.append(complete_out[pos])
        constraint_matrix.append((links_out == link_out) * flow)

    # constraints associated with the fact that only one link can flow into a single node
    for pos in current_pos:
        if complete_graph:
            exact_match.append(True)
        else:
            exact_match.append(complete_in[pos])
        link_in = key_dict[pos]
        constraint_matrix.append((links_in == link_in) * flow)

    constraint_matrix = np.stack(constraint_matrix, axis=-1)
    exact_match = np.array(exact_match)

    return constraint_matrix, exact_match


def construct_microstates(links_out, links_in, types, complete_graph= False):
    """Builds all the vectors encoding possible flow states that have to checked"""

    independents = []
    options = np.eye(len(links_out))

    # builds all vectors that encode a single input link per node in the later timepoint
    for link_in in np.unique(links_in):
        subset = np.array(options[links_in==link_in, :])

        rows = np.split(subset, subset.shape[0])

        if (not complete_graph) or (np.sum(types[links_in==link_in]=='previous_position')>0):
            rows.append(np.zeros(rows[0].shape))

        if (np.sum(types[links_in==link_in]=='division')==1):
            # encodes an disappearance + a division
            rows.append(links_in==link_in)

        independents.append(rows)

    # combine all these vectors to form a set containing all possible flow configurations (but which are not all possible)
    possibilities = []
    for combination in itertools.product(*independents):
        possibility = [sum(x) for x in zip(*combination)]  # Was np.sum(list(combination), axis=0), but that doesn't work with modern numpy

        possibilities.append(np.squeeze(possibility))

    return np.stack(possibilities, axis=0)


def marginalization(energies, constraint_matrix, constraint_matrix_link, exact_match, exact_match_link, microstates=None):
    """Performs marginalization"""

    # construct all microstates (innefficenty) if none were given
    if microstates is None:
        possibilities = list(itertools.product([0, 1], repeat=len(energies)))
        possibilities = np.stack(possibilities, axis=0)
    else:
        possibilities = microstates

    # get energy per microstate
    total_energy = np.matmul(possibilities, energies)

    # check if the flows are allowed
    total_flows = np.matmul(possibilities, constraint_matrix)
    allowed = (np.sum(total_flows > 1, axis=-1) == 0)  & (np.sum((total_flows != 1) * exact_match, axis=-1) == 0)
    total_probability = np.sum(10**(-total_energy[allowed]))

    # check which flows are allowed if the link of interest is part of the solution
    total_flows = np.matmul(possibilities, constraint_matrix_link)
    allowed = (np.sum(total_flows > 1, axis=-1) == 0) & (np.sum((total_flows != 1) * exact_match_link, axis=-1) == 0)
    total_probability_link = np.sum(10**(-total_energy[allowed]))

    #marginalize
    return total_probability_link/total_probability


def local_marginalization(position1, position2, experiment, steps=3, verbose = False, complete_graph=False, scale=1):
    """Constructs subgraph, retrieves flows and energies and performs marginalization"""

    if position1.time_point_number() > position2.time_point_number():
        position1, position2 = position2, position1

    # get local and temporal environment
    current_pos, previous_pos, local_links = find_local_set(position1, position2, experiment.links, steps=steps)

    # check if subgraph size is not too big for typical memory. Makes an educated guess about number (log2) of microstates that will need to be constructed.
    # If subset is to big lower the number of steps.
    while (np.log2(len(local_links)/len(current_pos)+1) * len(current_pos) + len(previous_pos)) > 16:

        if steps == 0:
            if verbose:
             print('do minimal marginalization instead')
            return minimal_marginalization(position1, position2, experiment)

        steps = steps-1
        if verbose:
         print('environment is too big')
        current_pos, previous_pos, local_links = find_local_set(position1, position2, experiment.links, steps=steps)

    # which cells are fully connected?
    complete_in, complete_out = check_completeness(current_pos.union(previous_pos), experiment.links)

    # set up equation
    links_out, links_in, flow, energies, key_dict, types = get_flows_and_energies(previous_pos, current_pos, local_links, experiment, complete_graph=complete_graph)

    # scale the energies
    energies = energies*scale

    # build constraint matrix
    constraint_matrix, exact_match = build_constraint_matrix(previous_pos, current_pos, links_out, links_in, flow,
                                                             key_dict, complete_in, complete_out, complete_graph=complete_graph)

    # further constrain on existence of the link
    constraint_matrix_link = np.array(constraint_matrix)
    exact_match_link = np.array(exact_match)

    link_id = np.where((links_in == key_dict[position2]) & (links_out == key_dict[position1]))
    constraint_matrix_link[link_id, 0] = 1
    exact_match_link[0] = True

    # perform marginalization
    return marginalization(energies, constraint_matrix, constraint_matrix_link, exact_match, exact_match_link,
                           microstates=construct_microstates(links_out, links_in, types, complete_graph=complete_graph ))


def energy_to_prob(energy):
    return 10**-energy/(1+10**-energy)


def prob_to_energy(prob):
    epsilon = 10**-10
    return np.log10((1-prob+epsilon)/(prob+epsilon))

def combine_events(energies):
    """combines mutually exclusive events"""
    probs = energy_to_prob(np.array(energies))
    prob = 1-np.product(1-probs)

    return prob_to_energy(prob)