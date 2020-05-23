"""
Answer queries regarding the optimal decision rule in various situations.


Author: Erel Segal-Halevi
Since:  2020-04
"""

import numpy as np
from powerset import powerset

def logodds(expertise_level: float):
    return np.log( expertise_level  / (1-expertise_level))


def is_minority_decisiveness_optimal(sorted_expertise_levels:list, minority_size:int=None)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is decisive in the optimal decision rule.
    "decisive" means that, if ALL members of minority agree, then their opinion is accepted.
    >>> is_minority_decisiveness_optimal(np.array([0.8]))
    False
    >>> is_minority_decisiveness_optimal(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisiveness_optimal(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_minority_decisiveness_optimal(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisiveness_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    True
    >>> is_minority_decisiveness_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), minority_size=1)
    False
    >>> is_minority_decisiveness_optimal(np.array([0.99, 0.9, 0.6, 0.6, 0.6]), minority_size=1)
    True
    >>> is_minority_decisiveness_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), minority_size=3)
    True
    """
    if minority_size is None:
        committee_size = len(sorted_expertise_levels)
        minority_size = int( (committee_size-1)/2 )
    weights = logodds(sorted_expertise_levels)
    half_total_weight = sum(weights)/2
    minority_weight = sum(weights[0:minority_size])
    return minority_weight > half_total_weight


def is_minority_tyranny_optimal(sorted_expertise_levels:list)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is tyrannic in the optimal decision rule.
    "tyrannic" means that the decision is accepted only by a vote within the minority, ignoring the majority altogether.
    NOTE: "tyrannic" implies "decisive", but not vice-versa.
    >>> is_minority_tyranny_optimal_b(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_tyranny_optimal_b(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_minority_tyranny_optimal_b(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_minority_tyranny_optimal_b(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    False
    >>> is_minority_tyranny_optimal_b(np.array([0.99, 0.99, 0.6, 0.6, 0.6]))
    False
    >>> is_minority_tyranny_optimal_b(np.array([0.99, 0.9, 0.6, 0.6, 0.6]))
    True
    """
    committee_size = len(sorted_expertise_levels)
    minority_size = int((committee_size - 1) / 2)
    weights = logodds(sorted_expertise_levels)
    minority_weights = weights[0:minority_size]
    sum_minority_weights = sum(minority_weights)
    sum_majority_weights = sum(weights[minority_size:])
    for minority_subset in powerset(minority_weights[1:]):  # Loop over all unordered partitions of the minority
        weight_difference_in_minority = np.abs(sum_minority_weights - 2*sum(minority_subset))
        if sum_majority_weights > weight_difference_in_minority:
            return False # the majority is essential in at least one case
    return True # the majority is never essential




def is_minority_tyranny_optimal_OLD(sorted_expertise_levels:list)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is tyrannic in the optimal decision rule.
    "tyrannic" means that the decision is accepted only by a vote within the minority, ignoring the majority altogether.
    NOTE: "tyrannic" implies "decisive", but not vice-versa.
    >>> is_minority_tyranny_optimal(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_tyranny_optimal(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_minority_tyranny_optimal(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_minority_tyranny_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    False
    >>> is_minority_tyranny_optimal(np.array([0.99, 0.99, 0.6, 0.6, 0.6]))
    False
    >>> is_minority_tyranny_optimal(np.array([0.99, 0.9, 0.6, 0.6, 0.6]))
    True
    """
    committee_size = len(sorted_expertise_levels)
    weights = logodds(sorted_expertise_levels)
    half_total_weight = sum(weights)/2
    if committee_size <= 3:  # minority size = 1.
        rule_1_00_optimal = weights[0] > half_total_weight
        return rule_1_00_optimal
    elif committee_size <= 5:  # minority size = 2.
        rule_10_000_optimal = weights[0] > half_total_weight
        return rule_10_000_optimal
    elif committee_size <= 7:  # minority size = 3.
        rule_100_0000_optimal = weights[0] > half_total_weight
        rule_111_0000_optimal = weights[1]+weights[2] > half_total_weight
        return rule_100_0000_optimal or rule_111_0000_optimal
    elif committee_size <= 9:  # minority size = 4.
        rule_1000_00000_optimal = weights[0] > half_total_weight
        rule_1110_00000_optimal = weights[1] + weights[2] > half_total_weight
        rule_2111_00000_optimal = (weights[0] + weights[3] > half_total_weight) \
                                  or (weights[1] + weights[2] + weights[3] > half_total_weight)
        return rule_1000_00000_optimal or rule_1110_00000_optimal or rule_2111_00000_optimal
    elif committee_size <= 11:  # minority size = 5.
        rule_10000_000000_optimal = weights[0] > half_total_weight
        rule_11100_000000_optimal = weights[1] + weights[2] > half_total_weight
        rule_11111_000000_optimal = weights[2] + weights[3] + weights[4] > half_total_weight
        rule_21110_000000_optimal = (weights[0] + weights[3] > half_total_weight) \
                                    or (weights[1] + weights[2] + weights[3] > half_total_weight)
        rule_22111_000000_optimal = (weights[0] + weights[1] > half_total_weight) \
                                    or (weights[1] + weights[3] + weights[4] > half_total_weight)
        rule_32211_000000_optimal = (weights[0] + weights[2] > half_total_weight) \
                                    or (weights[0] + weights[3] + weights[4] > half_total_weight) \
                                    or (weights[1] + weights[2] + weights[4] > half_total_weight)
        rule_31111_000000_optimal = (weights[0] + weights[4] > half_total_weight) \
                                    or (weights[1] + weights[2] + weights[3] + weights[4] > half_total_weight)
        return rule_10000_000000_optimal or rule_11100_000000_optimal or rule_11111_000000_optimal \
               or rule_21110_000000_optimal or rule_22111_000000_optimal or rule_32211_000000_optimal or rule_31111_000000_optimal
    else:
        raise ValueError("Committee sizes larger than 11 are not supported")



def fraction_majority_correct(sorted_expertise_levels:list, num_of_decisions:int)->float:
    """
    Estimate the probability in which the majority rule accepts the correct decision.

    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the majority rule accepts the correct decision.
    >>> f = fraction_majority_correct(np.array([0.8]), 1000)
    >>> np.abs(f-0.8) < 0.05
    True
    >>> f = fraction_majority_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f-0.896) < 0.05
    True
    >>> f = fraction_majority_correct(np.array([0.8, 0.8, 0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f-0.946) < 0.05
    True
    >>> f = fraction_majority_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f-.792) < 0.05
    True
    >>> f = fraction_majority_correct(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), 1000)
    >>> np.abs(f-.854) < 0.05
    True
    """
    committee_size = len(sorted_expertise_levels)
    majority_size = int(np.ceil(committee_size/2))
    num_rule_correct = 0
    for _ in range(num_of_decisions):
        num_correct = 0
        for level in sorted_expertise_levels:
            is_voter_correct = np.random.random() < level
            num_correct += is_voter_correct
        is_majority_correct = (num_correct >= majority_size)
        num_rule_correct += is_majority_correct
    return num_rule_correct / num_of_decisions


def fraction_expert_correct(sorted_expertise_levels:list, num_of_decisions:int)->float:
    """
    Estimate the probability in which the expert rule accepts the correct decision.

    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the expert rule accepts the correct decision.
    >>> f = fraction_expert_correct(np.array([0.8]), 1000)
    >>> np.abs(f-0.8) < 0.05
    True
    >>> f = fraction_expert_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f-0.8) < 0.05
    True
    >>> f = fraction_expert_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f-0.9) < 0.05
    True
    """
    num_rule_correct = 0
    for _ in range(num_of_decisions):
        level = sorted_expertise_levels[0]
        is_expert_correct = np.random.random() < level
        num_rule_correct += is_expert_correct
    return num_rule_correct / num_of_decisions



def fraction_compromise_correct(sorted_expertise_levels:list, num_of_decisions:int)->float:
    """
    Estimate the probability in which the following decision rule accepts the correct decision:
        If the minority of experts all agree - accept their opinion;
        otherwise - accept the majority opinion.

    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the above-described compromise rule accepts the correct decision.
    >>> f = fraction_compromise_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f-0.8) < 0.05
    True
    >>> f = fraction_compromise_correct(np.array([0.8, 0.8, 0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f-0.918) < 0.05
    True
    >>> f = fraction_compromise_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f-0.9) < 0.05
    True
    >>> f = fraction_compromise_correct(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), 1000)
    >>> np.abs(f-0.918) < 0.05
    True
    """
    committee_size = len(sorted_expertise_levels)
    minority_size = int((committee_size - 1) / 2)
    majority_size = committee_size - minority_size
    num_rule_correct = 0
    for _ in range(num_of_decisions):
        num_minority_correct = 0
        num_correct = 0
        for level in sorted_expertise_levels[0:minority_size]:
            is_voter_correct = np.random.random() < level
            num_minority_correct += is_voter_correct
            num_correct += is_voter_correct
        for level in sorted_expertise_levels[minority_size:]:
            is_voter_correct = np.random.random() < level
            num_correct += is_voter_correct
        if num_minority_correct==minority_size: # minority agrees on correct decision
            is_rule_correct = True
        elif num_minority_correct==0: # minority agrees on incorrect decision
            is_rule_correct = False
        else:
            is_rule_correct = (num_correct >= majority_size)
        num_rule_correct += is_rule_correct
    return num_rule_correct / num_of_decisions


def fraction_minority_colluding(sorted_expertise_levels:list, num_of_decisions:int, minority_size:int=None)->float:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make.
    :return: the empirical fraction of decisions in which the minority of experts agree.
    >>> fraction_minority_colluding(np.array([0.7, 0.8, 0.9]), 1000)
    1.0
    >>> f = fraction_minority_colluding(np.array([0.9, 0.9, 0.9, 0.9, 0.9]), 10000)
    >>> np.abs(f-0.81-0.01) < 0.05
    True
    >>> f = fraction_minority_colluding(np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]), 10000)
    >>> np.abs(f-0.729-0.001) < 0.05
    True
    """
    if minority_size is None:
        committee_size = len(sorted_expertise_levels)
        minority_size = int( (committee_size-1)/2 )
    num_minority_colluding = 0
    for _ in range(num_of_decisions):
        expert_vote = (np.random.random() < sorted_expertise_levels[0])
        is_minority_colluding = True
        for level in sorted_expertise_levels[1:minority_size]:
            if (np.random.random() < level)!=expert_vote:
                is_minority_colluding = False
                break
        num_minority_colluding += is_minority_colluding
    return num_minority_colluding / num_of_decisions


if __name__ == "__main__":
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print ("{} failures, {} tests".format(failures,tests))
