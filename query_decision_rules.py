"""
Answer queries regarding the optimal decision rule in various situations.


Since:  2020-04
Author: Erel Segal-Halevi
"""

def is_minority_decisive_optimal(sorted_expertise_levels:list, minority_size:int=None)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is decisive in the optimal decision rule.
    "decisive" means that, if ALL members of minority agree, then their opinion is accepted.
    >>> is_minority_decisive_optimal(np.array([0.8]))
    False
    >>> is_minority_decisive_optimal(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisive_optimal(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_minority_decisive_optimal(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisive_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    True
    >>> is_minority_decisive_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), minority_size=1)
    False
    >>> is_minority_decisive_optimal(np.array([0.99, 0.9, 0.6, 0.6, 0.6]), minority_size=1)
    True
    >>> is_minority_decisive_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), minority_size=3)
    True
    """
    if minority_size is None:
        committee_size = len(sorted_expertise_levels)
        minority_size = int( (committee_size-1)/2 )
    weights = logodds(sorted_expertise_levels)
    half_total_weight = sum(weights)/2
    minority_weight = sum(weights[0:minority_size])
    return minority_weight > half_total_weight


def is_meritocracy_optimal(sorted_expertise_levels:list)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is tyrannic in the optimal decision rule.
    "tyrannic" means that the decision is accepted only by a vote within the minority, ignoring the majority altogether.
    NOTE: that "tyrannic" implies "decisive", but not vice-versa.
    >>> is_meritocracy_optimal(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_meritocracy_optimal(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_meritocracy_optimal(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_meritocracy_optimal(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    False
    >>> is_meritocracy_optimal(np.array([0.99, 0.99, 0.6, 0.6, 0.6]))
    False
    >>> is_meritocracy_optimal(np.array([0.99, 0.9, 0.6, 0.6, 0.6]))
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
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the majority rule accepts the correct decision.
    >>> f08 = fraction_majority_correct(np.array([0.8]), 1000)
    >>> np.abs(f08-0.8) < 0.05
    True
    >>> f08 = fraction_majority_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f08-0.896) < 0.05
    True
    >>> f08 = fraction_majority_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f08-.792) < 0.05
    True
    """
    committee_size = len(sorted_expertise_levels)
    num_majority_correct = 0
    for _ in range(num_of_decisions):
        num_correct = 0
        for level in sorted_expertise_levels:
            is_expert_correct = np.random.random() < level
            num_correct += is_expert_correct
        is_majority_correct = 2*num_correct >= committee_size
        num_majority_correct += is_majority_correct
    return num_majority_correct / num_of_decisions


def fraction_expert_correct(sorted_expertise_levels:list, num_of_decisions:int)->float:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the expert rule accepts the correct decision.
    >>> f08 = fraction_expert_correct(np.array([0.8]), 1000)
    >>> np.abs(f08-0.8) < 0.05
    True
    >>> f08 = fraction_expert_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f08-0.8) < 0.05
    True
    >>> f08 = fraction_expert_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f08-0.9) < 0.05
    True
    """
    num_expert_correct = 0
    for _ in range(num_of_decisions):
        level = sorted_expertise_levels[0]
        is_expert_correct = np.random.random() < level
        num_expert_correct += is_expert_correct
    return num_expert_correct / num_of_decisions

