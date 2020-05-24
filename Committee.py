"""
A Committee is a group of experts with various expertise levels.
Their goal is to arrive at the "correct" decision.

Author: Erel Segal-Halevi
Since:  2020-05
"""

import numpy as np
from powerset import powerset
from statistics import mean

class Committee:
    def __init__(self, sorted_expertise_levels:list):
        self.sorted_expertise_levels = sorted_expertise_levels
        self.weights = logodds(sorted_expertise_levels)
        self.half_total_weight = sum(self.weights)/2
        self.committee_size = len(sorted_expertise_levels)
        self.minority_size = int(np.floor((self.committee_size - 1) / 2))
        self.majority_size = self.committee_size - self.minority_size

    @staticmethod
    def random_expertise_levels(mean:float, std:float, size:int):
        return Committee(random_expertise_levels(mean, std, size))

    def is_minority_decisiveness_optimal(self, minority_size:int=None)->bool:
        """
        :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
        :return: whether the minority of experts is decisive in the optimal decision rule.
        "decisive" means that, if ALL members of minority agree, then their opinion is accepted.
        >>> Committee(np.array([0.8])).is_minority_decisiveness_optimal()
        False
        >>> Committee(np.array([0.8, 0.8, 0.8])).is_minority_decisiveness_optimal()
        False
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_minority_decisiveness_optimal()
        True
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_minority_decisiveness_optimal()
        False
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_minority_decisiveness_optimal()
        True
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_minority_decisiveness_optimal(minority_size=1)
        False
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_minority_decisiveness_optimal(minority_size=1)
        True
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_minority_decisiveness_optimal(minority_size=3)
        True
        """
        if minority_size is None:
            minority_size = self.minority_size
        minority_weight = sum(self.weights[0:minority_size])
        return minority_weight > self.half_total_weight


    def is_minority_tyranny_optimal(self)->bool:
        """
        :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
        :return: whether the minority of experts is tyrannic in the optimal decision rule.
        "tyrannic" means that the decision is accepted only by a vote within the minority, ignoring the majority altogether.
        NOTE: "tyrannic" implies "decisive", but not vice-versa.
        >>> Committee(np.array([0.8, 0.8, 0.8])).is_minority_tyranny_optimal()
        False
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_minority_tyranny_optimal()
        True
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_minority_tyranny_optimal()
        False
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_minority_tyranny_optimal()
        False
        >>> Committee(np.array([0.99, 0.99, 0.6, 0.6, 0.6])).is_minority_tyranny_optimal()
        False
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_minority_tyranny_optimal()
        True
        """
        minority_weights = self.weights[0:self.minority_size]
        sum_minority_weights = sum(minority_weights)
        sum_majority_weights = sum(self.weights[self.minority_size:])
        for minority_subset in powerset(minority_weights[1:]):  # Loop over all unordered partitions of the minority
            weight_difference_in_minority = np.abs(sum_minority_weights - 2*sum(minority_subset))
            if sum_majority_weights > weight_difference_in_minority:
                return False # the majority is essential in at least one case
        return True # the majority is never essential


    def fraction_of_correct_decisions(self, rule, num_of_decisions:int)->float:
        """
        Estimate the probability in which the given rule accepts the correct decision.

        :param rule - a function that returns a bool and calculates a random outcome of the rule (True for correct, False for incorrect)
        :param num_of_decisions - number of times to use the rule for making a decision.
        :return: the empirical fraction of decisions in which the majority rule accepts the correct decision.
        """
        return sum([rule(self) for _ in range(num_of_decisions)]) / num_of_decisions


    def optimal_weighted_rule(self)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of the optimal weighted rule.
        :return: 0 (wrong decision) or 1 (correct decision).

        >>> f = Committee(np.array([0.8])).fraction_of_correct_decisions(Committee.optimal_weighted_rule, 1000)
        >>> np.abs(f-0.8) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.optimal_weighted_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.optimal_weighted_rule, 1000)
        >>> np.abs(f-0.94) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.6, 0.6])).fraction_of_correct_decisions(Committee.optimal_weighted_rule, 1000)
        >>> np.abs(f-.9) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).fraction_of_correct_decisions(Committee.optimal_weighted_rule, 1000)
        >>> np.abs(f-.93) < 0.05
        True
        """
        weight_correct = 0
        for (level,weight) in zip(self.sorted_expertise_levels,self.weights):
            is_voter_correct = np.random.random() < level
            weight_correct += weight*is_voter_correct
        return (weight_correct >= self.half_total_weight)


    def simple_majority_rule(self)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of the majority rule.
        :return: 0 (wrong decision) or 1 (correct decision).

        >>> f = Committee(np.array([0.8])).fraction_of_correct_decisions(Committee.simple_majority_rule, 1000)
        >>> np.abs(f-0.8) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.simple_majority_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.simple_majority_rule, 1000)
        >>> np.abs(f-0.94) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.6, 0.6])).fraction_of_correct_decisions(Committee.simple_majority_rule, 1000)
        >>> np.abs(f-.792) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).fraction_of_correct_decisions(Committee.simple_majority_rule, 1000)
        >>> np.abs(f-.854) < 0.05
        True
        """
        num_correct = 0
        for level in self.sorted_expertise_levels:
            is_voter_correct = np.random.random() < level
            num_correct += is_voter_correct
        return (num_correct >= self.majority_size)


    def expert_rule(self)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of the majority rule.
        :return: 0 (wrong decision) or 1 (correct decision).

        >>> f = Committee(np.array([0.8])).fraction_of_correct_decisions(Committee.expert_rule, 1000)
        >>> np.abs(f-0.8) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.expert_rule, 1000)
        >>> np.abs(f-0.8) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.6, 0.6])).fraction_of_correct_decisions(Committee.expert_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        """
        level = self.sorted_expertise_levels[0]
        return np.random.random() < level

    def compromise_noweights_rule(self)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of the compromise rule.
        :return: 0 (wrong decision) or 1 (correct decision).

        >>> f = Committee(np.array([0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.compromise_noweights_rule, 1000)
        >>> np.abs(f-0.8) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.compromise_noweights_rule, 1000)
        >>> np.abs(f-0.918) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.6, 0.6])).fraction_of_correct_decisions(Committee.compromise_noweights_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).fraction_of_correct_decisions(Committee.compromise_noweights_rule, 1000)
        >>> np.abs(f-0.918) < 0.05
        True
        """
        num_correct = 0
        num_minority_correct = 0
        for level in self.sorted_expertise_levels[0:self.minority_size]:
            is_voter_correct = np.random.random() < level
            num_minority_correct += is_voter_correct
            num_correct += is_voter_correct
        for level in self.sorted_expertise_levels[self.minority_size:]:
            is_voter_correct = np.random.random() < level
            num_correct += is_voter_correct
        if num_minority_correct == self.minority_size:  # minority agrees on correct decision
            return True
        elif num_minority_correct == 0:  # minority agrees on incorrect decision
            return False
        else:
            return (num_correct >= self.majority_size)

    def compromise_weights_rule(self)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of the compromise rule.
        :return: 0 (wrong decision) or 1 (correct decision).

        >>> f = Committee(np.array([0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.compromise_weights_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.compromise_weights_rule, 1000)
        >>> np.abs(f-0.94) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.6, 0.6])).fraction_of_correct_decisions(Committee.compromise_weights_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).fraction_of_correct_decisions(Committee.compromise_weights_rule, 1000)
        >>> np.abs(f-0.918) < 0.05
        True
        """
        if sum(self.weights[0:self.minority_size]) > self.half_total_weight:
            return self.compromise_noweights_rule()
        else:
            return self.simple_majority_rule()


    def compromise_strongmajority_rule(self)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of a strong-majority compromise rule.
        :return: 0 (wrong decision) or 1 (correct decision).

        >>> f = Committee(np.array([0.8])).fraction_of_correct_decisions(Committee.compromise_strongmajority_rule, 1000)
        >>> np.abs(f-0.8) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.compromise_strongmajority_rule, 1000)
        >>> np.abs(f-0.9) < 0.05
        True
        >>> f = Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).fraction_of_correct_decisions(Committee.compromise_strongmajority_rule, 1000)
        >>> np.abs(f-0.94) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.6, 0.6])).fraction_of_correct_decisions(Committee.compromise_strongmajority_rule, 1000)
        >>> np.abs(f-.9) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).fraction_of_correct_decisions(Committee.compromise_strongmajority_rule, 1000)
        >>> np.abs(f-.93) < 0.05
        True
        """
        num_correct = weight_correct = 0
        for (level,weight) in zip(self.sorted_expertise_levels,self.weights):
            is_voter_correct = np.random.random() < level
            num_correct += is_voter_correct
            weight_correct += weight*is_voter_correct
        if num_correct >= self.majority_size+1:  # strong majority correct
            return True
        elif num_correct < self.majority_size-1:  # strong majority incorrect
            return False
        else:
            return weight_correct >= self.half_total_weight


    def fraction_minority_colluding(self, num_of_decisions:int)->float:
        """
        :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
        :param num of decisions to make.
        :return: the empirical fraction of decisions in which the minority of experts agree.
        >>> Committee(np.array([0.7, 0.8, 0.9])).fraction_minority_colluding(1000)
        1.0
        >>> f = Committee(np.array([0.9, 0.9, 0.9, 0.9, 0.9])).fraction_minority_colluding(10000)
        >>> np.abs(f-0.81-0.01) < 0.05
        True
        >>> f = Committee(np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])).fraction_minority_colluding(10000)
        >>> np.abs(f-0.729-0.001) < 0.05
        True
        """
        num_minority_colluding = 0
        for _ in range(num_of_decisions):
            expert_vote = (np.random.random() < self.sorted_expertise_levels[0])
            is_minority_colluding = True
            for level in self.sorted_expertise_levels[1:self.minority_size]:
                if (np.random.random() < level)!=expert_vote:
                    is_minority_colluding = False
                    break
            num_minority_colluding += is_minority_colluding
        return num_minority_colluding / num_of_decisions




### UTILITY FUNCTIONS


def logodds(expertise_level: float):
    return np.log( expertise_level  / (1-expertise_level))



from scipy.stats import truncnorm
def random_expertise_levels(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is a probability in [0.5,1],
    drawn from a truncated-normal distribution with the given `mean` and `std`.
    :return: an array of `size` random expretise levels, sorted from high to low.
    """
    scale = std
    loc = mean
    lower = 0.5
    upper = 1
    # a*scale + loc = lower
    # b*scale + loc = upper
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return -np.sort(-truncnorm.rvs(a, b, loc=loc, scale=scale, size=size))





if __name__ == "__main__":
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print ("{} failures, {} tests".format(failures,tests))
