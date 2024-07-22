#!python3

"""
A Committee is a group of experts with various expertise levels.
Their goal is to arrive at the "correct" decision.

Author: Erel Segal-Halevi
Since:  2020-05
"""

import numpy as np
from powerset import powerset

class Committee:
    def __init__(self, sorted_expertise_levels:list):
        self.sorted_expertise_levels = sorted_expertise_levels
        self.weights = logodds(sorted_expertise_levels)
        # print(self.weights)
        self.half_total_weight = sum(self.weights)/2
        self.committee_size = len(sorted_expertise_levels)
        self.minority_size = int(np.floor((self.committee_size - 1) / 2))
        self.majority_size = self.committee_size - self.minority_size

    def __str__(self):
        return "probabilities: {}\n      weights: {}".format(
            round3(self.sorted_expertise_levels), round3(self.weights))


    ### CHECK WHETHER THE OPTIMAL RULE IS DECISIVE / TYRANIC - BASED ON THE WEIGHTS

    def is_optimal_strong_democracy(self)->bool:
        """
        :return: whether the optimal decision rule is "strongly democratic", that is, equivalent to simple-majority.
        >>> Committee(np.array([0.8])).is_optimal_strong_democracy()
        True
        >>> Committee(np.array([0.8, 0.8, 0.8])).is_optimal_strong_democracy()
        True
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_optimal_strong_democracy()
        False
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_optimal_strong_democracy()
        True
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_strong_democracy()
        False
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_optimal_strong_democracy()
        False
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_strong_democracy()
        False
        """
        minority_weight = sum(self.weights[0:self.minority_size])
        return minority_weight <= self.half_total_weight


    def is_optimal_weak_democracy(self)->bool:
        """
        :return: whether the optimal decision rule is "weakly democratic", that is, gives positive weights to all voters.
        NOTE: strong democracy implies weak democracy, but not vice-versa.
        >>> Committee(np.array([0.8, 0.8, 0.8])).is_optimal_weak_democracy()
        True
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_optimal_weak_democracy()
        False
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_optimal_weak_democracy()
        True
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_weak_democracy()
        True
        >>> Committee(np.array([0.99, 0.99, 0.6, 0.6, 0.6])).is_optimal_weak_democracy()
        True
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_optimal_weak_democracy()
        False
        """
        majority_weights = self.weights[0:self.committee_size-1]
        sum_majority_weights = sum(majority_weights)
        least_weight = self.weights[-1]
        for majority_subset in powerset(majority_weights[1:]):  # Loop over all unordered partitions of the majority
            sum_majority_subset = sum(majority_subset)
            sum_complement_subset = sum_majority_weights - sum_majority_subset
            weight_difference_in_majority = np.abs(sum_majority_subset - sum_complement_subset)
            if least_weight > weight_difference_in_majority:
                # print("least_weight={} majority_subset={} sum_majority_subset={} sum_complement_subset={}".
                #       format(np.round(least_weight,3),np.round(majority_subset,3),np.round(sum_majority_subset,3),np.round(sum_complement_subset,3)))
                return True # the least-weight agent is essential in at least one case
        return False # the least-weight agent is never essential


    def is_optimal_weak_epistocracy(self)->bool:
        """
        :return: whether the optimal decision rule is "weakly epistocratic", that is, 
                 some voters have zero weight.

        NOTE: strong epistocracy (=tyrannic) implies weak epistocracy (=decisiveness), but not vice-versa.

        >>> Committee(np.array([0.8, 0.8, 0.8])).is_optimal_weak_epistocracy()
        False
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_optimal_weak_epistocracy()
        True
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_optimal_weak_epistocracy()
        False
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_weak_epistocracy()
        False
        >>> Committee(np.array([0.99, 0.99, 0.6, 0.6, 0.6])).is_optimal_weak_epistocracy()
        False
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_optimal_weak_epistocracy()
        True
        """
        return not self.is_optimal_weak_democracy()


    def is_optimal_strong_epistocracy(self)->bool:
        """
        :return: whether the optimal decision rule is "strongly epistocratic", that is, 
                 the minority of experts is tyrannic.
        "tyrannic" means that the decision is accepted only by a vote within the minority, ignoring the majority altogether.
        NOTE: "tyrannic" implies "decisive", but not vice-versa.
        >>> Committee(np.array([0.8, 0.8, 0.8])).is_optimal_strong_epistocracy()
        False
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_optimal_strong_epistocracy()
        True
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_optimal_strong_epistocracy()
        False
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_strong_epistocracy()
        False
        >>> Committee(np.array([0.99, 0.99, 0.6, 0.6, 0.6])).is_optimal_strong_epistocracy()
        False
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_optimal_strong_epistocracy()
        True
        """
        minority_weights = self.weights[0:self.minority_size]
        sum_minority_weights = sum(minority_weights)
        sum_majority_weights = sum(self.weights[self.minority_size:])
        for minority_subset in powerset(minority_weights[1:]):  # Loop over all unordered partitions of the minority
            sum_minority_subset = sum(minority_subset)
            sum_complement_subset = sum_minority_weights -sum_minority_subset
            weight_difference_in_minority = np.abs(sum_minority_subset - sum_complement_subset)
            if sum_majority_weights > weight_difference_in_minority:
                # print("sum_majority_weights={} minority_subset={} sum_minority_subset={} sum_complement_subset={}".
                #       format(np.round(sum_majority_weights,3),np.round(minority_subset,3),np.round(sum_minority_subset,3),np.round(sum_complement_subset,3)))
                return False    # the majority is essential in at least one case
        return True             # the majority is never essential



    def is_optimal_minority_decisiveness(self, minority_size:int=None)->bool:
        """
        :return: whether, in the optimal decision rule, the minority of experts is decisive.
        "decisive" means that, if ALL members of minority agree, then their opinion is accepted.

        :note the optimal rule is minority-decisive, iff it is NOT the simple majority rule.
        Hence, this function is the opposite of is_optimal_strong_democracy.

        >>> Committee(np.array([0.8])).is_optimal_minority_decisiveness()
        False
        >>> Committee(np.array([0.8, 0.8, 0.8])).is_optimal_minority_decisiveness()
        False
        >>> Committee(np.array([0.9, 0.6, 0.6])).is_optimal_minority_decisiveness()
        True
        >>> Committee(np.array([0.8, 0.8, 0.8, 0.8, 0.8])).is_optimal_minority_decisiveness()
        False
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_minority_decisiveness()
        True
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_minority_decisiveness(minority_size=1)
        False
        >>> Committee(np.array([0.99, 0.9, 0.6, 0.6, 0.6])).is_optimal_minority_decisiveness(minority_size=1)
        True
        >>> Committee(np.array([0.9, 0.9, 0.6, 0.6, 0.6])).is_optimal_minority_decisiveness(minority_size=3)
        True
        """
        if minority_size is None:
            minority_size = self.minority_size
        minority_weight = sum(self.weights[0:minority_size])
        return minority_weight > self.half_total_weight





    ### CHECK THE FRACTION OF CORRECT DECISIONS - BASED ON THE VOTE

    def vote(self):
        """
        Draw opinions at random by the expertise levels.
        :return: a vector of n votes, one per voter, ordered from most expert to least expert.
        True means "correct", False means "incorrect".
        """
        return [np.random.random() < level for level in self.sorted_expertise_levels]


    def fraction_of_correct_decisions(self, rule, num_of_decisions:int)->float:
        """
        Estimate the probability in which the given rule accepts the correct decision.

        :param rule - a function that returns a bool and calculates a random outcome of the rule (True for correct, False for incorrect)
        :param num_of_decisions - number of times to use the rule for making a decision.
        :return: the empirical fraction of decisions in which the majority rule accepts the correct decision.
        """
        return sum([rule(self, self.vote()) for _ in range(num_of_decisions)]) / num_of_decisions

    def optimal_weighted_rule(self, vote:list)->bool:
        """
        Calculate the decision of the optimal weighted rule by the given vote.
        :return: False (wrong decision) or True (correct decision).

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
        for (is_voter_correct,weight) in zip(vote, self.weights):
            weight_correct += weight*is_voter_correct
        return (weight_correct >= self.half_total_weight)


    def simple_majority_rule(self, vote:list)->bool:
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
        for is_voter_correct in vote:
            num_correct += is_voter_correct
        return (num_correct >= self.majority_size)

    def expert_rule(self, vote:list)->bool:
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
        return vote[0]

    def compromise_noweights_rule(self, vote:list)->bool:
        """
        Draw opinions at random by the expertise levels, and calculate the decision of a compromise rule:
             The rule accepts the decision of the elite if they agree; otherwise it accepts the decision of the majority.
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
        for is_voter_correct in vote[0:self.minority_size]:
            num_minority_correct += is_voter_correct
            num_correct += is_voter_correct
        for is_voter_correct in vote[self.minority_size:]:
            num_correct += is_voter_correct
        if num_minority_correct == self.minority_size:  # minority agrees on correct decision
            return True
        elif num_minority_correct == 0:  # minority agrees on incorrect decision
            return False
        else:                            # minority do not agree
            return (num_correct >= self.majority_size)   # select the majority decision

    def compromise_weights_rule(self, vote:list)->bool:
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
            return self.compromise_noweights_rule(vote)
        else:
            return self.simple_majority_rule(vote)


    def compromise_strongmajority_rule(self, vote:list)->bool:
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
        for (is_voter_correct,weight) in zip(vote, self.weights):
            num_correct += is_voter_correct
            weight_correct += weight*is_voter_correct
        if num_correct >= self.majority_size+1:  # strong majority correct
            return True
        elif num_correct < self.majority_size-1:  # strong majority incorrect
            return False
        else:  # use the optimal rule
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
    """
    >>> logodds(0.5)
    0.0
    >>> logodds(np.array([0.001, 0.999]))
    array([-6.90675478,  6.90675478])
    """
    try:
        return np.log( expertise_level  / (1-expertise_level))
    except FloatingPointError as ex:
        # print("expertise_level=",expertise_level)
        raise FloatingPointError(f"{ex}: expertise_level={expertise_level}")


def round3(l:list):
    return [np.round(x,3) for x in l]






if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

