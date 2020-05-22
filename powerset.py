#!python3

"""
A powerset recipe from https://docs.python.org/3/library/itertools.html#itertools-recipes
"""

from itertools import chain, combinations

def powerset(iterable):
    """
    >>> list(powerset([1,2,3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

