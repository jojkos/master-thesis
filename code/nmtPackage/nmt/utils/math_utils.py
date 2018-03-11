# coding: utf-8


def incremental_average(old_value, added_value, n):
    """

    Args:
        old_value: computed average so far up to step n
        added_value: new value to be added to the average
        n: time step of the average (first is 1)

    Returns:

    """
    return old_value + ((added_value - old_value) / n)
