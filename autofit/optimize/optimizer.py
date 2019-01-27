def grid(fitness_function, no_dimensions, step_size):
    """
    Grid search using a fitness function over a given number of dimensions and a given step size between inclusive
    limits of 0 and 1.

    Parameters
    ----------
    fitness_function: function
        A function that takes a tuple of floats as an argument
    no_dimensions: int
        The number of dimensions of the grid search
    step_size: float
        The step size of the grid search

    Returns
    -------
    best_arguments: tuple[float]
        The tuple of arguments that gave the highest fitness
    """
    best_fitness = float("-inf")
    best_arguments = None

    for arguments in make_lists(no_dimensions, step_size):
        fitness = fitness_function(tuple(arguments))
        if fitness > best_fitness:
            best_fitness = fitness
            best_arguments = tuple(arguments)

    return best_arguments


def make_lists(no_dimensions, step_size, centre_steps=True):
    """
    Create a list of lists of floats covering every combination across no_dimensions of points of integer step size
    between 0 and 1 inclusive.

    Parameters
    ----------
    no_dimensions: int
        The number of dimensions, that is the length of the lists
    step_size: float
        The step size
    centre_steps: bool

    Returns
    -------
    lists: [[float]]
        A list of lists
    """
    if no_dimensions == 0:
        return [[]]

    sub_lists = make_lists(no_dimensions - 1, step_size, centre_steps=centre_steps)
    return [[step_size * value + (0.5 * step_size if centre_steps else 0)] + sub_list for value in
            range(0, int((1 / step_size))) for sub_list in sub_lists]
