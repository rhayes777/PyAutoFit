def grid(fitness_function, no_dimensions, step_size):
    for arguments in make_lists(no_dimensions, step_size):
        fitness_function(arguments)


def make_lists(no_dimensions, step_size):
    if no_dimensions == 0:
        return [[]]

    sub_lists = make_lists(no_dimensions - 1, step_size)
    return [[step_size * value] + sub_list for value in range(0, int((1 / step_size) + 1)) for sub_list in sub_lists]
