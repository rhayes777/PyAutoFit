# (0, 0), (0, 1), (1, 0), (1, 1)


def grid(fitness_function, no_dimensions, step_size):
    for value in range(0, int(1 / step_size)):
        fitness_function((step_size * value,))


def make_lists(no_dimensions, step_size):
    if no_dimensions == 0:
        return [[]]

    sub_lists = list(make_lists(no_dimensions - 1, step_size))
    return [[step_size * value] + sub_list for value in range(0, int(1 / step_size)) for sub_list in sub_lists]


if __name__ == "__main__":
    print(make_lists(2, 0.2))
