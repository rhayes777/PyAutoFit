def grid(fitness_function, no_dimensions, step_size):
    for value in range(0, int(1 / step_size)):
        fitness_function((step_size * value,))
