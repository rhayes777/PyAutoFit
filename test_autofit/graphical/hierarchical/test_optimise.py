import autofit as af


def test_optimise(
        hierarchical_factor
):
    optimizer = af.DynestyStatic(
        maxcall=10
    )
    factor = hierarchical_factor.factors[0]

    optimizer.optimise(
        factor,
        factor.mean_field_approximation()
    )
