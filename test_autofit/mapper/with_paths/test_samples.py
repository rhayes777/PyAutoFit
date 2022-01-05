import autofit as af


def test_sample():
    sample = af.Sample(
        log_likelihood=0,
        log_prior=0,
        weight=0,
        kwargs={
            ("centre",),
            ("intensity",),
            ("sigma",),
        }
    )
    with_paths = sample.with_paths([
        ("centre",)
    ])

    assert with_paths.kwargs == {
        ("centre",)
    }
