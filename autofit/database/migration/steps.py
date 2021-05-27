from autofit.database.migration import Step

steps = [
    Step(
        "ALTER TABLE fit ADD name VARCHAR;",
        "ALTER TABLE fit ADD path_prefix VARCHAR;"
    )
]
