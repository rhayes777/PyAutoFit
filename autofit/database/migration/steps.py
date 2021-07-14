from autofit.database.migration import Step, Migrator

steps = [
    Step(
        "ALTER TABLE fit ADD name VARCHAR;",
        "ALTER TABLE fit ADD path_prefix VARCHAR;"
    ),
    Step(
        "CREATE TABLE named_instance (id INTEGER NOT NULL , name VARCHAR, fit_id VARCHAR, PRIMARY KEY (id), FOREIGN KEY (fit_id) REFERENCES fit (id));"
    ),
    Step(
        "ALTER TABLE fit ADD max_log_likelihood FLOAT;"
    )
]

migrator = Migrator(*steps)
