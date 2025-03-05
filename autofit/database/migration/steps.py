from autofit.database.migration import Step, Migrator

steps = [
    Step(
        "ALTER TABLE fit ADD name VARCHAR;", "ALTER TABLE fit ADD path_prefix VARCHAR;"
    ),
    Step(
        "CREATE TABLE named_instance (id INTEGER NOT NULL , name VARCHAR, fit_id VARCHAR, PRIMARY KEY (id), FOREIGN KEY (fit_id) REFERENCES fit (id));"
    ),
    Step("ALTER TABLE fit ADD max_log_likelihood FLOAT;"),
    Step(
        "CREATE TABLE json (id INTEGER NOT NULL, name VARCHAR, string VARCHAR, fit_id VARCHAR, PRIMARY KEY (id), FOREIGN KEY (fit_id) REFERENCES fit (id));"
    ),
    Step(
        "CREATE TABLE array (id INTEGER NOT NULL, name VARCHAR, bytes BLOB, _dtype VARCHAR, _shape VARCHAR, array_type VARCHAR, fit_id VARCHAR, PRIMARY KEY (id), FOREIGN KEY (fit_id) REFERENCES fit (id));"
    ),
    Step(
        "CREATE TABLE hdu (id INTEGER NOT NULL, array_type VARCHAR, _header VARCHAR, PRIMARY KEY (id), FOREIGN KEY (id) REFERENCES array (id));"
    ),
    Step(
        "ALTER TABLE object ADD COLUMN latent_variables_for_id INTEGER;",
    ),
    Step(
        "ALTER TABLE object RENAME COLUMN latent_variables_for_id TO latent_samples_for_id;",
    ),
    Step(
        "CREATE TABLE fits (id INTEGER NOT NULL, name VARCHAR, fit_id VARCHAR, PRIMARY KEY (id), FOREIGN KEY (fit_id) REFERENCES fit (id));"
    ),
    Step(
        "ALTER TABLE hdu ADD COLUMN fits_id INTEGER;",
    ),
    Step(
        "ALTER TABLE hdu ADD COLUMN is_primary BOOLEAN;",
    ),
]

migrator = Migrator(*steps)
