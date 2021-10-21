import ast


def test_parse(
        examples_directory,
        eden_output_directory
):
    with open(examples_directory / "prior.py") as f:
        parsed = ast.parse(f.read())

    print(parsed)
