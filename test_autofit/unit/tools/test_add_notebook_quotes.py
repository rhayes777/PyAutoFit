from autofit.tools.add_notebook_quotes import add_notebook_quotes


def test():

    lines = [
        '"""',
        "docs",
        '"""',
        "code",
        '"""',
        "docs",
        '"""'
    ]

    lines = add_notebook_quotes(lines)

    assert lines[0] == "# %%"
    assert lines[1] == "\n"
    assert lines[2] == "'''\n"
    assert lines[3] == "docs"
    assert lines[4] == "'''"
    assert lines[5] == "\n\n"
    assert lines[6] == "# %%\n"
    assert lines[7] == "code"
