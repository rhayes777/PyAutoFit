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

    assert add_notebook_quotes(
        lines
    ) == ['# %%', '\n',
          "'''\n",
          'docs',
          "'''",
          '\n\n',
          '# %%\n',
          'code',
          '# %%',
          '\n',
          "'''\n",
          'docs'
          ]
