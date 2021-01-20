from typing import Iterable


def add_notebook_quotes(
        lines: Iterable[str]
):
    """
    Add %% above and below docs quotes with triple quotes.

    Used for conversion to ipynb notebooks

    Parameters
    ----------
    lines
        An iterable of lines loaded from a notebook file

    Returns
    -------
    Lines with %% inserted before and after docs
    """
    out = list()
    is_in_quotes = False

    for line in lines:
        if line.startswith('"""'):
            if is_in_quotes:
                out.extend([line, "# %%"])
            else:
                out.extend(["# %%", "\n", line])

            is_in_quotes = not is_in_quotes
        else:
            out.append(line)

    return out
