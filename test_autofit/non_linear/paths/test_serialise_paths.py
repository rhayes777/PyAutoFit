from pathlib import Path

import autofit as af
from autoconf.dictable import to_dict, from_dict


def test_path_prefix():
    path_prefix = Path("test_path_prefix")
    paths = af.DirectoryPaths(
        path_prefix=path_prefix,
    )
    paths = from_dict(to_dict(paths))

    assert paths.path_prefix == path_prefix
