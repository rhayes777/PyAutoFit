from autofit.tools.edenise import Import


def test_skip_shortcut(file):
    import_ = Import.parse_fragment(
        "from autoconf.tools.decorators import cached_property",
        parent=file
    )

    assert import_.target_string == """
from VIS_CTI_Autoconf.VIS_CTI_Tools.decorators import cached_property
"""
