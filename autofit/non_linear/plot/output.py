import matplotlib
from pathlib import Path
from os import path
from typing import List, Optional, Union

from autoconf import conf


def set_backend():
    backend = conf.get_matplotlib_backend()

    if backend not in "default":
        matplotlib.use(backend)

    try:
        hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
    except KeyError:
        hpc_mode = False

    if hpc_mode:
        matplotlib.use("Agg")


import matplotlib.pyplot as plt
import os


class Output:
    def __init__(
        self,
        path: Optional[Path] = None,
        filename: Optional[str] = None,
        format: Union[str, List[str]] = None,
        bypass: bool = False,
    ):
        """
        Sets how the figure or subplot is output, either by displaying it on the screen or writing it to hard-disk.

        This object wraps the following Matplotlib methods:

        - plt.show: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html
        - plt.savefig: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disk
        as a file.

        Parameters
        ----------
        path
            If the figure is output to hard-disk the path of the folder it is saved to.
        filename
            If the figure is output to hard-disk the filename used to save it.
        format
            The format of the output, 'show' displays on the computer screen, 'png' outputs to .png, 'fits' outputs to
            `.fits` format.
        bypass
            Whether to bypass the `plt.show` or `plt.savefig` methods, used when plotting a subplot.
        """
        if path is not None:
            path = Path(path)

        self.path = path

        if path is not None and path:
            os.makedirs(path, exist_ok=True)

        self.filename = filename
        self._format = format
        self.bypass = bypass

    @property
    def format(self) -> str:
        return self._format or "show"

    @property
    def format_list(self):
        if not isinstance(self.format, list):
            return [self.format]
        return self.format

    def to_figure(
        self,
        auto_filename: Optional[str] = None,
    ):
        """
        Output the figure, by either displaying it on the user's screen or to the hard-disk as a .png or .fits file.

        Parameters
        ----------
        structure
            The 2D array of image to be output, required for outputting the image as a fits file.
        """

        filename = auto_filename if self.filename is None else self.filename

        for format in self.format_list:
            if os.environ.get("PYAUTOARRAY_OUTPUT_MODE") == "1":
                return self.to_figure_output_mode(filename=filename)
            if not self.bypass:
                if format == "show":
                    plt.show()
                elif format == "png":
                    plt.savefig(self.path / f"{filename}.png")
                elif format == "pdf":
                    plt.savefig(self.path / f"{filename}.pdf")

    def subplot_to_figure(self, auto_filename=None):
        """
        Output a subplot figure, either as an image on the screen or to the hard-disk as a png or fits file.
        """

        filename = auto_filename if self.filename is None else self.filename

        for format in self.format_list:
            if os.environ.get("PYAUTOARRAY_OUTPUT_MODE") == "1":
                return self.to_figure_output_mode(filename=filename)
            if format == "show":
                plt.show()
            elif format == "png":
                plt.savefig(self.path / f"{filename}.png")
            elif format == "pdf":
                plt.savefig(self.path / f"{filename}.pdf")


    def to_figure_output_mode(self, filename: str):
        global COUNT

        try:
            COUNT += 1
        except NameError:
            COUNT = 0

        import sys

        script_name = path.split(sys.argv[0])[-1].replace(".py", "")

        output_path = path.join(os.getcwd(), "output_mode", script_name)
        os.makedirs(output_path, exist_ok=True)

        plt.savefig(
            path.join(output_path, f"{COUNT}_{filename}.png"),
        )
