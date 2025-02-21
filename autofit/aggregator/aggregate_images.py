import sys
from enum import Enum
from typing import Optional, List, Union
from pathlib import Path

from PIL import Image

from autofit.aggregator.search_output import SearchOutput
from autofit.aggregator.aggregator import Aggregator


class Subplot(Enum):
    """
    The subplots that can be extracted from the subplot_fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    Data = (0, 0)
    DataSourceScaled = (1, 0)
    SignalToNoiseMap = (2, 0)
    ModelImage = (3, 0)
    LensLightModelImage = (0, 1)
    LensLightSubtractedImage = (1, 1)
    SourceModelImage = (2, 1)
    SourcePlaneZoomed = (3, 1)
    NormalizedResidualMap = (0, 2)
    NormalizedResidualMapOneSigma = (1, 2)
    ChiSquaredMap = (2, 2)
    SourcePlaneNoZoom = (3, 2)


class SubplotFitImage:
    def __init__(self, image: Image.Image):
        """
        The subplot_fit image associated with one fit.

        Parameters
        ----------
        image
            The subplot_fit image.
        """
        self._image = image

        self._single_image_width = self._image.width // 4
        self._single_image_height = self._image.height // 3

    def image_at_coordinates(
        self,
        x: int,
        y: int,
    ) -> Image.Image:
        """
        Extract the image at the specified coordinates.

        Parameters
        ----------
        x
        y
            The integer coordinates of the plot (see Subplot).

        Returns
        -------
        The extracted image.
        """
        return self._image.crop(
            (
                x * self._single_image_width,
                y * self._single_image_height,
                (x + 1) * self._single_image_width,
                (y + 1) * self._single_image_height,
            )
        )


class AggregateImages:
    def __init__(
        self,
        aggregator: Aggregator,
    ):
        """
        Extracts images from the aggregator and combines them into one
        overall image.

        Parameters
        ----------
        aggregator
            The aggregator containing the fit results.
        """
        self._aggregator = aggregator

    def extract_image(
        self,
        *subplots: Union[Subplot, List[Image.Image]],
        subplot_width: Optional[int] = sys.maxsize,
    ) -> Image.Image:
        """
        Extract the images at the specified subplots and combine them into
        one overall image including those subplots for all fits in the
        aggregator.

        Parameters
        ----------
        subplots
            The subplots to extract.
        subplot_width
            Defines the width of each subplot in number of images.
            If this is greater than the number of subplots then it defaults to
            the number of subplots.
            If this is less than the number of subplots then it causes the
            images to wrap.

        Returns
        -------
        The combined image.
        """
        matrix = []
        for i, result in enumerate(self._aggregator):
            matrix.extend(
                self._matrix_for_result(
                    i,
                    result,
                    *subplots,
                    subplot_width=subplot_width,
                )
            )

        return self._matrix_to_image(matrix)

    def output_to_folder(
        self,
        folder: Path,
        *subplots: Union[Subplot, List[Image.Image]],
        subplot_width: Optional[int] = sys.maxsize,
        name: str = "name",
    ):
        """
        Output one subplot image for each fit in the aggregator.

        Parameters
        ----------
        folder
            The target folder in which to store the subplots.
        subplots
            The subplots to output.
        subplot_width
            Defines the width of each subplot in number of images.
            If this is greater than the number of subplots then it defaults to
            the number of subplots.
            If this is less than the number of subplots then it causes the
            images to wrap.
        name
            The attribute of each fit to use as the name of the output file.
        """
        folder.mkdir(exist_ok=True)

        for i, result in enumerate(self._aggregator):
            image = self._matrix_to_image(
                self._matrix_for_result(
                    i,
                    result,
                    *subplots,
                    subplot_width=subplot_width,
                )
            )
            image.save(folder / f"{getattr(result, name)}.png")

    @staticmethod
    def _matrix_for_result(
        i: int,
        result: SearchOutput,
        *subplots: Union[Subplot, List[Image.Image]],
        subplot_width: int = sys.maxsize,
    ) -> List[List[Image.Image]]:
        """
        Create a matrix of images each in the position they will be in the
        final image.

        Parameters
        ----------
        result
            The fit result.
        subplots
            The subplots to extract
        subplot_width
            The number of subplots to include in each row of the matrix.

        Returns
        -------
        The matrix of images.
        """
        subplot_fit_image = SubplotFitImage(result.image("subplot_fit"))
        matrix = []
        row = []
        for subplot in subplots:
            if isinstance(subplot, Subplot):
                row.append(
                    subplot_fit_image.image_at_coordinates(
                        *subplot.value,
                    )
                )
            else:
                row.append(subplot[i])
            if len(row) == subplot_width:
                matrix.append(row)
                row = []

        if len(row) > 0:
            matrix.append(row)

        return matrix

    @staticmethod
    def _matrix_to_image(matrix: List[List[Image.Image]]) -> Image.Image:
        """
        Create an image including all the images in the matrix.

        Parameters
        ----------
        matrix
            The matrix of images to combine into one image

        Returns
        -------
        The combined image.
        """
        total_width = sum(image.width for image in matrix[0])
        total_height = sum(image.height for image in list(zip(*matrix))[0])

        image = Image.new("RGB", (total_width, total_height))

        y_offset = 0
        for row in matrix:
            x_offset = 0
            for subplot_image in row:
                image.paste(subplot_image, (x_offset, y_offset))
                x_offset += subplot_image.width
            y_offset += row[0].height

        return image
