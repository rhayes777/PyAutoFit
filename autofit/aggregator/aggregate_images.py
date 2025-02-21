import sys
from enum import Enum
from typing import Optional

from PIL import Image

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
        *subplots: Subplot,
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

        Returns
        -------
        The combined image.
        """
        matrix = []
        for result in self._aggregator:
            subplot_fit_image = SubplotFitImage(result.image("subplot_fit"))
            row = []
            for subplot in subplots:
                row.append(
                    subplot_fit_image.image_at_coordinates(
                        *subplot.value,
                    )
                )
                if len(row) == subplot_width:
                    matrix.append(row)
                    row = []

            if len(row) > 0:
                matrix.append(row)

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
