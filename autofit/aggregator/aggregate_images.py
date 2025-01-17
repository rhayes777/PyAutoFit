from enum import Enum
from pathlib import Path
from typing import List

import PIL
from PIL import Image

from autofit.aggregator.aggregator import Aggregator


class Subplot(Enum):
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
        self._image = image

        self._single_image_width = self._image.width // 4
        self._single_image_height = self._image.height // 3

    def image_at_coordinates(self, x, y):
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
        self._aggregator = aggregator

    def extract_image(
        self,
        subplots: List[Subplot],
    ):
        matrix = []
        for result in self._aggregator:
            subplot_fit_image = SubplotFitImage(result.image("subplot_fit"))
            matrix.append(
                [
                    subplot_fit_image.image_at_coordinates(*subplot.value)
                    for subplot in subplots
                ]
            )

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
