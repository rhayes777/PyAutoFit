import autofit as af

from src.dataset import dataset as ds
from src.fit import fit as f

import numpy as np
from functools import partial

"""
This module contains convenience methods for computing standard objects from an `Aggregator` as generators. This is
described in tutorial 4 of chapter 2.
"""


def dataset_generator_from_aggregator(
    aggregator: af.Aggregator, settings_dataset: ds.SettingsDataset = None
) -> af.PhaseOutput:
    """
    Returns a generator of `Dataset` objects from an input aggregator : af.Aggregator, which generates a list of the
    `Dataset` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `dataset_from_agg_obj` with the aggregator : af.Aggregator, which sets up each masked
    `Dataset` using only generators ensuring that manipulating the `Dataset` objects of large sets of results is
    done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits.
    settings : SettingsDataset
        The dataset settings that define whether the `Dataset` is trimmed and by how much.
    """

    func = partial(dataset_from_agg_obj, settings_dataset=settings_dataset)

    return aggregator.map(func=func)


def dataset_from_agg_obj(
    agg_obj: af.PhaseOutput, settings_dataset: ds.SettingsDataset = None
) -> ds.Dataset:
    """
    Returns a `Dataset` object from an aggregator`s `PhaseOutput` class, which we call an `agg_obj` to describe
     that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator`s generator
     outputs such that the function can use the `Aggregator``s map function to to create a `Dataset` generator.

     The `Dataset` is created following the same method as the `Phase` classes, including using the
     `meta_dataset` instance output by the phase to load inputs of the `Dataset` (e.g. data_trim_left).

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator`s PhaseOutput object containing the generators of the results of model-fits.
    settings : SettingsDataset
        The dataset settings that define whether the `Dataset` is trimmed and by how much.
    """

    dataset = agg_obj.dataset
    settings = agg_obj.settings

    if settings_dataset is None:
        settings_dataset = settings.settings_dataset

    return dataset.trimmed_dataset_from_settings(settings=settings_dataset)


def model_data_generator_from_aggregator(
    aggregator: af.Aggregator, settings_dataset: ds.SettingsDataset = None
) -> af.PhaseOutput:
    """
    Returns a generator of the model data arrays of the 1D profile models from an input aggregator : af.Aggregator, which
    generates a list of 1D ndarrays every set of results loaded in the aggregator.

    This is performed by mapping the `model_data_from_agg_obj` with the aggregator : af.Aggregator, which sets up each model data array
    using  only generators ensuring that manipulating the profiles of large sets of results is done in a memory
    efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits.
    settings : SettingsDataset
        The dataset settings that define whether the `Dataset` is trimmed and by how much.
    """

    func = partial(model_data_from_agg_obj, settings_dataset=settings_dataset)

    return aggregator.map(func=func)


def model_data_from_agg_obj(
    agg_obj: af.PhaseOutput, settings_dataset: ds.SettingsDataset = None
) -> np.ndarray:
    """
    Returns model data as a 1D ndarray from an aggregator`s `PhaseOutput` class, which we call an `agg_obj` to
    describe that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator`s
    generator outputs such that the function can use the `Aggregator``s map function to to create a model-data generator.

     The model-data is created following the same method as the `Analysis` classes using an instance of the maximum
     log likelihood model's profiles.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator`s PhaseOutput object containing the generators of the results of model-fits.
    settings : SettingsDataset
        The dataset settings that define whether the `Dataset` is trimmed and by how much.
    """

    dataset = dataset_from_agg_obj(agg_obj=agg_obj, settings_dataset=settings_dataset)

    xvalues = dataset.xvalues
    instance = agg_obj.samples.max_log_likelihood_instance
    profiles = instance.profiles

    return sum([profile.profile_from_xvalues(xvalues=xvalues) for profile in profiles])


def fit_generator_from_aggregator(
    aggregator: af.Aggregator, settings_dataset: ds.SettingsDataset = None
) -> af.PhaseOutput:
    """
    Returns a generator of `FitDataset` objects from an input aggregator : af.Aggregator, which generates a list of the
    `FitDataset` objects for every set of results loaded in the aggregator.

    This is performed by mapping the `fit_from_agg_obj` with the aggregator : af.Aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits.
    settings : SettingsDataset
        The dataset settings that define whether the `Dataset` is trimmed and by how much.
    """

    func = partial(fit_from_agg_obj, settings_dataset=settings_dataset)

    return aggregator.map(func=func)


def fit_from_agg_obj(
    agg_obj: af.PhaseOutput, settings_dataset: ds.SettingsDataset = None
) -> f.FitDataset:
    """
    Returns a `Fit` object from an aggregator`s `PhaseOutput` class, which we call an `agg_obj` to describe  that it
    acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator`s generator outputs such
    that the function can use the `Aggregator``s map function to to create a `Fit` generator.

     The `Fit` is created following the same method as the PyAutoGalaxy `Phase` classes.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator`s PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    settings : SettingsDataset
        The dataset settings that define whether the `Dataset` is trimmed and by how much.
    """
    dataset = dataset_from_agg_obj(agg_obj=agg_obj, settings_dataset=settings_dataset)
    model_data = model_data_from_agg_obj(
        agg_obj=agg_obj, settings_dataset=settings_dataset
    )

    return f.FitDataset(dataset=dataset, model_data=model_data)
