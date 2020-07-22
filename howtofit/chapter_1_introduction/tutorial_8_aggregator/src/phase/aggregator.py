from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.dataset import (
    dataset as ds,
)
from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.fit import fit as f

# This module contains convenience methods for computing standard objects from an *Aggregator* as generators.

# You should read this class in detail once you come to th end of part of tutorial 8, as the functions are written in a
# specific way such that they can be used as generators will become clear once the tutorial is completed.


def masked_dataset_generator_from_aggregator(aggregator):
    """Compute a generator of *MaskedDataset* objects from an input aggregator, which generates a list of the 
    *MaskedDataset* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *masked_dataset_from_agg_obj* with the aggregator, which sets up each masked
    dataset using only generators ensuring that manipulating the masked dataset objects of large sets of results is
    done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=masked_dataset_from_agg_obj)


def masked_dataset_from_agg_obj(agg_obj):
    """Compute a *MaskedDataset* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe 
     that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator 
     outputs such that the function can use the *Aggregator*'s map function to to create a *MaskedDataset* generator.

     The *MaskedDataset* is created following the same method as the *Phase* classes, including using the
     *meta_dataset* instance output by the phase to load inputs of the *MaskedDataset* (e.g. data_trim_left). 

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of model-fits.
    """

    dataset = agg_obj.dataset
    mask = agg_obj.mask

    masked_dataset = ds.MaskedDataset(dataset=dataset, mask=mask)

    meta_dataset = agg_obj.meta_dataset

    masked_dataset = masked_dataset.with_left_trimmed(
        data_trim_left=meta_dataset.data_trim_left
    )
    masked_dataset = masked_dataset.with_right_trimmed(
        data_trim_right=meta_dataset.data_trim_right
    )

    return masked_dataset


def model_data_generator_from_aggregator(aggregator):
    """Compute a generator of the model data arrays of the 1D profile models from an input aggregator, which
    generates a list of 1D ndarrays every set of results loaded in the aggregator.

    This is performed by mapping the *model_data_from_agg_obj* with the aggregator, which sets up each model data array
    using  only generators ensuring that manipulating the profiles of large sets of results is done in a memory
    efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=model_data_from_agg_obj)


def model_data_from_agg_obj(agg_obj):
    """Compute model data as a 1D ndarray from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to 
    describe that it acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's 
    generator outputs such that the function can use the *Aggregator*'s map function to to create a model-data generator.

     The model-data is created following the same method as the *Analysis* classes using an instance of the maximum 
     log likelihood model's profiles.

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of model-fits.
    """
    xvalues = agg_obj.dataset.xvalues
    instance = agg_obj.samples.max_log_likelihood_instance
    profiles = instance.profiles
    model_data = sum(
        [profile.profile_from_xvalues(xvalues=xvalues) for profile in profiles]
    )

    return model_data


def fit_generator_from_aggregator(aggregator):
    """Compute a generator of *FitDataset* objects from an input aggregator, which generates a list of the
    *FitDataset* objects for every set of results loaded in the aggregator.

    This is performed by mapping the *fit_from_agg_obj* with the aggregator, which sets up each fit using
    only generators ensuring that manipulating the fits of large sets of results is done in a memory efficient way.

    Parameters
    ----------
    aggregator : af.Aggregator
        A PyAutoFit aggregator object containing the results of PyAutoGalaxy model-fits."""
    return aggregator.map(func=fit_from_agg_obj)


def fit_from_agg_obj(agg_obj):
    """Compute a *Fit* object from an aggregator's *PhaseOutput* class, which we call an 'agg_obj' to describe  that it
    acts as the aggregator object for one result in the *Aggregator*. This uses the aggregator's generator outputs such
    that the function can use the *Aggregator*'s map function to to create a *Fit* generator.

     The *Fit* is created following the same method as the PyAutoGalaxy *Phase* classes. 

    Parameters
    ----------
    agg_obj : af.PhaseOutput
        A PyAutoFit aggregator's PhaseOutput object containing the generators of the results of PyAutoGalaxy model-fits.
    """
    masked_dataset = masked_dataset_from_agg_obj(agg_obj=agg_obj)
    model_data = model_data_from_agg_obj(agg_obj=agg_obj)

    return f.FitDataset(masked_dataset=masked_dataset, model_data=model_data)
