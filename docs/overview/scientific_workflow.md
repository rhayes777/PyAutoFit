(scientific-workflow)=

# Scientific Workflow

After fitting a model, you need to inspect the result, understand how it was
obtained and decide what to try next. A scientific workflow connects these
steps across your study: perhaps many models for one dataset, or the same
models fitted to thousands of datasets.

This guide continues the [Natural Language Inference](natural_language.md)
example. Every task can be requested through
[autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant) in natural
language. We use a simple 1D Gaussian profile, but you supply the scientific
meaning of the models, data and derived quantities for your own project.
The [workspace example](https://github.com/PyAutoLabs/autofit_workspace/blob/main/scripts/overview/overview_2_scientific_workflow.py)
contains the corresponding runnable Python.

## Contents

- **Hard Disk Output**: Save every fit's model, samples, search settings and a domain-specific summary to disk in a readable layout.
- **Visualization**: Save plots of the data before inference and of the best fit and residuals during it.
- **On The Fly**: Watch the fit and residual figure update live while the search runs.
- **Loading Results**: Reload saved fits with the aggregator and tabulate or reinspect them without rerunning.
- **Result Customization**: Expose the fitted profile, residuals and derived quantities directly from the result.
- **Model Composition**: Make competing assumptions explicit by fitting free, fixed and shared-parameter variants of the model.
- **Searches**: Compare Nautilus, Dynesty, Emcee and an optimizer on the same model, data and priors.
- **Configs**: Put shared priors and search settings into configuration files and record the effective settings with each result.
- **Database**: Collect saved runs into a SQLite database and query them by model, metadata or result properties.
- **Scaling Up**: Organize a study of many datasets, models and searches so every conclusion traces back to its saved output.

## Hard Disk Output

> Describe the contents of the output folder from our completed inference,
> explaining what each file and subfolder tells me. Add a domain-specific
> results summary in `science_summary.json`.

Saving results to hard disk makes it possible to:

- Inspect many fits without keeping their original notebook sessions open.
- Check intermediate results and images while inference is running.
- Revisit the model, priors, search settings and samples behind a conclusion.
- Resume an interrupted search when its saved search state supports resuming.
- Collect results from runs performed on another machine or a computing cluster.

If your previous fit did not save output, first ask:

> Configure this analysis to save its results under `output/scientific_workflow`,
> with readable labels for the dataset, model and search. Run the fit and show
> me its output location.

PyAutoFit enables directory output when a search is given a name or path
prefix. Within the chosen location, an identifier derived from the model and
search configuration distinguishes runs. It is not a random folder name or a
substitute for labelling datasets: retain a dataset tag and record which data
were fitted. Reusing an existing run's identity can reload or resume it.

At the top level of a run, `model.info` describes the model and its priors,
`model.results` summarizes parameter estimates, and `search.summary` records
search information such as runtime. The `image/` folder holds visualizations,
and `search_internal/` can hold the search's own state. The `files/` folder
contains structured information that both you and the assistant can read:

| File | What it records |
| --- | --- |
| `model.json` | Named model components, parameters, priors and their relationships. |
| `search.json` | The search type and its saved configuration. |
| `samples_summary.json` | A compact summary of the inferred solution and available uncertainty estimates. |
| `samples.csv` | Sampled parameter values, log likelihoods, log priors and weights. |
| `samples_info.json` | Metadata needed to interpret and reload the samples. |
| `info.json` | Additional metadata supplied for the fit, such as dataset and model labels; present when supplied. |
| `covariance.csv` | A parameter covariance matrix, when available and enabled. |
| `science_summary.json` | The custom scientific summary requested above; added by your analysis. |

The exact files depend on the search, output settings and how far the run has
progressed. They are not all JSON, and an incomplete run may not yet have a
final summary.

For example, this excerpt from the workspace example's `model.json` records
the prior on the Gaussian's centre; the other parameters are omitted here:

```json
{
    "type": "model",
    "class_path": "autofit.example.model.Gaussian",
    "arguments": {
        "centre": {
            "type": "Uniform",
            "id": 3,
            "lower_limit": 0.0,
            "upper_limit": 100.0
        }
    }
}
```

You can see what the parameter is called, which model it belongs to and what
values were allowed. This matters when two fits give different answers:
their saved models let you check whether the assumptions also changed.

For the Gaussian example, the custom summary can record the full width at
half maximum and residual statistics at the maximum-likelihood solution,
together with units and the dataset label. These are derived diagnostics;
posterior uncertainties require evaluating the derived quantity over the
samples. Ask for quantities that answer questions in your own science:

> Include the full width at half maximum and the residual root mean square
> in the saved summary. State their units and which fitted solution they
> describe, so I can compare them across datasets later.

Structured output is the foundation of a scientific workflow. Each fit
retains a readable record of its assumptions and results, so adding more
datasets does not mean losing track of what was fitted or how to interpret it.

## Visualization

> Before inference, save a plot of the data with its uncertainties, so I can
> check the dataset that will be fitted.

> During inference, update separate plots of the best-fitting profile over
> the data and its residuals. Keep the data plot alongside them and save the
> final versions when inference finishes.

You can specify visualization before fitting separately from visualization
during fitting. The first shows quantities that do not change, such as the
measurements and noise map. The second uses the best solution found so far,
so you can follow the model as the search explores parameter space.

In the workspace example these images are `data.png`, `model_fit.png` and
`residuals.png`. Separate filenames preserve both the fit and its residuals.
For convenient inspection, ask for a combined view too:

> Also save a two-panel figure with the model fit above its residuals, using
> the same horizontal scale. Use this figure for live updates as well.

This produces `fit.png`, the shared figure used by the live display in the
next section. In your own workflow, choose plots that reveal scientifically
meaningful shortcomings: a good-looking best fit can still leave systematic
structure in the residuals. Search-specific posterior plots provide another
view of how well parameters are constrained.

## On The Fly

> While inference runs, refresh the combined model-fit and residual figure
> in my notebook and report the best parameters found so far. Start with an
> update every 500 likelihood evaluations, and explain what the updates
> suggest about the search's progress.

Live output connects the visualization you have just chosen to the running
inference. In a notebook, the fit figure updates in place. When running a
Python script on a desktop, live visual updates can instead use a separate
viewer. The saved images remain available for inspection on disk.

Quick updates show the best solution encountered so far without waiting for
a full update of the search's samples and posterior products. Their cadence
is measured in likelihood evaluations; full search updates have their own
cadence. The workspace example explicitly enables quick updates and live
visualization, and uses the same plotting routine for the live figure and
the saved fit image.

> Keep the live updates frequent while I develop the analysis. If plotting
> becomes expensive, reduce their frequency while retaining periodic saved
> output. For a cluster run, save the images without opening a viewer.

This feedback helps build intuition about inference: is the model finding
the signal, do the residuals retain structure, and is the search making
progress? A slowly changing image may motivate checking priors, the
likelihood or the search settings. A stable best-fit image alone does not
establish convergence or show that the posterior has been fully explored.

## Loading Results

> Reload the saved fits under `output/scientific_workflow` without rerunning
> inference. Make a table of the Gaussian widths and their 68% credible
> intervals, labelled by dataset, model and search, and include each run's
> output path.

PyAutoFit's aggregator loads results from a collection of output folders.
The assistant can inspect saved models and samples, calculate summaries and
return to particular runs. Results are loaded lazily, so processing a large
collection does not require holding every sample from every fit in memory
at once.

> Open the fit with the widest uncertainty on sigma. Show its posterior and
> saved residual plot, and explain what might account for the uncertainty.

This turns a table entry into an inspectable scientific result. Retain the
original data and analysis code as well as the saved outputs if you want to
compute new model predictions or plots later. See the
[result cookbook](../cookbooks/result.md) for more ways to inspect a result.

## Result Customization

> Make the best-fitting profile, residuals and full width at half maximum
> directly accessible from the result. Use those same quantities in the
> scientific summary saved for each fit.

A useful result should expose the quantities you need to interpret your
science. For this example, that means the fitted 1D profile and its width;
another project might require an integrated signal or a derived physical
quantity. The workspace example shows how to extend a result while keeping
access to the model, samples and analysis.

> Calculate the posterior median and 68% credible interval of the full width
> at half maximum from the saved samples. Distinguish this uncertainty from
> the width evaluated at the maximum-likelihood solution.

Derived quantities, also called latent variables, need not be sampled
parameters themselves. Computing them from posterior samples lets you
propagate parameter uncertainty into quantities that matter for your study.
The [result cookbook](../cookbooks/result.md) develops these customizations.

## Model Composition

> Fit this dataset with a Gaussian whose width is free, then with its width
> fixed to an independently measured value of 10 in the same coordinate
> units. Show the priors and save each model under a readable label.

Model composition is introduced in [Natural Language Inference](natural_language.md)
and developed in the [model cookbook](../cookbooks/model.md). Here, its role
is to make competing scientific assumptions explicit and comparable: one
component or two, a fixed parameter or a free one, shared parameters or
independent ones.

> Compare the saved free-width and fixed-width fits to this same dataset.
> Show residuals, parameter constraints and Bayesian evidence where
> available. Explain how the priors affect the comparison, and link to the
> saved model definitions.

The workflow must make many models feasible to interpret as well as feasible
to fit. Named components, consistent diagnostics and saved assumptions let
you work out what changed and whether the additional complexity is useful.
Keep model comparisons within the same dataset and likelihood convention;
evidence values from different datasets are not a ranking of model quality.

## Searches

> Fit the same model with Nautilus and Dynesty, keeping the data, likelihood
> and priors fixed. Compare runtime, likelihood evaluations, parameter
> constraints and the searches' available convergence diagnostics.

Model dimension, parameter correlations and likelihood cost all affect
which search works well. Trying more than one method early helps establish
whether results are reliable and how expensive the wider study will be.
Give each search its own saved location so both outcomes remain available.

> Compare these posterior constraints with an Emcee run, and the best-fit
> values with an optimizer. Report Bayesian evidence only for searches
> that estimate it, and flag runs whose sampling is insufficient.

Agreement in best-fit parameters does not imply agreement in posterior
uncertainties. Similarly, a fast unfinished run is not evidence that an
algorithm is more efficient. If your likelihood supports differentiation,
gradient-based searches offer further possibilities. See the
[search cookbook](../cookbooks/search.md) for the available interfaces.

## Configs

> Put the shared priors and search settings for this study into configuration
> files. Show the defaults and any per-fit overrides, and preserve the
> effective model and search settings with each saved result.

As the study grows, consistent defaults reduce repeated setup and make
intentional differences easier to see. Configuration files can also control
output and update frequency. A later configuration change should not erase
your record of the settings used for a previous inference. The
[configs cookbook](../cookbooks/configs.md) explains how to organize them.

## Database

> Collect the saved runs into a SQLite database. Select the completed
> free-width Gaussian fits and make a table of their widths and uncertainties,
> retaining dataset, model and search labels.

Folders remain useful for direct inspection. A database adds a way to query
a large collection by model, metadata or result properties, then load only
the results needed for a comparison. You can build it from existing output
folders, so the first exploratory fits can become part of the larger study.

> Find runs with missing summaries or unusually large residuals. Show the
> relevant output paths and explain which need further inspection.

The [multiple datasets cookbook](../cookbooks/multiple_datasets.md) describes
collecting and querying results at this scale.

## Scaling Up

> Set up a study of five datasets, fitting each with free-width and fixed-width
> Gaussian models using Nautilus and Dynesty. Organize the output by dataset, model
> and search, and retain the same scientific summaries and diagnostic plots
> for every run.

The following is an **illustrative directory layout**, not a claim that this
page has performed twenty fits. Each search folder contains its run's
identifier directory; one run is expanded to show what you can inspect.

```text
output/scientific_workflow/
├── dataset_01/
│   ├── gaussian_free_width/
│   │   ├── nautilus/<identifier>/
│   │   │   ├── model.info
│   │   │   ├── model.results
│   │   │   ├── search.summary
│   │   │   ├── files/
│   │   │   │   ├── model.json
│   │   │   │   ├── search.json
│   │   │   │   ├── samples.csv
│   │   │   │   ├── samples_info.json
│   │   │   │   ├── samples_summary.json
│   │   │   │   ├── info.json
│   │   │   │   └── science_summary.json
│   │   │   └── image/
│   │   │       ├── data.png
│   │   │       ├── model_fit.png
│   │   │       ├── residuals.png
│   │   │       └── fit.png
│   │   └── dynesty/<identifier>/
│   └── gaussian_fixed_width/
│       ├── nautilus/<identifier>/
│       └── dynesty/<identifier>/
├── dataset_02/
│   ├── gaussian_free_width/  (nautilus and dynesty runs)
│   └── gaussian_fixed_width/    (nautilus and dynesty runs)
├── dataset_03/
│   ├── gaussian_free_width/  (nautilus and dynesty runs)
│   └── gaussian_fixed_width/    (nautilus and dynesty runs)
├── dataset_04/
│   ├── gaussian_free_width/  (nautilus and dynesty runs)
│   └── gaussian_fixed_width/    (nautilus and dynesty runs)
└── dataset_05/
    ├── gaussian_free_width/  (nautilus and dynesty runs)
    └── gaussian_fixed_width/    (nautilus and dynesty runs)
```

You can browse a particular run on disk, open its plots and read its model.
Or you can ask about the collection:

> Inspect all five datasets and compare the models and searches fitted to
> each. Summarize parameter constraints, fit quality and runtime, compare
> Bayesian evidence where available, and link each assessment to its saved
> output. Flag incomplete runs and results that need closer inspection.

This is the purpose of a scientific workflow: as the number of experiments
grows, you can still trace a conclusion to its data, assumptions and results.
The same organization supports early experimentation with a handful of fits
and later studies where manual inspection of every run is impractical.

Continue with [Statistical Methods](statistical_methods.md) for hierarchical
models, search chaining and other ways to extend your inference. To see how
the workflow above is implemented, open the
[runnable workspace overview](https://github.com/PyAutoLabs/autofit_workspace/blob/main/scripts/overview/overview_2_scientific_workflow.py).
