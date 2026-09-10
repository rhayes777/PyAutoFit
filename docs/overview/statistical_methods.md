(statistical-methods)=

# Statistical Methods

Once a single fit works and its results are organized, the next questions are
usually statistical ones: how do I combine many datasets, which model does the
data actually support, how do I fit something too complex for one search, and
what would I need to measure to tell two models apart?

This page continues the [Natural Language Inference](natural_language.md) and
[Scientific Workflow](scientific_workflow.md) example. Every method below can
be requested through
[autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant) in natural
language. We keep using simple 1D Gaussian data, but you supply the scientific
meaning of the models, datasets and derived quantities in your own project.
The
[runnable workspace overview](https://github.com/PyAutoLabs/autofit_workspace/blob/main/scripts/overview/overview_3_statistical_methods.py)
contains the corresponding Python, and each method has a longer worked example
in the
[features folder](https://github.com/PyAutoLabs/autofit_workspace/tree/main/notebooks/features).

Each section below shows **two prompts**, and they are meant to be used in that
order. Ask the first to have the assistant explain the method — what it does,
when it applies to your data and what it will and will not tell you — so you
can decide whether it suits your problem. Ask the second to have the assistant
actually set the method up and run it.

## Contents

- **Graphical Models**: Fit many datasets simultaneously with parameters that are local to each dataset and global across all of them.
- **Hierarchical Models**: Assume a shared parameter is drawn from a parent distribution and infer that distribution.
- **Expectation Propagation**: Scale graphical and hierarchical models to large datasets by fitting one dataset at a time and passing messages.
- **Model Comparison**: Fit competing models to the same data and use the Bayesian evidence to decide which is supported.
- **Interpolation**: Fit similar datasets one-by-one and interpolate their parameters to any point in between.
- **Search Grid Search**: Grid a subset of parameters and run a full non-linear search in every cell.
- **Search Chaining**: Break a difficult fit into a sequence of simpler searches, each seeding the next.
- **Sensitivity Mapping**: Establish what data quality would be needed for a more complex model to be favoured.

## Graphical Models

:::{container} nl-prompt
> Walk me through graphical models in PyAutoFit before we run anything. What is
> a factor graph, what counts as a local parameter versus a global one, and how
> do I recognise that my datasets share a parameter? Cover the inputs it needs
> and how to read the result. Go into more detail than this docs page, using the
> three low signal-to-noise 1D Gaussian datasets as the running example.
:::

When you have many datasets, you are often not interested in each fit on its
own. You want the trend across the whole sample: a parameter that is the same
everywhere, a relationship that only becomes visible once every dataset
contributes. Fitting each dataset separately and averaging the answers
afterwards throws away information and misstates the uncertainty.

A graphical model describes the dependencies instead of hiding them. Each
dataset keeps its own local parameters, while parameters believed to be shared
appear once in the model and are constrained by every dataset at the same time.
The workspace example makes this concrete with three noisy 1D Gaussians
simulated with the same centre: the centre is composed as a single shared
prior used by all three components, so the dimensionality of the fit is lower
than three independent fits and the shared centre is constrained by all of the
data at once.

When you ask for this, the assistant pairs each dataset's analysis with its own
model, declares which parameters are shared, and builds the factor graph that
connects them. Deliberately low signal-to-noise data are used in the example
because that is where the difference shows: individually the datasets say
little about the centre, together they say a great deal.

:::{container} nl-prompt
> Fit the three low signal-to-noise 1D Gaussian datasets with a graphical model
> in which the centre is shared and the normalization and width stay local to
> each dataset. Report the shared centre and its uncertainty, and compare it
> with what I get by fitting the three datasets individually.
:::

:::{container} ai-first-design
**AI First Design:** A node in the factor graph is just a named model paired with an analysis — the same two objects a single-dataset fit already uses. The assistant can therefore extend a working one-dataset fit into a graphical one by declaring which parameters are shared, without rewriting your likelihood.
:::

Full example: [graphical_models.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/graphical_models.ipynb).

## Hierarchical Models

:::{container} nl-prompt
> Give me a run-through of hierarchical models in PyAutoFit: what a parent
> distribution is, when I should assume my parameters are drawn from one, what
> gets inferred and how the result differs from fitting each dataset alone. Go
> into more detail than the docs page and use the 1D Gaussian datasets as the
> running example.
:::

A hierarchical model is a particular kind of graphical model. Instead of
asserting that a parameter is exactly the same in every dataset, you assert
that each dataset's value is drawn from a common parent distribution — and you
infer the parameters of that distribution, typically its mean and its scatter,
from the data.

This is the right assumption whenever the individual values are expected to
vary but not arbitrarily: measurements of the same physical population, repeats
of an experiment under slightly different conditions, sources drawn from a
common underlying sample. The parent distribution is a scientific result in its
own right, and it also feeds back on the individual fits, since a dataset that
constrains its own parameter poorly is informed by where the rest of the sample
sits.

Asked for a hierarchical fit, the assistant composes the parent distribution,
draws each dataset's parameter from it, and fits everything together, so the
parent and the individual parameters are inferred consistently rather than in
two disconnected stages.

:::{container} nl-prompt
> Fit the sample of 1D Gaussian datasets with a hierarchical model in which
> each centre is drawn from a parent Gaussian. Report the inferred mean and
> scatter of the parent, show how each dataset's centre shifts relative to
> fitting it in isolation, and say which datasets are affected most.
:::

Full example: [tutorial_4_hierachical_models.ipynb](https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/tutorial_4_hierachical_models.ipynb).

## Expectation Propagation

:::{container} nl-prompt
> Explain expectation propagation to me before we use it. Why does a joint fit
> to a graphical model stop working as datasets are added, what does EP do
> instead, what are messages and damping, and how do I tell whether it has
> converged or gone wrong? Use the three shared-centre 1D Gaussian datasets and
> go beyond what this page says.
:::

Graphical and hierarchical models both have a ceiling. Fitting them with a
single non-linear search means sampling one joint parameter space, and that
space grows with every dataset you add. With tens of datasets it becomes
inefficient, and with hundreds or thousands it becomes impossible.

Expectation propagation is the way past that ceiling. Rather than one
high-dimensional fit, EP fits the factor graph one node at a time: each dataset
is fitted on its own, at low dimensionality, and the fits exchange messages
that carry what each has learned about the shared parameters. Cycling through
the datasets several times converges on an approximation to the full joint
posterior. Because the cost grows with the number of datasets rather than with
the dimension of a single fit, the same graphical or hierarchical model you
prototyped on three datasets can be scaled to a large sample.

EP is an approximation and it can fail to settle, so it is worth asking the
assistant for its convergence diagnostics as well as its parameter estimates.
The feature example dissects a single EP update at the low-level API — the
mean-field approximation, the cavity distribution, the tilted fit, moment
matching and the damped message update — before running the same loop
end-to-end through the high-level interface.

:::{container} nl-prompt
> Refit the shared-centre graphical model with expectation propagation instead
> of a single joint search. Report the shared centre and each dataset's local
> parameters, show how the estimate changed over the EP iterations, and tell me
> whether the run converged.
:::

Full example: [expectation_propagation.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/expectation_propagation.ipynb),
with a dedicated tutorial in
[HowToFit chapter 3](https://github.com/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_3_graphical_models/tutorial_5_expectation_propagation.ipynb).

## Model Comparison

:::{container} nl-prompt
> Before we compare anything, explain Bayesian model comparison in PyAutoFit:
> what the Bayesian evidence is, how it differs from the log likelihood, how
> priors enter it and what would make a comparison misleading. Use the
> two-Gaussian dataset as the example and give me more detail than the docs
> page.
:::

What model should I use? How many components should it have? Is it too complex
or too simple? These are the questions model comparison answers, by fitting
several models to the same data and asking which the data actually support.

The log likelihood alone cannot answer them. Adding parameters can only improve
the best achievable likelihood, so it will keep favouring the more complex
model whether or not the extra complexity is real. The Bayesian evidence
penalizes complexity, so a more elaborate model wins only if it fits
substantially better. The workspace example fits data that were simulated from
two Gaussians with models of one, two and three Gaussians. The three-Gaussian
model can only match or exceed the two-Gaussian likelihood, so it is the
evidence, not the likelihood, that separates the model the data came from
from an overfit.

The assistant composes each candidate model, fits them with the same data,
likelihood and search settings, and reports both metrics side by side. Keep the
comparison honest: evidence values are only comparable within the same dataset
and likelihood convention, and they depend on the priors, so ask what the
priors were doing before accepting a verdict.

:::{container} nl-prompt
> Fit the two-Gaussian dataset with models of one, two and three Gaussians,
> keeping the likelihood, priors and search settings the same. Give me a table
> of log likelihood and Bayesian evidence for each, say which model is
> favoured, and show the residuals so I can see what the rejected models miss.
:::

:::{container} ai-first-design
**AI First Design:** Because the model is a separate, named object from the likelihood and the search, a comparison changes only the model. The assistant reuses the identical analysis and search for every candidate, so the differences it reports are differences in the model rather than in how it was fitted.
:::

Full example: [model_comparison.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/model_comparison.ipynb).

## Interpolation

:::{container} nl-prompt
> Talk me through PyAutoFit's interpolation feature: what it takes as input,
> what it produces, when interpolating fitted parameters is legitimate and when
> it is not. Use the 1D Gaussian datasets observed at different times, and give
> more detail than this page.
:::

It is common to fit the same model to many similar datasets in which one or
more parameters are expected to vary smoothly — observations taken at different
times, or at different wavelengths, temperatures or positions. Having fitted
each dataset, you often want the model at a point where you have no data.

PyAutoFit's interpolation feature does exactly this. Each dataset is fitted
individually, the resulting instances are collected against their coordinate,
and a `LinearInterpolator` returns a model instance at any requested value of
that coordinate. The workspace example uses three 1D Gaussian datasets taken at
three times, whose centre drifts smoothly, and estimates the centre at times in
between. The interpolator can be serialized to JSON and rebuilt from results
loaded through the aggregator, so this scales to a study you have already run.

Interpolating between fits is a statement about the parameters, not a fit to
the intervening data, so it is only as good as the smoothness assumption. Ask
for the fitted values and their uncertainties alongside the interpolated curve
so you can judge that.

:::{container} nl-prompt
> Fit the three time-ordered 1D Gaussian datasets individually, build a linear
> interpolator over the results, and report the interpolated centre at a time
> between two of the observations. Plot the fitted centres with their
> uncertainties against the interpolated curve.
:::

Full example: [interpolate.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/interpolate.ipynb).

## Search Grid Search

:::{container} nl-prompt
> Explain the search grid search before we use it. How does it differ from a
> plain grid over the likelihood, what happens to the gridded parameters during
> each cell's fit, when does it help with multimodality, and how should I choose
> the grid resolution? Use the Gaussian dataset with the small feature and go
> deeper than this page.
:::

A classic grid search divides parameters onto a grid and samples the likelihood
at each point. For low-dimensional problems this is enough, but it scales
badly: the number of grid points explodes with the number of parameters.

PyAutoFit's search grid search is a hybrid. A subset of parameters is gridded,
and in every cell a full non-linear search fits all of the remaining
parameters. The gridded parameters are not frozen — they remain part of the fit,
with their values simply confined to the bounds of their cell. This is what
makes the method useful for awkward parameter spaces: if you know which
parameters drive the multimodality, gridding them means each individual fit
sees a unimodal problem, and every cell also returns its own goodness-of-fit,
giving an evidence map over the grid rather than a single number. The grid
cells are independent, so the whole thing is embarrassingly parallel and
PyAutoFit can run the cells concurrently.

The workspace example uses 1D data containing a main Gaussian plus a small
feature near pixel 70, and grids the feature's position so that each cell asks
whether the feature belongs there.

:::{container} nl-prompt
> Run a search grid search on the Gaussian dataset with the feature, gridding
> the feature's centre and fitting everything else with a non-linear search in
> each cell. Give me the evidence map over the grid, point out which cells
> favour the feature, and run the cells in parallel.
:::

Full example: [search_grid_search.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/search_grid_search.ipynb).

## Search Chaining

:::{container} nl-prompt
> Give me a proper introduction to search chaining: what can be handed from one
> search to the next, the difference between passing a fitted instance and
> passing updated priors, how to choose the width of a passed prior, and how a
> chain can go wrong by locking in an early mistake. Use the two split
> Gaussians as the example and go beyond this page.
:::

Fitting a complex model with a single search is often slower and less reliable
than fitting a sequence of simpler ones. Search chaining breaks the problem
into bite-sized stages: early searches fit simplified models with cheap
settings, and their results initialize later searches that fit the full model
properly.

What makes this work is your domain knowledge. The workspace example uses data
containing two clearly separated Gaussians and fits them in three chained
searches: the left Gaussian alone, then the right Gaussian alone, then both
together with six free parameters, seeded by the first two. Each stage is a
three-parameter problem instead of a six-parameter one, and the final search
starts where the answer already is.

There are two currencies for the hand-off. Passing a result's fitted
**instance** fixes those parameters at their measured values, so they cost
nothing in the next stage. Passing the result's **model** hands the component on
as still free, but with its priors updated from the previous posterior, and
variants let you set the width of the passed priors explicitly rather than
inheriting the posterior's. The rule of thumb is that a chain should narrow the
search, never bias it: a prior tight enough to lock in an early systematic has
replaced inference with anchoring. Ask the assistant to state what was passed
and how tightly at every link.

:::{container} nl-prompt
> Fit the two split Gaussians as a three-search chain: left Gaussian, right
> Gaussian, then both together. For each link tell me exactly what was passed
> as a fixed instance and what was passed as updated priors, including the
> prior widths, and compare the final result with fitting all six parameters in
> one search.
:::

:::{container} ai-first-design
**AI First Design:** Each search returns a structured `Result` that exposes both an `instance` and a `model` built from the same named components. Prior passing is therefore a one-line hand-off the assistant can compose from your description of the stages, and it can say in plain language what each link fixed and what it left free.
:::

Full example: [search_chaining.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/search_chaining.ipynb).

## Sensitivity Mapping

:::{container} nl-prompt
> Before we map anything, explain sensitivity mapping: how it differs from
> model comparison, what the base and perturbed models are, what the grid is
> over, what is simulated and fitted at each grid point, and what the resulting
> map does and does not prove. Use the Gaussian dataset with the small feature
> as the example, in more detail than this page.
:::

Model comparison can tell you that a complex model is not favoured, but it
cannot tell you why. The complex model may be wrong — or it may be right, and
your data simply are not good enough to reveal it. Sensitivity mapping
separates these cases by asking what quality of data would be needed for the
more complex model to be favoured.

It works by simulation. You define a base model and a perturbed model that adds
the feature in question, and a grid over the perturbation's parameters. At each
grid point a dataset is simulated containing that perturbation, and it is fitted
twice, once with the base model and once with the perturbed model. Comparing
the evidence of the two fits gives, at every grid point, the strength of
evidence you would expect to obtain. In the workspace example, the base model is
the main Gaussian, the perturbation is the small feature at pixel 70, and the
grid runs over the feature's normalization: the map shows how bright the
feature must be before the data can detect it.

The result is a statement about your experiment, not only about the dataset in
hand. It tells you what your current data could have detected, and by extension
what a deeper or cleaner observation would buy you.

:::{container} nl-prompt
> Run sensitivity mapping on the Gaussian-with-feature data. Use the single
> Gaussian as the base model and the feature as the perturbation, grid over the
> feature's normalization, and simulate and fit a dataset at each grid point
> with both models. Give me the map of evidence difference and tell me the
> normalization above which the feature would be detected.
:::

:::{container} ai-first-design
**AI First Design:** Sensitivity mapping runs on the simulate function and analysis you already wrote for your own science. The assistant supplies the grid, the base and perturbed models and the bookkeeping, so the simulated datasets are generated by your code and not by a generic stand-in.
:::

Full example: [sensitivity_mapping.ipynb](https://github.com/PyAutoLabs/autofit_workspace/blob/main/notebooks/features/sensitivity_mapping.ipynb).

## What Next?

These methods compose. A hierarchical model can be scaled with expectation
propagation, a chained search can seed a graphical fit, and model comparison
and sensitivity mapping answer two halves of the same question. Pick the one
that matches the question your data are actually posing, and ask the assistant
to explain it before you run it.

To see the Python behind everything described here, read
[The Python API](https://pyautofit.readthedocs.io/en/latest/overview/python_api.html),
then work through the runnable examples in the
[features folder](https://github.com/PyAutoLabs/autofit_workspace/tree/main/notebooks/features)
of the [autofit_workspace](https://github.com/PyAutoLabs/autofit_workspace).
Those examples are the material the
[autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant) is built
on, so reading them is also the fastest way to understand what it is doing on
your behalf. Chapter 3 of [HowToFit](https://github.com/PyAutoLabs/HowToFit)
teaches graphical models, hierarchical models and expectation propagation from
first principles.
