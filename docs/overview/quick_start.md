# Quick Start

> **Access requirements**
>
> The assistant runs inside an AI coding agent: Claude Code or OpenAI Codex are
> recommended. Sustained scientific use normally needs paid access to one of them
> (a personal subscription, institutional access or API billing); OpenCode is an
> experimental alternative whose model access and capability depend on the
> provider. See the [assistant setup guide](https://github.com/PyAutoLabs/autofit_assistant#getting-started)
> for current options. PyAutoFit itself is open source.

**PyAutoFit is a Python package for scientific model fitting and Bayesian
inference.** Bring your own model classes, data and likelihood code;
PyAutoFit provides priors, inference algorithms and tools for interpreting
and organising the results. It is domain agnostic: inference can sit around
your existing scientific software.

We recommend getting started with [autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant), which lets you
**perform scientific inference using natural language**. Describe the model
you want to fit, ask the assistant to run the analysis, and explore the
results through follow-up requests. PyAutoFit's named models, separate
likelihood, interchangeable searches and structured results make this
workflow well suited to AI agents.

To begin instantly, follow the [assistant setup guide](https://github.com/PyAutoLabs/autofit_assistant#getting-started) and ask:

:::{container} nl-prompt
> I want to perform scientific inference with PyAutoFit (https://github.com/PyAutoLabs/PyAutoFit) and the
> autofit_assistant (https://github.com/PyAutoLabs/autofit_assistant).
>
> Begin the "start here" guide for a new user.
:::

The assistant answers this prompt with its guided
[**start here** mode](https://github.com/PyAutoLabs/autofit_assistant/blob/main/modes/start_here.md):
six steps on the bundled 1D Gaussian which you type yourself, followed by the same
six steps on your own science. At any step you can ask a question (for example
"what is a prior?") or say "teacher mode" to get a full explanation of everything
as you go. The six steps are the sections of [Natural Language Inference](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html).

## Bring Your Own Likelihood

Already have a likelihood function for your science problem? **Point the
assistant at your existing code and it can set it up with PyAutoFit** —
defining the model, choosing priors with you, configuring a search and
organising the results:

:::{container} nl-prompt
> Set up PyAutoFit with my existing science project. An example likelihood
> function can be found at [GitHub link or local directory].
>
> First, give me an overview of my project and likelihood function. Compose
> an appropriate model, explain it to me, and recommend a non-linear search
> (for example MCMC, nested sampling or maximum-likelihood estimation).
>
> Do not begin inference until we have discussed the setup and I give you
> the go-ahead.
>
> Once inference is running, explain how the results are written to disk and
> show me how to inspect and interpret them with PyAutoFit.
:::

Your existing science code remains the source of the likelihood. With
PyAutoFit built around it, you can perform inference through natural
language while gaining access to features such as flexible priors and model
composition, MCMC and nested sampling, automated result handling, model
comparison and scalable workflows.

## Next: Natural Language Inference

[Natural Language Inference](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html) describes
using **PyAutoFit** through natural language in full. It walks a complete fit end to end as a conversation with the
assistant — composing the model, defining the likelihood, choosing a search, running the fit and inspecting the
results — and shows why PyAutoFit's design suits AI agents. The workflow is:
**model → priors → likelihood → search → results → scientific workflow**, and every step can be requested in
natural language, without writing Python.

If you do want to see how **PyAutoFit** works under the hood, [The Python API](https://pyautofit.readthedocs.io/en/latest/overview/python_api.html) explains PyAutoFit's
inner workings and Python API, including the model, Analysis, search and Result objects that the assistant
uses behind the scenes.
