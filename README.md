# PyAutoFit: Scientific Inference with Natural Language

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python Versions](https://img.shields.io/pypi/pyversions/autofit)](https://pypi.org/project/autofit/)
[![PyPI Version](https://img.shields.io/pypi/v/autofit.svg)](https://pypi.org/project/autofit/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.8.1/notebooks/overview/overview_1_the_basics.ipynb)
[![Tests](https://github.com/PyAutoLabs/PyAutoFit/actions/workflows/main.yml/badge.svg)](https://github.com/PyAutoLabs/PyAutoFit/actions)
[![Build](https://github.com/PyAutoLabs/PyAutoHands/actions/workflows/release.yml/badge.svg)](https://github.com/PyAutoLabs/PyAutoHands/actions)
[![Documentation Status](https://readthedocs.org/projects/pyautofit/badge/?version=latest)](https://pyautofit.readthedocs.io/en/latest/?badge=latest)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.02550/status.svg)](https://doi.org/10.21105/joss.02550)

[AI Assistant](https://github.com/PyAutoLabs/autofit_assistant) |
[Documentation](https://pyautofit.readthedocs.io/en/latest/index.html) |
[Installation Guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) |
[Introduction on Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.8.1/notebooks/overview/overview_1_the_basics.ipynb) |
[HowToFit](https://github.com/PyAutoLabs/HowToFit)

**Bring your models, data and likelihood code. Fit models, explore results and develop your analysis through conversation.**

[**autofit_assistant**](https://github.com/PyAutoLabs/autofit_assistant) connects natural-language requests to runnable
**PyAutoFit** workflows. Ask it to compose a model, discuss priors, run inference or compare competing explanations.
You guide the science; it writes and runs Python scripts you can inspect, rerun and share.

**PyAutoFit** is a domain-agnostic Python package for scientific model fitting and Bayesian inference. It supports
nested sampling, MCMC and optimisation, alongside advanced methods such as hierarchical models, search chaining
and Bayesian model comparison. The source code has an **AI First Design**; check out the
[ReadTheDocs natural-language inference page](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html)
for the details.

## Getting Started

The following links are useful for new starters:

- **[Start with autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant#getting-started)** to build and run inference workflows using natural language. The setup guide explains supported coding agents and access requirements.
- [The PyAutoFit readthedocs](https://pyautofit.readthedocs.io/en/latest), which includes an [installation guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) and an overview of **PyAutoFit**'s core features.
- [The introduction Jupyter Notebook on Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.8.1/notebooks/overview/overview_1_the_basics.ipynb), where you can try **PyAutoFit** in a web browser (without installation).
- [The autofit_workspace GitHub repository](https://github.com/PyAutoLabs/autofit_workspace), which includes example scripts demonstrating **PyAutoFit**'s features.
- [The standalone HowToFit repository](https://github.com/PyAutoLabs/HowToFit), a series of Jupyter notebook lectures which give new users a step-by-step introduction to **PyAutoFit**.

## Support

Support for installation issues, help with Fit modeling and using **PyAutoFit** is available by
[raising an issue on the GitHub issues page](https://github.com/PyAutoLabs/PyAutoFit/issues).

We also offer support on the **PyAutoFit** [Slack channel](https://pyautoFit.slack.com/), where we also provide the
latest updates on **PyAutoFit**. Slack is invitation-only, so if you'd like to join send
an [email](https://github.com/Jammy2211) requesting an invite.

## HowToFit

For users less familiar with Bayesian inference and scientific analysis you may wish to read through
the **HowToFits** lectures. These teach you the basic principles of Bayesian inference, with the
content pitched at undergraduate level and above.

The lectures are available in the [standalone HowToFit repository](https://github.com/PyAutoLabs/HowToFit).

## Inference with Natural Language

Start with a simple example: fitting a Gaussian profile to noisy data.
Ask [autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant):

> Fit the bundled 1D Gaussian dataset. Explain the model and priors,
> use nested sampling to infer its parameters, and plot the fitted
> profile over the data.

<img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/files/toy_model_fit.png" alt="Gaussian model fitted to noisy one-dimensional data" width="400" />

Then extend the analysis through a follow-up request:

> Fit a model with two Gaussians instead. Show me the priors and
> compare the Bayesian evidence to assess whether the additional
> component is justified.

For your own science, point the assistant to your data and existing model or likelihood code.

Follow the [natural-language introduction](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html)
for the complete workflow, or see [The Python API](https://pyautofit.readthedocs.io/en/latest/overview/python_api.html)
for PyAutoFit's Python interface and inner workings.
