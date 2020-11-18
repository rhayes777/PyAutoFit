---
title: "`PyAutoFit`: A Classy Probabilistic Programming Language for Model Composition and Fitting"
tags:
  - Python
  - statistics
  - Bayesian inference
  - probabilistic programming
  - model fitting
authors:
  - name: James. W. Nightingale
    orcid: 0000-0002-8987-7401
    affiliation: 1
  - name: Richard G. Hayes
    affiliation: 1
affiliations:
  - name: Institute for Computational Cosmology, Stockton Rd, Durham, United Kingdom, DH1 3LE
    index: 1
date: 17 July 2020
codeRepository: https://github.com/rhayes777/PyAutoFit
license: MIT
bibliography: paper.bib
---

# Summary

A major trend in academia and data science is the rapid adoption of Bayesian statistics for data analysis and modelling, 
leading to the development of probabilistic programming languages (PPL). A PPL provides a framework that allows users 
to easily specify a probabilistic model and perform inference automatically. `PyAutoFit` is a Python-based PPL aimed 
at model fitting problems tackled by long-term software development projects. By interfacing with all aspects of the 
modelling (e.g. the model, data, fitting procedure, visulization, results, etc.), `PyAutoFit` provides a more complete 
management of modeling than other PPLs. This includes composing high-dimensionality models from individual model 
components, customizing the fitting procedure and performing data augmentation before a model-fit. Advanced features 
include database tools for analysing large suites of modeling results and exploiting domain-specific knowledge of a 
problem via transdimensional model-fitting pipelines. Accompanying `PyAutoFit` is 
the [autofit workspace](https://github.com/Jammy2211/autofit_workspace), which includes example scripts and 
the `HowToFit` lecture series which introduces non-experts to model-fitting and provides a guide on how to write a 
software project using `PyAutoFit`. To get started readers should go to 
our [readthedocs](https://pyautofit.readthedocs.io/en/latest/) and contact us to join 
the [PyAutoFit Slack channel](https://pyautofit.slack.com/) where we are building our online community.

# Background of Probabilistic Programming

Probabilistic programming languages (PPLs) have enabled contemporary statistical inference techniques to be applied 
to a diverse range of problems across academia and industry. Packages such as PyMC3 [@Salvatier2016], 
Pyro [@Bingham2019] and STAN [@Carpenter2017] offer general-purpose frameworks where users can specify a generative 
model and fit it to data using a variety of non-linear fitting techniques. Each package is specialized to problems 
of a certain nature, with many focused on problems like generalized linear modeling or determining the 
distribution(s) from which the data was drawn. For these problems the model is typically composed of linear equations 
which are easily expressed syntactically, such that the PPL API offers an expressive way to define the model and 
extensions can be implemented in an intuitive and straightforward way.

# Software Description

`PyAutoFit` is a PPL whose core design is providing a direct interface with the model, data, fitting procedure and 
results, allowing it to provide more complete management of the model-fitting task than other PPLs and making it 
suited to longer term software projects. Model components are written as Python classes, allowing `PyAutoFit` to 
define the model and associated parameters in an expressive way that is tied to the modeling software's API. A 
model fit then only requires that a `PyAutoFit` `Analysis` class is writen, which combines the data, model and 
likelihood function and defines how the model-fit is performed using a `NonLinearSearch` 
(e.g. `dynesty` [@dynesty], `emcee` [@emcee] or `PySwarms` [@pyswarms]). 

The `Analysis` class provides a model specific interface between `PyAutoFit` and the modeling software, allowing it 
to handle the 'heavy lifting' that comes with writing model-fitting software. This includes interfacing with the 
non-linear search, outputting results in a structured path format and model-specific visualization during and 
after the non-linear search. Results are output in a database structure with metadata that allows 
the `Aggregator` tool to load results post-analysis via a Python script or Jupyter notebook. This includes methods 
for summarizing the results of every fit, filtering results to inspect subsets of model fits and visualizing results. 
Results are loaded as `Python` generators, ensuring the `Aggregator` can be used to interpret large result datasets 
in a memory efficient way. `PyAutoFit` is therefore suited to 'big data' problems where independent fits to large 
homogeneous data-sets using an identical model-fitting procedure are performed. 

# Model Abstraction and Composition

For many modeling problems the model comprises abstract model components representing objects or processes in a 
physical system. For example, galaxy morphology studies in astrophysics where model components represent the light 
profile of stars [@Haussler2013] [@Nightingale2019]. For these problems the likelihood function is typically a 
sequence of numerical processes (e.g. convolutions, Fourier transforms, linear algebra) and extensions to the model 
often requires the addition of new model components in a way that is non-trivially included in the fitting process 
and likelihood function. Existing PPLs have tools for these problems, for example `black-box' likelihood functions 
in PyMC3. However, these solutions decouple model composition from the data and fitting procedure, making the model 
less expressive, restricting model customization and reducing flexibility in how the model-fit is performed.

By writing model components as Python classes, the model and its associated parameters are defined in an expressive 
way that is tied to the modeling software’s API. Model composition with `PyAutoFit` allows complex models to be built 
from these individual components, abstracting the details of how they change model-fitting procedure from the user. 
Models can be fully customized, allowing adjustment of individual parameter priors, the fixing or coupling of 
parameters between model components and removing regions of parameter space via parameter assertions. Adding new model 
components to a `PyAutoFit` project is straightforward, whereby adding a new Python class means it works within 
the entire modeling framework. `PyAutoFit` is therefore ideal for problems where there is a desire to compose, fit and 
compare many similar (but slightly different) models to a single dataset, with the `Aggregator` including tools to 
facilitate this. 

For many model fitting problems, domain specific knowledge of the model can be exploited to speed up the non-linear 
search and ensure it locates the global maximum likelihood solution. For example, initial fits can be performed 
using simplified model parameterizations, augmented datasets and faster non-linear fitting techniques. Through 
experience users may know that certain model components share minimal covariance, meaning that separate fits to each 
model component (in parameter spaces of reduced dimensionality) can be performed before fitting them simultaneously. 
The results of these simplified fits can then be used to initialize fits using a higher dimensionality model. 
Breaking down a model-fit in this way 
uses `PyAutoFit`'s [transdimensional model-fitting pipelines](https://pyautofit.readthedocs.io/en/latest/advanced/pipelines.html), 
which granularize the non-linear fitting procedure into a series of linked non-linear searches. Initial model-fits 
are followed by fits that gradually increase the model complexity, using the information gained throughout the 
pipeline to guide each `NonLinearSearch` and thus enable accurate fitting of models of arbitrary complexity.

# History

`PyAutoFit` is a generalization of [PyAutoLens](https://github.com/Jammy2211/PyAutoLens), an Astronomy package 
developed to analyse images of gravitationally lensed galaxies. Modeling gravitational lenses historically requires 
large amounts of human time and supervision, an approach which does not scale to the incoming samples of $100000$ 
objects. Domain exploitation enabled full automation of the lens modeling 
procedure [@Nightingale2015] [@Nightingale2018], with model customization and the aggregator enabling one to fit 
large datasets with many different models. More recently, `PyAutoFit` has been applied to calibrating radiation 
damage to charge coupled imaging devices and a model of cancer tumour growth.  
 
# Workspace and HowToFit Tutorials

`PyAutoFit` is distributed with the [autofit workspace](https://github.com/Jammy2211/autofit_workspace), which 
contains example scripts for composing a model, performing a fit and using the _Aggregator_. Also included are 
the `HowToFit` tutorials, a series of Jupyter notebooks aimed at non-experts, introducing them to model-fitting and 
Bayesian inference. They teach users how to write a `phase` package in `PyAutoFit`, which interfaces with their 
modeling software and ensures the design follows the concepts of object-oriented design which cleanly separates 
different parts of the modeling problem. The lectures can be viewed on 
our [readthedocs](https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html).

# Software Citations

`PyAutoFit` is written in Python 3.6+ [@python] and uses the following software packages:

- `corner.py` https://github.com/dfm/corner.py [@corner]
- `dynesty` https://github.com/joshspeagle/dynesty [@dynesty]
- `emcee` https://github.com/dfm/emcee [@emcee]
- `matplotlib` https://github.com/matplotlib/matplotlib [@matplotlib]
- `NumPy` https://github.com/numpy/numpy [@numpy]
- `PyMulitNest` https://github.com/JohannesBuchner/PyMultiNest [@multinest] [@pymultinest]
- `PySwarms` https://github.com/ljvmiranda921/pyswarms [@pyswarms]
- `Scipy` https://github.com/scipy/scipy [@scipy]

# Related Probabilistic Programming Languages

- `PyMC3` https://github.com/pymc-devs/pymc3 [@Salvatier2016]
- `Pyro` https://github.com/pyro-ppl/pyro [@Bingham2019]
- `STAN` https://github.com/stan-dev/stan [@Carpenter2017]
- `TensorFlow Probability` https://github.com/tensorflow/probability [@tensorflow]
- `uravu` https://github.com/arm61/uravu [@uravu]

# Acknowledgements

JWN and RJM are supported by the UK Space Agency, through grant ST/V001582/1, and by InnovateUK through grant TS/V002856/1. RGH is supported by STFC Opportunities grant ST/T002565/1.
This work used the DiRAC@Durham facility managed by the Institute for Computational Cosmology on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The equipment was funded by BEIS capital funding via STFC capital grants ST/K00042X/1, ST/P002293/1, ST/R002371/1 and ST/S002502/1, Durham University and STFC operations grant ST/R000832/1. DiRAC is part of the National e-Infrastructure.

# References
