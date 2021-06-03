.. _expectation_propagation:

Expectation Propagation
-----------------------

For large datasets, a graphical model may have hundreds, thousands, or *hundreds of thousands* of parameters. The
high dimensionality of such a parameter space can make it inefficient or impossible to fit the model.

Graphical models in **PyAutoFit** support the message passing framework below, which allows one to fit the local model
to every dataset individually and pass messages 'up and down' the graph to infer the global parameters efficiently.
Again, this feature is **still in beta** so contact us if you are interested in using this functionality ( https://github.com/Jammy2211 ).

https://arxiv.org/pdf/1412.4869.pdf