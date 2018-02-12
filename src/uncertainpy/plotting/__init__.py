"""
This module contains ``PlotUncertainpy``, which  creates plot of the data from
the uncertainty quantification and sensitivity analysis. ``PlotUncertainpy``
plots the results for all zero and one dimensional statistical metrics, and some
of the two dimensional statistical metrics It is intended as a quick way to get
an overview of the data, and does not create publication ready plots. Custom
plots of the data can easily be created by retrieving the results from the
``Data`` class.
"""

__all__ = ["PlotUncertainty"]

from .plot_uncertainty import PlotUncertainty