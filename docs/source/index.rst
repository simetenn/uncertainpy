.. image:: ../../logo/uncertainpy.png

|

Welcome to Uncertainpy's documentation!
=======================================


.. automodule:: uncertainpy

.. toctree::
    :maxdepth: 1
    :caption: Content:
    :hidden:

    installation
    quickstart

    examples

    faq

    uncertainty_estimation

    models
    parameters
    features
    data

    distribution

    plotting
    logging
    utilities

    core

    theory

Uncertainpy paper
=================

The Uncertainpy paper can be found here: `Tennøe S, Halnes G, and Einevoll GT (2018) Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience. Front. Neuroinform. 12:49. doi: 10.3389/fninf.2018.00049 <paper>`_.

.. _paper: https://www.frontiersin.org/articles/10.3389/fninf.2018.00049/full

Getting started
===============

* :ref:`Installation <installation>`
* :ref:`Quickstart <quickstart>`


Examples
========

This is a collection of examples that shows the use of Uncertainpy for a few
different case studies.

* :ref:`A simple cooling coffee cup <coffee_cup>`
* :ref:`A cooling coffee cup with dependent parameters <coffee_dependent>`
* :ref:`The Hodgkin-Huxley model <hodgkin_huxley>`
* :ref:`A multi-compartment model of a thalamic interneuron <interneuron>`
* :ref:`A sparsely connected recurrent network <brunel>`


Frequently asked questions
==========================

* :ref:`A simple cooling coffee cup <coffee_cup>`



Content of Uncertainpy
======================

This is the content of Uncertainpy and contains instructions for how to use
all classes and functions, along with their API.

* :ref:`UncertaintyQuantification <UncertaintyQuantification>`
* :ref:`Models <models>`
    * :ref:`General models <model>`
    * :ref:`Nest models <nest_model>`
    * :ref:`Neuron models <neuron_model>`
    * :ref:`Multiple model outputs <multiple_outputs>`
* :ref:`Parameters <parameters>`
* :ref:`Features <features>`
    * :ref:`General features <main_features>`
    * :ref:`Spiking features <spiking>`
    * :ref:`Spikes <spikes>` (used by the spiking features)
    * :ref:`Electrophys Feature Extraction Library (eFEL) features <efel>`
    * :ref:`Network features <network>`
    * :ref:`General spiking features <general_spiking>`
    * :ref:`General network features <general_network>`
* :ref:`Data <data>`
* :ref:`Utility distributions <distributions>`
* :ref:`Plotting <plotting>`
* :ref:`Logging <logging>`
* :ref:`Utilities <utilities>`
* :ref:`Core <core>`
    * :ref:`Base classes <base>`
    * :ref:`Parallel <parallel>`
    * :ref:`Run model <run_model>`
    * :ref:`Uncertainty calculations<uncertainty_calculations>`


Theory
======

Here we give an overview of the theory behind uncertainty quantification and
sensitivity analysis with a focus on (quasi-)Monte Carlo methods and polynomial
chaos expansions, the methods implemented in Uncertainpy.


* :ref:`Theory on uncertainty quantification and sensitivity analysis <theory>`
    * :ref:`Problem definition <problem>`
    * :ref:`Uncertainty quantification <uq>`
    * :ref:`Sensitivity analysis <sa>`
    * :ref:`(Quasi-)Monte Carlo methods <qmc>`
    * :ref:`Polynomial chaos expansions <pce>`
    * :ref:`Dependency between uncertain parameters <rosenblatt>`

