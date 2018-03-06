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

    uncertainty_estimation

    models
    parameters
    features
    data

    distribution

    plotting
    utilities

    core

    theory

Uncertainpy paper
=================

The preprint for the Uncertainpy paper can be found here `here <preprint>`_.

.. _preprint: https://www.biorxiv.org/content/early/2018/03/05/274779

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


Content of Uncertainpy
======================

This is the content of Uncertainpy and contains instructions for how to use
all classes and functions, along with their API.

* :ref:`UncertaintyQuantification <UncertaintyQuantification>`
* :ref:`Models <models>`
    * :ref:`General models <model>`
    * :ref:`Nest models <nest_model>`
    * :ref:`Neuron models <neuron_model>`
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

