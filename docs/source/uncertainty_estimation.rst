.. _UncertaintyQuantification:

UncertaintyQuantification
=========================


The :py:class:`uncertainpy.UncertaintyQuantification` class is used to define the problem,
perform the uncertainty quantification and sensitivity analysis,
and save and visualize the results.
``UncertaintyQuantification`` combines the three main components required to
perform an uncertainty quantification and sensitivity analysis:

    * The **model** we want to examine.
    * The **parameters** of the model.
    * Specifications of **features** in the model output.

The model and parameters are required components,
while the feature specifications are optional.

Among others, UncertaintyQuantification takes the arguments:::

    UQ = un.UncertaintyQuantification(
            model=Model(...),                        # Required
            parameters=Parameters(...),              # Required
            features=Features(...)                   # Optional
    )

The arguments are given as instances of their corresponding Uncertainpy classes
(:ref:`Models <models>`, :ref:`Parameters <parameters>`, and :ref:`Features <features>`).


After the problem is set up,
an uncertainty quantification and sensitivity analysis can be performed by using the
:py:meth:`uncertainpy.UncertaintyQuantification.quantify` method.
Among others, ``quantify`` takes the optional arguments:::

    UQ.quantify(
        method="pc"|"mc",
        pc_method="collocation"|"spectral",
        rosenblatt=False|True
    )

The `method` argument allows the user to choose whether Uncertainpy
should use polynomial chaos expansions (``"pc"``)
or quasi-Monte Carlo (``"mc"``) methods to
calculate the relevant statistical metrics.
If polynomial chaos expansions are chosen,
`pc_method` further specifies whether point collocation
(``"collocation"``) or spectral projection (``"spectral"``)
methods are used to calculate the expansion coefficients.
Finally,
`rosenblatt` (``False`` or ``True``) determines if the
Rosenblatt transformation should be used.
If nothing is specified,
Uncertainpy by default uses polynomial chaos expansions based on point
collocation without the
Rosenblatt transformation.


The results from the uncertainty quantification are automatically saved and
plotted.
After the calculations are performed the results are available in
``UQ.data`` as a :ref:`Data <data>` object.
The results are also saved in a ``data`` folder,
and figures are saved in a ``figures`` folder,
both in the current directory.


Polynomial chaos expansions are recommended as long as the number of uncertain
parameters is small (typically :math:`>20`),
as polynomial chaos expansions in these cases are much faster than
quasi-Monte Carlo methods.
Additionally,
sensitivity analysis is currently not yet available for studies based on
the quasi-Monte Carlo method.
Which of the polynomial chaos expansions methods to choose is problem dependent,
but in general the pseudo-spectral method is faster than point collocation,
but has lower stability.
We therefore generally recommend the point collocation method.


We note that there is no guarantee each set of sampled parameters produces
a valid model or feature output.
For example,
a feature such as the spike width will not be defined in a model evaluation that
produces no spikes.
In such cases,
Uncertainpy gives a warning which includes the number of runs that
failed to return a valid output,
and performs the uncertainty quantification and sensitivity analysis
using the reduced set of valid runs.
Point collocation (as well as the quasi-Monte Carlo method) are robust towards
missing values as long as the number of results remaining is high enough,
another reason the point collocation method is recommend.
However, if a large fraction of the simulations fail,
the user could consider redefining the problem
(e.g., by using narrower parameter distributions).


API Reference
-------------

.. autoclass:: uncertainpy.UncertaintyQuantification
   :members:
   :inherited-members:
   :undoc-members: