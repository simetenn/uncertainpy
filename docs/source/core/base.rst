.. _Base:

Base and ParameterBase
======================

These classes enable setting and updating the model, features and parameters
(not in all classes) across classes from the top of the hierarchy
(:ref:`UncertaintyQuantification <UncertaintyQuantification>`) and down
(:ref:`Parallel <parallel>`).
To add updating of the current class, as well as the classes further down the
setters can be overridden.
One example of this from :ref:`RunModel <run_model>`)::

    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self._parallel.model = self.model





API Reference
-------------

Base
....

.. autoclass:: uncertainpy.core.Base
   :members:
   :inherited-members:

ParameterBase
.............

.. autoclass:: uncertainpy.core.ParameterBase
   :members:
   :inherited-members:
