class Model(object):
    """
run(**parameters) must return t and U
    """
    def __init__(self,
                 run_function=None,
                 adaptive_model=False,
                 labels=[]):
        """
        """

        self.adaptive_model = adaptive_model
        self.labels = labels

        # TODO must either remove this or implement a permanent solution for setting adaptive_model = true when having a function as a model
        if run_function is not None:
            self.run = run_function
        else:
            self.name = self.__class__.__name__


    @property
    def run(self):
        return self._run


    @run.setter
    def run(self, new_run_function):
        if not callable(new_run_function):
            raise TypeError("run function must be callable")

        self._run = new_run_function
        self.name = new_run_function.__name__


    def set_parameters(self, **parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])


    def _run(self, **parameters):
        """
        Run must return t, U
        """
        raise NotImplementedError("No run() method implemented")


    def postprocess(self, t, U):
        return t, U